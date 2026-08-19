"""
Occlusion-aware reachability prior on the metric floor.

Why this module exists
----------------------
`scene_geometry` recovers *where the floor is*; `scene_depth` recovers *what is
standing on it*. Both were, until now, used only for calibration diagnostics -
the live association logic never saw them. Track rebinding scored a candidate
with an isotropic Gaussian around ``last_bbox_centre + velocity * elapsed``, in
**pixels**. That prior is wrong in three ways on a ceiling camera:

* a pixel is worth wildly different metres near the lens and at the far wall;
* the Gaussian spreads probability straight *through* solid furniture, so a
  person who stepped behind a cabinet is scored as if they could re-emerge in
  the middle of it;
* its sigma grows without bound, so after a few seconds in a small room it is
  uniform - it stops constraining anything exactly when appearance is weakest.

What replaces it is the obvious physical statement: a person who vanished at
one spot and re-appeared at another had to *walk between them, around the
furniture*. So the prior is a geodesic distance over the walkable floor,
compared against a walking-speed budget.

The walkable map is not the depth occluder map
----------------------------------------------
The natural move is to reuse ``SceneModel.free_space``. Measured on the
reference clip that would have been a serious mistake: the depth-derived floor
mask contains only **~25 % of the floor cells people were actually observed
standing in**. A prior built on it would veto most true rebinds. The tallest-
surface-per-cell reduction behind ``obstacle`` is not robust - one grazing wall
pixel condemns a cell - and 18 cm of depth residual smears furniture edges.

So the map here is built from *two* independent evidences and keeps a third
state for ignorance:

* **FREE**    - the depth map sees floor here, *or* feet were observed here.
* **BLOCKED** - the depth map sees a tall solid thing here (no floor pixels,
                mostly >0.8 m) and no feet were ever observed here.
* **UNKNOWN** - everything else. Traversable at a penalty, never free.

Keeping UNKNOWN distinct from BLOCKED is what makes this safe to switch on
before a site has accumulated footage: with no evidence at all every cell is
UNKNOWN, the prior degrades to a plain metric distance budget, and that is
already strictly better than the pixel Gaussian it replaces. As tracking runs,
observed footfall promotes cells to FREE and the map sharpens itself. No
annotation, at any point.

Validation of the BLOCKED tier on the reference clip: it covers 10.1 m^2 and
contains **0.0 %** of all feet-*visible* observations, while containing ~16 %
of head-inferred ones - i.e. exactly the people standing behind the furniture,
whose inferred position lands on its footprint. Query points are therefore
snapped to the nearest walkable cell rather than rejected.

Conventions follow `scene_depth`: floor coordinates in metres, same origin and
sign as ``GroundPlane.floor_xy`` / ``scene_depth.cam_point``.
"""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:                                             # optional, only for closing
    from scipy import ndimage as _ndi
except Exception:                                # pragma: no cover
    _ndi = None


# ── tiers ────────────────────────────────────────────────────────────────────
FREE, UNKNOWN, BLOCKED = 0, 1, 2

FLOOR_TOL_M = 0.20          # matches scene_depth.FLOOR_TOL_M
TALL_M = 0.80               # "clearly a solid thing", not a step-over

# Build thresholds. Deliberately asymmetric: cheap to call a cell FREE,
# expensive to call it BLOCKED, because a false BLOCKED vetoes a true rebind.
FLOOR_FRAC_FREE = 0.30      # >=30% of the cell's surface points are at floor level
FLOOR_FRAC_BLOCK = 0.05     # <5% floor, i.e. essentially none
TALL_FRAC_BLOCK = 0.70      # >=70% of them are above TALL_M
MIN_FOOT_OBS = 3            # feet-visible observations that promote a cell to FREE

# Motion model. Values are the ones bench/08 selected on the reference clip.
V_MAX_MS = 2.2              # hard feasibility bound (brisk indoor walk/jog)
V_TYP_MS = 0.6              # how fast the position belief spreads, m/s
LOC_SIGMA_M = 0.08          # irreducible localisation noise, floors the spread
SLACK_M = 1.0               # metres added to the reachability budget; 0.6 cost
                            # 3.2% of true continuations at short gaps, 1.0
                            # costs 1.5% while still catching 27% of impostors
P_FLOOR = 0.15              # reachable-but-unpredicted still gets this much prior
RAMP_TAU_S = 1.5            # p_floor and the portal bonus fade in over this;
                            # at short gaps the pixel prior is sharp and these
                            # constants must not be allowed to outrank it
UNKNOWN_COST = 2.0          # traversal penalty for a cell with no evidence
HEAD_LOOKAHEAD_S = 0.6      # how long ballistic extrapolation is trusted

# How far a lost person actually gets. The obvious model is ballistic - they
# walk away at v_typ, so the belief spreads by v_typ * elapsed. Measured on
# gunsan_test (unsupervised: displacement over a lag *inside* tracklets, no
# labels), that is wrong by more than an order of magnitude at long gaps:
#
#     elapsed    rms displacement    ballistic (0.6 m/s)
#       1.3 s          0.50 m              0.80 m
#       5.3 s          1.07 m              3.20 m
#      40.0 s          1.68 m             24.00 m
#
# A fit gives sigma(t) = 0.43 * t**0.43, i.e. essentially DIFFUSIVE (exponent
# 0.5) rather than ballistic (1.0). People in a room mill about; they do not
# depart in a straight line. A ballistic sigma makes the prior 14x too loose at
# 40 s, which flattens it into uselessness exactly when discrimination is most
# needed. These are defaults, not constants - `Reachability.fit_spread` can
# re-fit them per site without annotation.
SPREAD_A = 0.45             # metres at t = 1 s
SPREAD_B = 0.5              # diffusive exponent


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Reachability:
    """Walkable floor as a cost grid, plus geodesic queries over it."""

    cell_m: float
    origin: np.ndarray                  # (2,) floor coords of cell (0,0)
    tier: np.ndarray                    # (ny,nx) uint8, FREE/UNKNOWN/BLOCKED
    foot_obs: np.ndarray                # (ny,nx) int32, feet-visible observations
    _fields: dict = field(default_factory=dict, repr=False)
    _region: np.ndarray | None = field(default=None, repr=False)

    # ── construction ────────────────────────────────────────────────────────
    @staticmethod
    def build(scene, foot_xy: np.ndarray | None = None,
              close_iters: int = 2) -> "Reachability":
        """From a `scene_depth.SceneModel`, optionally seeded with observed
        floor positions of *feet-visible* detections (the only ones that are
        measurements rather than inferences)."""
        ny, nx = scene.obstacle.shape
        cell_m, origin = float(scene.cell_m), np.asarray(scene.origin, float)

        gx = ((scene.xy[..., 0] - origin[0]) / cell_m).astype(int)
        gy = ((scene.xy[..., 1] - origin[1]) / cell_m).astype(int)
        ok = (np.isfinite(scene.height) & (gx >= 0) & (gx < nx)
              & (gy >= 0) & (gy < ny) & (scene.height < 3.0))
        idx, h = gy[ok] * nx + gx[ok], scene.height[ok]
        tot = np.bincount(idx, minlength=ny * nx)
        flo = np.bincount(idx[np.abs(h) < FLOOR_TOL_M], minlength=ny * nx)
        tall = np.bincount(idx[h > TALL_M], minlength=ny * nx)
        seen = tot > 0
        frac = np.where(seen, flo / np.maximum(tot, 1), 0.0)
        tfrac = np.where(seen, tall / np.maximum(tot, 1), 0.0)

        obs = np.zeros(ny * nx, np.int32)
        if foot_xy is not None and len(foot_xy):
            r = Reachability(cell_m, origin, np.zeros((ny, nx), np.uint8), obs.reshape(ny, nx))
            cells = r._cells(np.asarray(foot_xy, float))
            good = cells >= 0
            np.add.at(obs, cells[good], 1)

        free = (frac >= FLOOR_FRAC_FREE) | (obs >= MIN_FOOT_OBS)
        blocked = (seen & (frac < FLOOR_FRAC_BLOCK)
                   & (tfrac >= TALL_FRAC_BLOCK) & (obs == 0))

        free = free.reshape(ny, nx)
        blocked = blocked.reshape(ny, nx)
        if _ndi is not None and close_iters:
            # Close pinholes in the free space so the geodesic graph is not
            # fragmented by single missing cells; never at a BLOCKED cell's
            # expense, which stays authoritative.
            st = np.ones((3, 3), bool)
            free = _ndi.binary_closing(free, structure=st, iterations=close_iters)
            free &= ~blocked

        tier = np.full((ny, nx), UNKNOWN, np.uint8)
        tier[blocked] = BLOCKED
        tier[free] = FREE
        return Reachability(cell_m, origin, tier, obs.reshape(ny, nx))

    # ── grid helpers ────────────────────────────────────────────────────────
    @property
    def shape(self) -> tuple[int, int]:
        return self.tier.shape

    @property
    def walkable(self) -> np.ndarray:
        return self.tier != BLOCKED

    def _cells(self, xy: np.ndarray) -> np.ndarray:
        """(N,2) floor metres -> flat cell index, -1 when outside/NaN."""
        ny, nx = self.tier.shape
        xy = np.atleast_2d(np.asarray(xy, float))
        gx = (xy[:, 0] - self.origin[0]) / self.cell_m
        gy = (xy[:, 1] - self.origin[1]) / self.cell_m
        ok = (np.isfinite(gx) & np.isfinite(gy) & (gx >= 0) & (gx < nx)
              & (gy >= 0) & (gy < ny))
        out = np.full(len(xy), -1, np.int64)
        out[ok] = gy[ok].astype(int) * nx + gx[ok].astype(int)
        return out

    def cell_of(self, xy) -> int:
        """Single point -> walkable cell index, snapping out of BLOCKED cells.

        A person occluded by a desk is localised from the head, and that
        estimate lands on the desk's footprint often enough that rejecting it
        would throw away the very observations this module exists to serve.
        Snapping to the nearest walkable cell is the honest repair.
        """
        c = int(self._cells(np.asarray(xy, float).reshape(1, 2))[0])
        if c < 0:
            return -1
        if self.tier.flat[c] != BLOCKED:
            return c
        return self._snap(c)

    def _snap(self, c: int, max_r: int = 8) -> int:
        ny, nx = self.tier.shape
        cy, cx = divmod(c, nx)
        for r in range(1, max_r + 1):
            y0, y1 = max(cy - r, 0), min(cy + r + 1, ny)
            x0, x1 = max(cx - r, 0), min(cx + r + 1, nx)
            sub = self.tier[y0:y1, x0:x1]
            cand = np.argwhere(sub != BLOCKED)
            if len(cand):
                d = ((cand[:, 0] + y0 - cy) ** 2 + (cand[:, 1] + x0 - cx) ** 2)
                b = cand[int(np.argmin(d))]
                return int((b[0] + y0) * nx + (b[1] + x0))
        return -1

    def centre_of(self, c: int) -> np.ndarray:
        ny, nx = self.tier.shape
        cy, cx = divmod(int(c), nx)
        return self.origin + (np.array([cx, cy], float) + 0.5) * self.cell_m

    # ── geodesic ────────────────────────────────────────────────────────────
    def field(self, c: int) -> np.ndarray:
        """Geodesic distance in metres from cell ``c`` to every cell.

        Cached: the field depends only on the source, not on elapsed time, and
        there are at most a few thousand cells - so every source is computed
        at most once for the whole session.
        """
        if c in self._fields:
            return self._fields[c]
        ny, nx = self.tier.shape
        cost = np.where(self.tier == UNKNOWN, UNKNOWN_COST, 1.0).ravel()
        blocked = (self.tier == BLOCKED).ravel()
        dist = np.full(ny * nx, np.inf)
        if 0 <= c < ny * nx and not blocked[c]:
            dist[c] = 0.0
            s, d = self.cell_m, self.cell_m * np.sqrt(2.0)
            steps = ((-1, 0, s), (1, 0, s), (0, -1, s), (0, 1, s),
                     (-1, -1, d), (-1, 1, d), (1, -1, d), (1, 1, d))
            pq = [(0.0, c)]
            while pq:
                du, u = heapq.heappop(pq)
                if du > dist[u]:
                    continue
                uy, ux = divmod(u, nx)
                for dy, dx, step in steps:
                    vy, vx = uy + dy, ux + dx
                    if not (0 <= vy < ny and 0 <= vx < nx):
                        continue
                    v = vy * nx + vx
                    if blocked[v]:
                        continue
                    dv = du + step * 0.5 * (cost[u] + cost[v])
                    if dv < dist[v]:
                        dist[v] = dv
                        heapq.heappush(pq, (dv, v))
        dist = dist.reshape(ny, nx)
        if len(self._fields) > 4096:
            self._fields.clear()
        self._fields[c] = dist
        return dist

    def distance(self, xy_a, xy_b) -> float:
        """Geodesic walking distance in metres, inf if not connected."""
        a, b = self.cell_of(xy_a), self.cell_of(xy_b)
        if a < 0 or b < 0:
            return float("inf")
        return float(self.field(a).flat[b])

    # ── hidden regions ("went in here, must come out there") ────────────────
    @property
    def regions(self) -> np.ndarray:
        """Connected components of BLOCKED cells; 0 = walkable. Each component
        is one physical occluder, and its walkable perimeter is the set of
        places a person who disappeared behind it can re-emerge."""
        if self._region is None:
            blocked = self.tier == BLOCKED
            if _ndi is None:
                self._region = blocked.astype(np.int32)
            else:
                lab, _ = _ndi.label(blocked, structure=np.ones((3, 3), bool))
                self._region = lab.astype(np.int32)
        return self._region

    def region_near(self, xy, radius_m: float = 0.5) -> int:
        """Which occluder a point is standing against, 0 if in the open.

        This is the explicit form of the intuition the geodesic prior encodes
        implicitly: a track that dies at a cabinet's edge has *entered* that
        cabinet's hidden region, and a track born at the same cabinet's edge is
        a strong candidate to be the same person coming out the other side.
        """
        c = self.cell_of(xy)
        if c < 0:
            return 0
        ny, nx = self.tier.shape
        r = max(1, int(round(radius_m / self.cell_m)))
        cy, cx = divmod(c, nx)
        sub = self.regions[max(cy - r, 0):cy + r + 1, max(cx - r, 0):cx + r + 1]
        lab = sub[sub > 0]
        if not len(lab):
            return 0
        vals, cnt = np.unique(lab, return_counts=True)
        return int(vals[int(np.argmax(cnt))])

    # ── online learning ─────────────────────────────────────────────────────
    def observe(self, xy) -> None:
        """Record feet-*visible* floor positions. Cells crossed often enough
        are promoted to FREE, which is how the map improves itself on site
        without anyone drawing anything."""
        cells = self._cells(np.atleast_2d(np.asarray(xy, float)))
        cells = cells[cells >= 0]
        if not len(cells):
            return
        np.add.at(self.foot_obs.reshape(-1), cells, 1)
        promote = np.unique(cells[self.foot_obs.reshape(-1)[cells] >= MIN_FOOT_OBS])
        promote = promote[self.tier.reshape(-1)[promote] != FREE]
        if len(promote):
            self.tier.reshape(-1)[promote] = FREE
            self._fields.clear()          # geometry changed, caches are stale
            self._region = None

    # ── persistence ─────────────────────────────────────────────────────────
    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, tier=self.tier, foot_obs=self.foot_obs,
                            meta=np.frombuffer(json.dumps(
                                {"cell_m": self.cell_m,
                                 "origin": self.origin.tolist()}).encode(),
                                dtype=np.uint8))

    @staticmethod
    def load(path) -> "Reachability":
        z = np.load(str(path))
        m = json.loads(z["meta"].tobytes().decode())
        return Reachability(float(m["cell_m"]), np.array(m["origin"], float),
                            z["tier"].astype(np.uint8),
                            z["foot_obs"].astype(np.int32))

    def summary(self) -> str:
        a = self.cell_m ** 2
        n = (self.tier == FREE).sum(), (self.tier == UNKNOWN).sum(), (self.tier == BLOCKED).sum()
        nreg = int(self.regions.max())
        return (f"Reachability[free={n[0] * a:.1f}m2 unknown={n[1] * a:.1f}m2 "
                f"blocked={n[2] * a:.1f}m2 occluders={nreg} cell={self.cell_m}m]")


# ─────────────────────────────────────────────────────────────────────────────
# The prior
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ReachParams:
    v_max: float = V_MAX_MS
    v_typ: float = V_TYP_MS
    loc_sigma_m: float = LOC_SIGMA_M
    slack_m: float = SLACK_M
    p_floor: float = P_FLOOR
    ramp_tau_s: float = RAMP_TAU_S
    lookahead_s: float = HEAD_LOOKAHEAD_S
    portal_bonus: float = 0.85          # prior floor for same-hidden-region pairs
    spread_a: float = SPREAD_A          # sigma(t) = spread_a * t**spread_b
    spread_b: float = SPREAD_B          # 1.0 = ballistic, 0.5 = diffusive
    ballistic_spread: bool = False      # True restores the pre-measurement
                                        # sigma = v_typ * t, kept for A/B


def rebind_prior(reach: Reachability, last_xy, vel_xy, new_xy,
                 elapsed_s: float, p: ReachParams | None = None) -> dict:
    """Geodesic replacement for the pixel-space Gaussian spatial prior.

    Returns ``{"prior", "geo_m", "feasible", "portal"}``.

    Three separable statements, deliberately not collapsed into one number:

    ``feasible``
        Could the person have *walked* from where they vanished to where this
        candidate appeared, around the furniture, in the time available? This
        is a hard physical constraint, and a violation of it should veto a
        rebind more forcefully than any appearance disagreement.

    ``prior``
        Among the feasible places, how well does this one match where they
        were heading? Ballistic extrapolation is trusted for ``lookahead_s``
        and no longer - past that, a person in a room has turned. The floor
        ``p_floor`` keeps reachable-but-unpredicted candidates alive, because
        after a long gap a random walk in a small room *is* nearly uniform
        over what it can reach, and pretending otherwise is what made the old
        prior peak on a fiction.

    ``portal``
        Whether both ends sit against the same occluder - "went in this side,
        came out that side". Scored separately because it survives long gaps
        that flatten the distance term.
    """
    p = p or ReachParams()
    out = {"prior": 0.0, "geo_m": float("nan"), "feasible": False, "portal": 0.0}

    a = reach.cell_of(last_xy)
    b = reach.cell_of(new_xy)
    if a < 0 or b < 0:
        out["prior"] = p.p_floor          # off-map: abstain rather than veto
        out["feasible"] = True
        return out

    geo = float(reach.field(a).flat[b])
    out["geo_m"] = geo
    budget = p.v_max * max(elapsed_s, 0.0) + p.slack_m
    out["feasible"] = bool(np.isfinite(geo) and geo <= budget)
    if not out["feasible"]:
        return out                         # unreachable: prior stays 0

    # Heading-aware concentration, measured from the extrapolated point.
    lead = min(max(elapsed_s, 0.0), p.lookahead_s)
    pred = np.asarray(last_xy, float) + np.asarray(vel_xy, float) * lead
    c = reach.cell_of(pred)
    d = float(reach.field(c).flat[b]) if c >= 0 else geo
    if not np.isfinite(d):
        d = geo
    t = max(elapsed_s, 0.0)
    spread = (p.v_typ * t if p.ballistic_spread
              else p.spread_a * t ** p.spread_b)
    sigma = float(np.hypot(p.loc_sigma_m, spread))
    conc = float(np.exp(-0.5 * (d / sigma) ** 2))

    ra = reach.region_near(last_xy)
    rb = reach.region_near(new_xy)
    out["portal"] = 1.0 if (ra > 0 and ra == rb) else 0.0

    # Both constants say "position has stopped being predictive"; that is only
    # true once enough time has passed, so they fade in rather than applying
    # from frame one, where they would drown a still-sharp ballistic estimate.
    ramp = 1.0 - float(np.exp(-max(elapsed_s, 0.0) / max(p.ramp_tau_s, 1e-6)))
    out["prior"] = max(conc, p.p_floor * ramp,
                       p.portal_bonus * ramp * out["portal"])
    return out
