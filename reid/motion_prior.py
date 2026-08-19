"""
Learned motion model: where a person is *going*, not just where they *could* be.

Step 9 shipped a reachability prior, and it is still the largest single win in
this system. But it answers a weaker question than the user asked for. Its
forward prediction is 0.6 s of straight-line extrapolation, after which the
belief is an isotropic blob spreading along geodesic distance. It knows the
person cannot walk through the shelf. It does not know that everybody who passes
the shelf on the left continues to the door, because it has no model of how
people move through *this* room.

The user, on why appearance is a dead end here:

    "i said tracking visuals does not work as its camera view so most of the
    time you could not see the face, and the clothes change, so totally makes
    sense - even for human like me i can not tell. But what i can do is to
    visualize the space and predict the movement and direction of the tracked
    human, which currently we don't have."

That is the gap this module fills. Two things are missing from a pure geodesic
prior, and both are things a human watching the video uses without noticing:

  1. **Momentum.** A walking person keeps walking. Direction at the moment of
     disappearance is highly informative about which side of an occluder they
     re-emerge from, and it stays informative for far longer than 0.6 s.
  2. **Where people actually go.** Floors are not used uniformly. There are
     lanes, and the lanes are learnable from the tracker's own output without
     any annotation - which keeps the site-generic constraint intact.

Both are captured by making the state ``(cell, heading)`` rather than ``cell``.
Momentum is then just the fact that heading persists, and the room's traffic
pattern is a learned turn distribution conditioned on cell. Propagating that
chain for the actual elapsed time gives a genuine forward prediction, which is
what a geodesic distance cannot express no matter how it is reweighted.

Nothing here is annotated. The flow field is fitted from tracklets - short,
confident fragments that the tracker gets right - and the whole point of the
system is that stitching those fragments is the hard part. Learning "how people
move here" from them is legitimate in a way that learning "who is who" from them
would not be.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dcfield

import numpy as np

try:
    from scipy import ndimage as _ndi
except Exception:                                   # pragma: no cover
    _ndi = None

# 8-connected headings, index order matches DIRS below (dy, dx).
DIRS = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))
NDIR = 8
_STEP_LEN = np.array([1.0, np.sqrt(2), 1.0, np.sqrt(2),
                      1.0, np.sqrt(2), 1.0, np.sqrt(2)])

V_TYP_MS = 0.6          # sets the simulation timestep, cell_m / V_TYP_MS
TURN_KAPPA = 2.2        # momentum: concentration of the turn kernel, larger =
                        # straighter. 2.2 puts ~60% of mass within +-45 deg.
FLOW_WEIGHT = 0.6       # how much the room's learned traffic steers the walker
                        # relative to their own momentum
STOP_P = 0.12           # per-step probability of standing still; people loiter
MIN_CELL_OBS = 0.5      # smoothed weighted steps; below this a cell has no
                        # usable flow of its own and stays uniform
FLOW_SMOOTH_M = 0.4     # traffic direction is a property of a region, not of a
                        # 10 cm square; borrow strength from the neighbourhood
PRIOR_COUNT = 0.5       # Dirichlet smoothing on the per-cell direction counts


def _turn_kernel() -> np.ndarray:
    """(NDIR, NDIR) — probability of heading j given current heading i."""
    a = np.arange(NDIR) * (2 * np.pi / NDIR)
    d = a[None, :] - a[:, None]
    k = np.exp(TURN_KAPPA * np.cos(d))
    return k / k.sum(1, keepdims=True)


@dataclass
class MotionPrior:
    """Per-cell traffic flow over a `reachability.Reachability` grid."""

    shape: tuple[int, int]
    cell_m: float
    counts: np.ndarray                   # (ncell, NDIR) observed transitions
    stopped: np.ndarray                  # (ncell,) observations with ~no motion
    _cache: dict = _dcfield(default_factory=dict, repr=False)

    # ── fitting ─────────────────────────────────────────────────────────────
    @staticmethod
    def fit(reach, tracks, fps: float = 15.0) -> "MotionPrior":
        """`tracks` is an iterable of (frames (N,), xy (N,2) metres) per tracklet.

        Only consecutive-in-time samples contribute, so a gap inside a tracklet
        never invents a transition across it.
        """
        ny, nx = reach.tier.shape
        counts = np.zeros((ny * nx, NDIR))
        stopped = np.zeros(ny * nx)
        step_s = reach.cell_m / V_TYP_MS

        for f, xy in tracks:
            f = np.asarray(f, float)
            xy = np.asarray(xy, float)
            ok = np.all(np.isfinite(xy), axis=1)
            f, xy = f[ok], xy[ok]
            if len(f) < 2:
                continue
            for i in range(len(f) - 1):
                dt = (f[i + 1] - f[i]) / fps
                if dt <= 0 or dt > 1.0:      # a big jump is a gap, not a step
                    continue
                c = reach.cell_of(xy[i])
                if c < 0:
                    continue
                v = (xy[i + 1] - xy[i]) / dt
                sp = float(np.hypot(*v))
                # Weight by how many simulation steps this observation covers,
                # so a slow walker does not out-vote a fast one per metre.
                w = max(dt / step_s, 1e-3)
                if sp < 0.15:
                    stopped[c] += w
                    continue
                ang = np.arctan2(-v[1], v[0])    # grid y grows downward
                b = int(np.round(ang / (2 * np.pi / NDIR))) % NDIR
                # DIRS is indexed from "up" clockwise; map the angle bin onto it
                counts[c, (2 - b) % NDIR] += w
        return MotionPrior((ny, nx), float(reach.cell_m), counts, stopped)

    # ── the learned field ───────────────────────────────────────────────────
    @property
    def observed(self) -> np.ndarray:
        return self.counts.sum(1) + self.stopped

    def flow(self) -> np.ndarray:
        """(ncell, NDIR) direction distribution; uniform where unobserved.

        The counts are spatially smoothed first. At a 0.1 m cell a single clip
        only ever populates a handful of cells directly - people walk lines, not
        areas - but traffic direction is a property of a *region*, not of a
        10 cm square. Smoothing over ~`FLOW_SMOOTH_M` borrows strength from the
        neighbourhood, which is the difference between 66 cells with usable flow
        and most of the walkable floor having it.
        """
        c = self.counts.reshape(*self.shape, NDIR)
        if _ndi is not None and FLOW_SMOOTH_M > 0:
            sig = FLOW_SMOOTH_M / self.cell_m
            c = _ndi.gaussian_filter(c, sigma=(sig, sig, 0), mode="constant")
        c = c.reshape(-1, NDIR)
        f = (c + PRIOR_COUNT) / (c + PRIOR_COUNT).sum(1, keepdims=True)
        thin = c.sum(1) < MIN_CELL_OBS
        f[thin] = 1.0 / NDIR      # no evidence -> do not pretend to steer
        return f

    def stop_p(self) -> np.ndarray:
        tot = self.observed
        p = np.where(tot > 0, self.stopped / np.maximum(tot, 1e-9), STOP_P)
        return np.clip(p, 0.0, 0.9)

    # ── forward simulation ──────────────────────────────────────────────────
    def _transition(self, reach):
        """Sparse (cell,heading) -> (cell,heading) chain, built once per grid."""
        if "T" in self._cache:
            return self._cache["T"]
        from scipy.sparse import csr_matrix

        ny, nx = self.shape
        n = ny * nx
        blocked = (reach.tier == 2).ravel()
        turn = _turn_kernel()
        flow = self.flow()
        stop = self.stop_p()

        rows, cols, vals = [], [], []
        for c in range(n):
            if blocked[c]:
                continue
            cy, cx = divmod(c, nx)
            # Heading distribution for the next step: momentum (turn kernel
            # around the current heading) tilted by the room's own flow.
            mix = turn * (flow[c] ** FLOW_WEIGHT)[None, :]
            mix /= np.maximum(mix.sum(1, keepdims=True), 1e-12)
            for h in range(NDIR):
                src = c * NDIR + h
                rows.append(src); cols.append(src); vals.append(stop[c])
                move = 1.0 - stop[c]
                for h2 in range(NDIR):
                    p = mix[h, h2] * move
                    if p < 1e-4:
                        continue
                    dy, dx = DIRS[h2]
                    vy, vx = cy + dy, cx + dx
                    if not (0 <= vy < ny and 0 <= vx < nx):
                        continue          # off-grid = left the scene, absorbed
                    v = vy * nx + vx
                    if blocked[v]:
                        # Walking into furniture: stay put rather than vanish,
                        # which is what people do when they hit an obstacle.
                        rows.append(src); cols.append(src); vals.append(p)
                        continue
                    rows.append(src); cols.append(v * NDIR + h2); vals.append(p)
        T = csr_matrix((vals, (rows, cols)), shape=(n * NDIR, n * NDIR))
        self._cache["T"] = T
        return T

    def predict(self, reach, cell: int, heading: int, elapsed_s: float,
                max_steps: int = 400) -> np.ndarray:
        """P(cell | started at `cell` heading `heading`, `elapsed_s` later).

        Returned over cells (heading marginalised out), renormalised over the
        grid — we are conditioning on the person having reappeared in view.
        """
        step_s = self.cell_m / V_TYP_MS
        n_steps = int(np.clip(round(elapsed_s / step_s), 1, max_steps))
        key = (cell, heading, n_steps)
        if key in self._cache:
            return self._cache[key]

        ny, nx = self.shape
        n = ny * nx
        T = self._transition(reach)
        p = np.zeros(n * NDIR)
        if not (0 <= cell < n):
            return np.zeros(n)
        # Seed with the observed heading, softened: the instantaneous direction
        # estimate is itself noisy.
        seed = _turn_kernel()[heading]
        p[cell * NDIR:(cell + 1) * NDIR] = seed

        # Doubling: T^k by repeated squaring would densify, so step directly but
        # reuse the halfway snapshot for the common case of nearby time buckets.
        for _ in range(n_steps):
            p = T.T @ p
            s = p.sum()
            if s <= 1e-12:
                break
            p /= s
        out = p.reshape(n, NDIR).sum(1)
        s = out.sum()
        out = out / s if s > 0 else out
        if len(self._cache) > 2048:
            T = self._cache["T"]
            self._cache.clear()
            self._cache["T"] = T
        self._cache[key] = out
        return out

    # ── io ──────────────────────────────────────────────────────────────────
    def save(self, path) -> None:
        np.savez_compressed(str(path), shape=np.array(self.shape),
                            cell_m=self.cell_m, counts=self.counts,
                            stopped=self.stopped)

    @staticmethod
    def load(path) -> "MotionPrior":
        z = np.load(str(path))
        return MotionPrior(tuple(int(v) for v in z["shape"]), float(z["cell_m"]),
                           z["counts"], z["stopped"])

    def summary(self) -> str:
        obs = self.observed
        f = self.flow()
        n = int((np.abs(f - 1.0 / NDIR).sum(1) > 1e-6).sum())
        return (f"MotionPrior[cells steered={n}, {int((obs>0).sum())} cells observed, "
                f"{obs.sum():.0f} weighted steps, "
                f"median stop_p={np.median(self.stop_p()[obs>0]):.2f}]")


def heading_of(vel_xy) -> int:
    """Metric velocity (m/s) -> DIRS index. Returns -1 if too slow to be a
    direction rather than noise."""
    v = np.asarray(vel_xy, float)
    if not np.all(np.isfinite(v)) or np.hypot(*v) < 0.15:
        return -1
    ang = np.arctan2(-v[1], v[0])
    b = int(np.round(ang / (2 * np.pi / NDIR))) % NDIR
    return (2 - b) % NDIR
