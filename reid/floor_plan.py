"""
Hand-drawn walkable floor and obstacle footprints.

Why this exists
---------------
`reachability` builds its walkable map from two evidences: the depth-derived
occluder map, and cells where feet were actually observed. Both are weak where
it matters most. On `office_cam1` the monocular depth is off by **152 cm
(1 sigma) on pixels that are known floor**, while `scene_depth` separates floor
from obstacle at a 20 cm threshold -- a signal-to-noise ratio around 0.13. The
consequence is measurable in the built map:

    office_cam1   FREE 30.4 m2 (7.9%)   UNKNOWN 341.2 m2 (88.6%)   BLOCKED 13.6 m2 (3.5%)

UNKNOWN is traversable at a penalty, so on 96.5% of that room the geodesic
prior degrades to a plain distance budget that does not know furniture is
solid. That is exactly the regime where appearance has to carry the decision
alone, and appearance is measured at EER 0.22 (CLIP) -- wrong one time in five.

A drawn footprint fixes the input rather than the inference. It is calibration,
not per-site policy: "a body here is invisible to this camera" is true at
every site and carries no semantics about doors, entrances or shift patterns,
so it does not make the solution site-specific in the way a labelled exit
zone would.

What is stored
--------------
Polygons in **floor metres**, the same frame as `GroundPlane.floor_xy` and
`reachability` -- not pixels. Storing metres means the plan survives a change
of resolution, crop or lens correction, and it is directly comparable against
the reachability grid without re-projecting anything at load time.

Two kinds:

* ``walkable`` - the outer boundary of floor a person can stand on. Anything
  outside every walkable polygon becomes BLOCKED, which is what converts the
  88.6% UNKNOWN into a real constraint.
* ``blocked``  - one obstacle footprint each: not primarily "cannot walk
  here" but "cannot be SEEN here" -- a person whose real position falls
  inside loses their feet, or all of their box, to the object in front of
  the lens. Solid furniture makes both true at once, which is why it also
  routes the geodesic path around rather than through. Wins over
  ``walkable`` wherever they overlap, so you can draw one loose boundary
  and then punch out the furniture inside it.

Drawing only ``blocked`` polygons is valid and leaves everything else as the
depth map found it -- useful when you trust the auto map and only want to
correct a few obstacles.

Reading obstacle height back from depth
----------------------------------------
The footprint you draw only says *where* something stands, not how tall it
is -- and height is what decides whether it hides a whole person or just
their feet. ``obstacle_heights`` answers that from the depth model, without
asking you to tag it by hand: `scene_depth.SceneModel` already holds a
per-pixel height-above-floor field, rescaled against real foot observations
(the same fit whose residual is reported as ``floor_resid_m`` -- 14/37 cm on
the two small rooms, 152 cm on office_cam1's larger one). Sampling that field
inside a drawn polygon turns a shape into "ankle-high, this is a rug" or
"1.4 m, this hides everyone but the head" for free. Office_cam1's 152 cm
noise floor is still fine for telling a desk from a doorway -- it is not fine
for telling a desk from a low table, so treat borderline classes there as
approximate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

WALKABLE = "walkable"
BLOCKED_KIND = "blocked"
KINDS = (WALKABLE, BLOCKED_KIND)


def points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """(N,2) points, (M,2) polygon -> (N,) bool. Vectorised ray casting.

    Implemented here rather than pulled from matplotlib or cv2 so that `reid`
    keeps needing nothing but numpy; this module is imported by the tracker on
    a path where dragging in a plotting stack would be gratuitous.

    Points exactly on an edge are not guaranteed either way. That is fine: the
    caller rasterises 10 cm cells, so a boundary cell is ambiguous regardless.
    """
    pts = np.atleast_2d(np.asarray(pts, float))
    poly = np.asarray(poly, float)
    if len(poly) < 3:
        return np.zeros(len(pts), bool)
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    for xa, ya, xb, yb in zip(x1, y1, x2, y2):
        if ya == yb:                       # horizontal edge casts no crossing
            continue
        # does the horizontal ray at y cross this edge?
        straddles = (ya > y) != (yb > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = xa + (y - ya) * (xb - xa) / (yb - ya)
        inside ^= straddles & (x < xint)
    return inside


@dataclass
class FloorPlan:
    """Hand-drawn floor geometry for one camera, in floor metres."""

    polygons: list[dict] = field(default_factory=list)
    camera: str = ""
    note: str = ""

    # ── authoring ───────────────────────────────────────────────────────────
    def add(self, kind: str, xy: np.ndarray) -> None:
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        xy = np.asarray(xy, float).reshape(-1, 2)
        if len(xy) < 3:
            raise ValueError("a polygon needs at least 3 points")
        if not np.isfinite(xy).all():
            raise ValueError(
                "polygon has non-finite floor coordinates -- a clicked point "
                "was at or above the horizon and has no floor intersection"
            )
        self.polygons.append({"kind": kind, "xy": xy.tolist()})

    def of_kind(self, kind: str) -> list[np.ndarray]:
        return [np.asarray(p["xy"], float) for p in self.polygons if p["kind"] == kind]

    # ── analysis ────────────────────────────────────────────────────────────
    def obstacle_heights(self, scene) -> list[dict]:
        """Sample `scene.obstacle` (the depth model's calibrated height-above-
        floor grid) inside each ``blocked`` polygon and classify it.

        `scene` and this plan share the same floor-metre frame but not
        necessarily the same grid, so cell centres are generated fresh from
        `scene.cell_m`/`scene.origin` rather than reusing a `Reachability`'s.
        Unseen cells (NaN in `scene.obstacle`) are dropped before the
        percentiles are taken; a polygon that is entirely unseen is reported
        as such rather than silently scored on zero samples.
        """
        from .reachability import FLOOR_TOL_M, TALL_M

        # `TALL_M` (0.80) is reachability's "solid enough to block walking"
        # line, not a "hides the whole person" line -- a desk at 0.8m still
        # leaves someone's torso and head visible. HEAD_M is a second,
        # separate line for that: the measured median stature on this
        # footage (bench/06) is 1.70m, and an obstacle at or above roughly
        # shoulder height starts hiding someone almost entirely rather than
        # just their legs.
        HEAD_M = 1.45

        ny, nx = scene.obstacle.shape
        iy, ix = np.mgrid[0:ny, 0:nx]
        centres = np.stack([
            scene.origin[0] + (ix.ravel() + 0.5) * scene.cell_m,
            scene.origin[1] + (iy.ravel() + 0.5) * scene.cell_m,
        ], axis=1)
        h = scene.obstacle.ravel()

        out = []
        for poly in self.of_kind(BLOCKED_KIND):
            inside = points_in_polygon(centres, poly)
            seen = inside & np.isfinite(h)
            n_inside = int(inside.sum())
            if not seen.any():
                out.append({"n_cells": n_inside, "n_seen": 0, "cls": "unseen"})
                continue
            hv = h[seen]
            p50, p90 = float(np.median(hv)), float(np.percentile(hv, 90))
            if p90 < FLOOR_TOL_M:
                cls = "low (walk-over, ~floor height)"
            elif p90 < TALL_M:
                cls = "knee-high (hides feet/lower legs, person still visible)"
            elif p90 < HEAD_M:
                cls = "chest-high (hides legs and torso, head may still show)"
            else:
                cls = "near/above head height (can hide a person almost entirely)"
            out.append({
                "n_cells": n_inside, "n_seen": int(seen.sum()),
                "height_p50_m": p50, "height_p90_m": p90,
                "height_max_m": float(hv.max()), "cls": cls,
            })
        return out

    # ── persistence ─────────────────────────────────────────────────────────
    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "units": "metres, GroundPlane.floor_xy frame",
            "camera": self.camera,
            "note": self.note,
            "polygons": self.polygons,
        }, indent=2))

    @staticmethod
    def load(path) -> "FloorPlan":
        d = json.loads(Path(path).read_text())
        fp = FloorPlan(camera=d.get("camera", ""), note=d.get("note", ""))
        fp.polygons = d["polygons"]
        return fp

    # ── application ─────────────────────────────────────────────────────────
    def apply_to(self, reach, verbose: bool = True):
        """Stamp this plan onto a `Reachability`, in place, and return it.

        Order matters: walkable first (it says where the room ends), then
        blocked (it punches furniture out of that room). A cell outside every
        walkable polygon is BLOCKED -- leaving it UNKNOWN would preserve the
        vagueness this module exists to remove. Inside the outline, only
        UNKNOWN is promoted to FREE. Depth-found furniture stays BLOCKED;
        a walkable polygon is not a claim that the desks vanished.

        Mutates ``reach`` in place for this process. Do not write the stamped
        map back to ``reachability.npz`` -- that cache is the auto map plus
        foot observations, and the plan is re-applied on every load.
        """
        from .reachability import FREE, UNKNOWN, BLOCKED

        ny, nx = reach.tier.shape
        iy, ix = np.mgrid[0:ny, 0:nx]
        centres = np.stack([
            reach.origin[0] + (ix.ravel() + 0.5) * reach.cell_m,
            reach.origin[1] + (iy.ravel() + 0.5) * reach.cell_m,
        ], axis=1)

        before = {t: int((reach.tier == t).sum()) for t in (FREE, UNKNOWN, BLOCKED)}
        tier = reach.tier.ravel().copy()

        walk = self.of_kind(WALKABLE)
        if walk:
            inside = np.zeros(len(centres), bool)
            for poly in walk:
                inside |= points_in_polygon(centres, poly)
            tier[~inside] = BLOCKED
            tier[inside & (tier == UNKNOWN)] = FREE

        for poly in self.of_kind(BLOCKED_KIND):
            tier[points_in_polygon(centres, poly)] = BLOCKED

        reach.tier = tier.reshape(ny, nx)
        reach._region = None            # regions are derived; force a recompute

        if verbose:
            a = reach.cell_m ** 2
            after = {t: int((reach.tier == t).sum()) for t in (FREE, UNKNOWN, BLOCKED)}
            name = {FREE: "free", UNKNOWN: "unknown", BLOCKED: "blocked"}
            parts = ", ".join(
                f"{name[t]} {before[t] * a:.1f}->{after[t] * a:.1f}m2"
                for t in (FREE, UNKNOWN, BLOCKED)
            )
            print(f"[floor-plan] {len(walk)} walkable + "
                  f"{len(self.of_kind(BLOCKED_KIND))} blocked polygons; {parts}")
        return reach
