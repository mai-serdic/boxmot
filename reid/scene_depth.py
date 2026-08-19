"""Static 3D scene structure from a metric depth model, fused with the
calibrated ground plane.

`scene_geometry` recovers *where the floor is*. This module recovers *what is
standing on it* - and that is the missing piece, because the measured failure
of stature as an identity cue traced back to bounding boxes whose bottom edge
rests on furniture rather than on feet. A person occluded by a desk is
localised as if standing further away and measures short; the artefact is
0.19 m within a single track, larger than the 11.4 cm that separates two
different people. No appearance model can fix that, and no amount of tracking
smoothness hides it: it is a geometry error and it needs a geometry fix.

The fix needs one thing the ground plane alone cannot give: the height of the
static scene surface at every pixel. A monocular metric depth model supplies
it. We deliberately do *not* trust its absolute scale - it was already caught
returning f = 790 px against a true 1354 px - so it is rescaled against the
floor we already trust, and the residual of that rescaling is reported as the
model's own quality score rather than assumed away.

What comes out
--------------
* ``height`` - metres above the floor, per pixel. This *is* the occluder map,
  derived rather than drawn, so it transfers to any site.
* ``floor_mask`` - pixels where a foot could physically rest.
* ``obstacle`` - top-down metric grid of obstacle height, plus free space.
* ``localize`` - foot point when the feet are visible, head point when they
  are not, and an honest NaN for stature in the second case instead of a
  confident wrong number.

Conventions follow `scene_geometry`: undistorted normalized image coordinates,
camera coordinates with world-up ``r3``, floor at ``X . r3 == cam_height_m``,
so height above floor is ``X . r3 - cam_height_m``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .scene_geometry import GroundPlane, RadialModel

# A foot cannot rest more than this far above the floor, but a doorsill, a
# cable tray or plain depth noise can. Tuned to be generous: the cost of
# calling a real floor pixel an obstacle is a discarded observation, while the
# cost of the reverse is a corrupted metre measurement.
FLOOR_TOL_M = 0.20
STEP_OVER_M = 0.25   # obstacle below this is walk-over-able, not an occluder


# --------------------------------------------------------------------------
# depth -> metric scene
# --------------------------------------------------------------------------


def physical_frame(gp: GroundPlane) -> tuple[np.ndarray, float]:
    """The *physically realisable* (up, camera-height) pair for a GroundPlane.

    `GroundPlane` is fitted up to a global sign: stature and distance are both
    invariant under ``X -> -X``, so the fit has no reason to prefer one, and
    `floor_xy` happily returns floor points reconstructed behind the camera.
    That is harmless for every consumer that only ever measures lengths - and
    silently wrong for anything that fuses a depth map or draws geometry back
    into the image, because those need the branch where the scene is actually
    in front of the lens.

    Returns ``(up, h)`` with ``up`` the unit world-up in camera coordinates and
    ``h > 0`` the camera height, such that a floor point along ray ``r`` is
    ``X = -h * r / (r . up)`` with ``X_z > 0``.
    """
    up = np.asarray(gp.r3, float).copy()
    # a ray aimed below the principal point must hit the floor in front of us
    probe = np.array([0.0, 0.5 / gp.f, 1.0])
    if probe @ up > 0:
        up = -up
    return up, abs(float(gp.cam_height_m))


def cam_point(gp: GroundPlane, xy: np.ndarray, h_m: float = 0.0) -> np.ndarray:
    """Floor coordinates (metres) + height -> camera coordinates, front-facing.

    The floor basis is negated relative to `GroundPlane.floor_xy` for exactly
    the reason above; keeping that negation here means `xy` values from either
    module refer to the same physical spot.
    """
    up, h = physical_frame(gp)
    xy = np.atleast_2d(np.asarray(xy, float))
    return -(xy[:, :1] * gp.e1 + xy[:, 1:2] * gp.e2) + (h_m - h) * up


def _pixel_rays(gp: GroundPlane, shape: tuple[int, int], mode: str) -> np.ndarray:
    """Unit-consistent viewing ray per pixel of an *undistorted* image.

    ``mode='z'`` treats the depth value as distance along the optical axis
    (the usual z-buffer convention); ``mode='ray'`` treats it as Euclidean
    distance from the camera centre. Which one a given checkpoint uses is
    rarely documented, so we fit both and keep the one that makes the known
    floor flat - a decision the data can make for us.
    """
    H, W = shape
    s = max(gp.width, gp.height) / 2.0
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    x = (u - gp.width / 2.0) / s * (gp.width / W)
    y = (v - gp.height / 2.0) / s * (gp.height / H)
    rays = np.stack([x / gp.f, y / gp.f, np.ones_like(x)], axis=-1)
    if mode == "ray":
        rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def _fit_depth_scale(
    depth: np.ndarray, rays: np.ndarray, gp: GroundPlane,
    floor_px: np.ndarray,
) -> tuple[float, float]:
    """Rescale the depth model against floor points we already trust.

    ``floor_px`` are pixel coordinates known to lie on the floor - in practice
    the observed foot points of tracked people, which cost nothing and are
    spread over exactly the walkable area we care about. One global scale is
    fitted (a monocular metric model's error is dominated by scale, and more
    parameters here would start absorbing real structure). Returns the scale
    and the robust residual in metres, which is the honest error bar on every
    height this module reports.
    """
    H, W = depth.shape[:2]
    u = np.clip(floor_px[:, 0].astype(int), 0, W - 1)
    v = np.clip(floor_px[:, 1].astype(int), 0, H - 1)
    up, h = physical_frame(gp)
    d = depth[v, u]
    proj = rays[v, u] @ up             # X . up == scale * d * proj, floor at -h
    good = np.isfinite(d) & (d > 0.1) & np.isfinite(proj) & (np.abs(proj) > 1e-6)
    if good.sum() < 20:
        raise RuntimeError("too few valid floor samples to rescale the depth map")
    scale = float(np.median(-h / (d[good] * proj[good])))
    if scale <= 0:
        raise RuntimeError("depth map reconstructs the floor behind the camera")
    resid = scale * d[good] * proj[good] + h
    return scale, float(np.median(np.abs(resid)) * 1.4826)


@dataclass
class SceneModel:
    """Metric static scene: per-pixel height above the floor, plus a top-down
    obstacle grid. Everything is derived from the footage, so commissioning a
    new camera means running it, not annotating it."""

    gp: GroundPlane
    height: np.ndarray          # (H,W) metres above floor
    xy: np.ndarray              # (H,W,2) floor coords of each surface point
    depth_scale: float
    depth_mode: str
    floor_resid_m: float        # 1-sigma agreement of depth with the known floor
    cell_m: float
    origin: np.ndarray          # (2,)
    obstacle: np.ndarray        # (ny,nx) max obstacle height per cell, NaN = unseen

    # -- construction ------------------------------------------------------
    @staticmethod
    def build(
        depth: np.ndarray, gp: GroundPlane, floor_px: np.ndarray,
        cell_m: float = 0.10, margin_m: float = 0.5,
    ) -> "SceneModel":
        best = None
        for mode in ("z", "ray"):
            rays = _pixel_rays(gp, depth.shape[:2], mode)
            try:
                scale, resid = _fit_depth_scale(depth, rays, gp, floor_px)
            except RuntimeError:
                continue
            if best is None or resid < best[0]:
                best = (resid, mode, scale, rays)
        if best is None:
            raise RuntimeError("depth map could not be aligned to the ground plane")
        resid, mode, scale, rays = best

        up, h = physical_frame(gp)
        X = rays * (scale * depth)[..., None]           # camera coords, metres
        height = X @ up + h                             # floor sits at -h
        # negated basis so these agree with `GroundPlane.floor_xy` (see cam_point)
        xy = -np.stack([X @ gp.e1, X @ gp.e2], axis=-1)

        # top-down obstacle grid over the region the floor actually spans
        fm = np.abs(height) < FLOOR_TOL_M
        if fm.sum() < 100:
            raise RuntimeError("no floor pixels found; depth/plane disagree badly")
        lo = np.nanpercentile(xy[fm], 1, axis=0) - margin_m
        hi = np.nanpercentile(xy[fm], 99, axis=0) + margin_m
        nx = max(int(np.ceil((hi[0] - lo[0]) / cell_m)), 1)
        ny = max(int(np.ceil((hi[1] - lo[1]) / cell_m)), 1)
        obstacle = np.full((ny, nx), np.nan)
        gx = ((xy[..., 0] - lo[0]) / cell_m).astype(int)
        gy = ((xy[..., 1] - lo[1]) / cell_m).astype(int)
        ok = (np.isfinite(height) & (gx >= 0) & (gx < nx)
              & (gy >= 0) & (gy < ny) & (height < 3.0))
        # np.maximum.at accumulates the *tallest* surface seen over each cell,
        # which is what an occluder map wants: the thing that hides feet.
        flat = np.full(ny * nx, -np.inf)
        np.maximum.at(flat, gy[ok] * nx + gx[ok], height[ok])
        obstacle = np.where(np.isfinite(flat), flat, np.nan).reshape(ny, nx)
        obstacle[~np.isfinite(flat).reshape(ny, nx)] = np.nan

        return SceneModel(gp, height, xy, scale, mode, resid,
                          cell_m, np.asarray(lo, float), obstacle)

    # -- use ---------------------------------------------------------------
    @property
    def floor_mask(self) -> np.ndarray:
        return np.abs(self.height) < FLOOR_TOL_M

    @property
    def free_space(self) -> np.ndarray:
        """Floor cells a person could stand in. NaN cells were never observed -
        not the same as blocked, and kept distinct on purpose."""
        return np.isfinite(self.obstacle) & (self.obstacle < STEP_OVER_M)

    def surface_height_at(self, px: np.ndarray) -> np.ndarray:
        """Height of the static scene at image pixels (N,2). This is the whole
        occluder query: if it is well above the floor, a box bottom landing
        here is furniture, not feet."""
        px = np.atleast_2d(np.asarray(px, float))
        H, W = self.height.shape
        sx, sy = W / self.gp.width, H / self.gp.height
        u = np.clip((px[:, 0] * sx).astype(int), 0, W - 1)
        v = np.clip((px[:, 1] * sy).astype(int), 0, H - 1)
        return self.height[v, u]

    def foot_occluded(self, foot_px: np.ndarray, tol_m: float = FLOOR_TOL_M) -> np.ndarray:
        return self.surface_height_at(foot_px) > tol_m

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            p, height=self.height.astype(np.float32),
            xy=self.xy.astype(np.float32), obstacle=self.obstacle.astype(np.float32),
            meta=np.frombuffer(json.dumps({
                "depth_scale": self.depth_scale, "depth_mode": self.depth_mode,
                "floor_resid_m": self.floor_resid_m, "cell_m": self.cell_m,
                "origin": self.origin.tolist(),
            }).encode(), dtype=np.uint8),
        )

    @staticmethod
    def load(path: str | Path, gp: GroundPlane) -> "SceneModel":
        z = np.load(str(path))
        m = json.loads(z["meta"].tobytes().decode())
        return SceneModel(gp, z["height"].astype(np.float64), z["xy"].astype(np.float64),
                          m["depth_scale"], m["depth_mode"], m["floor_resid_m"],
                          m["cell_m"], np.array(m["origin"]), z["obstacle"].astype(np.float64))


# --------------------------------------------------------------------------
# occlusion-aware localisation
# --------------------------------------------------------------------------


def floor_xy_from_head(gp: GroundPlane, head_h: np.ndarray,
                       stature_m: float = 1.70) -> np.ndarray:
    """Floor position from the *head* point, assuming a stature.

    From a ceiling-mounted camera the head is the one landmark that furniture
    almost never hides, so this is the fallback that keeps an occluded person
    in the metric world at all. It trades a measurement for an assumption -
    stature is now an input, so it can no longer be an output - which is
    exactly the right trade when the alternative is a foot point that is
    silently wrong.
    """
    head_h = np.atleast_2d(np.asarray(head_h, float))
    rays = head_h @ np.diag([1.0 / gp.f, 1.0 / gp.f, 1.0]).T
    denom = rays @ gp.r3
    # Height above the floor is X.r3 - cam_h and `up` is the direction of
    # increasing X.r3, so the head sits where lam * (ray.r3) == cam_h + stature.
    lam = np.where(np.abs(denom) > 1e-9, (gp.cam_height_m + stature_m) / denom, np.nan)
    Xh = rays * lam[:, None]
    Xb = Xh - stature_m * gp.r3
    return np.stack([Xb @ gp.e1, Xb @ gp.e2], axis=1)


@dataclass
class StatureField:
    """Per-cell multiplicative correction for measured stature.

    Occlusion gating removes the gross errors but leaves a residual of roughly
    15-20 cm that also varies with where a person stands - and a person's
    height does not depend on where they stand, so all of it is error. Fitting
    it as a two-way model ``log(stature) = person + cell`` separates the two:
    the person term is the cue we want, the cell term is a lens/plane/detector
    artefact that pretends to be one.

    It is worth being careful about *why* this is legitimate rather than a
    self-fulfilling fit. The person term is keyed on tracker output, not on
    ground truth, so no labels are consumed and the field can be estimated on
    site from unlabelled footage. Independently fitting the field on two
    disjoint halves of the tracklets and correlating the results (r = +0.66 on
    the reference clip) is what shows it is real scene structure: noise would
    not reproduce across a track split.
    """

    nb: int
    bias: np.ndarray            # (nb*nb,) log-space correction
    seen: np.ndarray            # (nb*nb,) bool, cells with enough support

    @staticmethod
    def _cells(px: np.ndarray, gp: GroundPlane, nb: int) -> np.ndarray:
        gx = np.clip((px[:, 0] / gp.width * nb).astype(int), 0, nb - 1)
        gy = np.clip((px[:, 1] / gp.height * nb).astype(int), 0, nb - 1)
        return gy * nb + gx

    @staticmethod
    def fit(px: np.ndarray, stature: np.ndarray, track_id: np.ndarray,
            gp: GroundPlane, nb: int = 16, min_per_cell: int = 4,
            iters: int = 50) -> "StatureField":
        ok = np.isfinite(stature) & (stature > 0.9) & (stature < 2.4)
        if ok.sum() < 100:
            raise RuntimeError("not enough clean statures to fit a bias field")
        y = np.log(stature[ok])
        cl = StatureField._cells(px[ok], gp, nb)
        ids = {t: i for i, t in enumerate(np.unique(track_id[ok]))}
        pi = np.array([ids[t] for t in track_id[ok]])
        bias = np.zeros(nb * nb)
        per = np.zeros(len(ids))
        uc = np.unique(cl)
        for _ in range(iters):
            r = y - bias[cl]
            for i in range(len(ids)):
                m = pi == i
                if m.sum():
                    per[i] = np.median(r[m])
            r = y - per[pi]
            for c in uc:
                m = cl == c
                if m.sum() >= min_per_cell:
                    bias[c] = np.median(r[m])
            bias -= np.median(bias[uc])       # the mean scale stays with stature
        seen = np.zeros(nb * nb, bool)
        seen[uc] = np.bincount(cl, minlength=nb * nb)[uc] >= min_per_cell
        return StatureField(nb, bias, seen)

    def apply(self, px: np.ndarray, stature: np.ndarray,
              gp: GroundPlane) -> np.ndarray:
        """Corrected stature; NaN where the field was never observed, because
        an uncorrected value here is exactly the kind of confident-but-wrong
        number that made stature useless to begin with."""
        cl = self._cells(np.atleast_2d(px), gp, self.nb)
        out = np.where(self.seen[cl], stature * np.exp(-self.bias[cl]), np.nan)
        return np.asarray(out, float)

    def to_dict(self) -> dict:
        return {"nb": self.nb, "bias": self.bias.tolist(),
                "seen": self.seen.astype(int).tolist()}

    @staticmethod
    def from_dict(d: dict) -> "StatureField":
        return StatureField(int(d["nb"]), np.array(d["bias"], float),
                            np.array(d["seen"], bool))


def localize(
    gp: GroundPlane, scene: SceneModel | None,
    foot_px: np.ndarray, foot_h: np.ndarray, head_h: np.ndarray,
    stature_prior_m: float = 1.70,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Occlusion-aware floor position + stature for a batch of observations.

    Returns ``(xy, stature_m, feet_visible)``. Where the feet are visible both
    outputs are measurements. Where they are not, the position comes from the
    head under a stature prior and the stature is returned as NaN - refusing
    to answer rather than answering wrongly, which is the entire point: the
    corrupted samples are what destroyed stature as a cue in the first place.
    """
    xy = gp.floor_xy(foot_h)
    st = gp.stature_m(foot_h, head_h)
    if scene is None:
        return xy, st, np.ones(len(xy), bool)
    vis = ~scene.foot_occluded(foot_px)
    xy_head = floor_xy_from_head(gp, head_h, stature_prior_m)
    xy = np.where(vis[:, None], xy, xy_head)
    st = np.where(vis, st, np.nan)
    return xy, st, vis


def floor_from_boxes(
    gp: GroundPlane, scene: SceneModel | None, boxes: np.ndarray,
    stature_prior_m: float = 1.70,
) -> tuple[np.ndarray, np.ndarray]:
    """Person boxes (N,4) x1y1x2y2 in *distorted* pixels -> floor xy + feet-visible.

    Wraps the undistort/normalise/localise dance that every caller needs, so
    the live tracker and the offline renderer cannot drift apart on it. Foot
    point is the bottom-centre of the box, head point the top-centre.
    """
    boxes = np.atleast_2d(np.asarray(boxes, float))
    if not len(boxes):
        return np.empty((0, 2)), np.empty(0, bool)
    model = gp.radial_model()
    cx = 0.5 * (boxes[:, 0] + boxes[:, 2])
    foot, head = np.column_stack([cx, boxes[:, 3]]), np.column_stack([cx, boxes[:, 1]])

    def _nh(px):
        u = model.undistort_norm(model.to_norm(px))
        return np.column_stack([u, np.ones(len(u))])

    foot_u = model.to_pixel(model.undistort_norm(model.to_norm(foot)))
    xy, _st, vis = localize(gp, scene, foot_u, _nh(foot), _nh(head), stature_prior_m)
    return xy, vis
