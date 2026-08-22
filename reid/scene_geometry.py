"""Automatic metric ground-plane calibration from pedestrians.

No checkerboard, no surveyor, no per-site annotation, no hand-drawn zones.
Everything here is estimated from the footage itself, which is what makes it
transferable: any fixed camera that watches people walk can be calibrated by
this module, so nothing below is specific to one site.

Pipeline
--------
1. `estimate_radial_k1` - plumb-line distortion estimate from a static
   background. Built environments are full of straight edges; the correct
   undistortion is the one under which the most total straight-line length is
   detectable.
2. `collect_observations` - turn tracklet boxes into (foot, head) pairs,
   keeping only fully-visible upright standing people.
3. `vertical_vanishing_point` - every head->foot line is the image of a world
   vertical, so they all meet at Vz.
4. `horizon_from_equal_heights` - for two people of *equal* height, the line
   through their feet and the line through their heads meet on the horizon.
   Many pairs give many horizon points; fit a line.
5. `GroundPlane` - Vz + horizon pins down the focal length (the horizon is the
   pole of Vz w.r.t. the image of the absolute conic); assuming a mean stature
   fixes the remaining scale, giving camera height and a metric floor map.

Why this matters for ReID
-------------------------
Association stops happening in pixels. Distance becomes metres, so a speed
limit is a physical constraint rather than a tuned pixel budget, and a
person's *stature* becomes observable - a clothing-independent cue that,
unlike a face, is fully visible from behind.

Conventions: all geometry is done in **normalized image coordinates**, i.e.
x = (u - W/2) / s, y = (v - H/2) / s with s = max(W, H) / 2. The principal
point is therefore the origin, which keeps the conic algebra simple and the
least-squares problems well conditioned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

MEAN_STATURE_M = 1.70  # only sets the global metre scale; see GroundPlane


# --------------------------------------------------------------------------
# 1. radial distortion
# --------------------------------------------------------------------------


@dataclass
class RadialModel:
    """Single-parameter division model about the image centre.

    Undistortion is ``x_u = x_d / (1 + k1 * r_d^2)`` in normalized coords, so
    ``k1 < 0`` corrects barrel distortion. One parameter is deliberate: with
    only static-scene edges to fit, more parameters buy noise, not accuracy.
    """

    k1: float
    width: int
    height: int

    @property
    def scale(self) -> float:
        return max(self.width, self.height) / 2.0

    def to_norm(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        c = np.array([self.width / 2.0, self.height / 2.0])
        return (pts - c) / self.scale

    def to_pixel(self, pts: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
        c = np.array([self.width / 2.0, self.height / 2.0])
        return pts * self.scale + c

    def undistort_norm(self, pts_norm: np.ndarray) -> np.ndarray:
        p = np.asarray(pts_norm, dtype=np.float64).reshape(-1, 2)
        r2 = (p ** 2).sum(axis=1, keepdims=True)
        return p / (1.0 + self.k1 * r2)

    def distort_norm(self, pts_undist: np.ndarray) -> np.ndarray:
        """Inverse of ``undistort_norm``. Fixed-point, same as ``undistort_image``."""
        xu = np.asarray(pts_undist, dtype=np.float64).reshape(-1, 2)
        xd = xu.copy()
        for _ in range(12):
            r2 = (xd ** 2).sum(axis=1, keepdims=True)
            xd = xu * (1.0 + self.k1 * r2)
        return xd

    def undistort_points(self, pts_pixel: np.ndarray) -> np.ndarray:
        """Pixel -> undistorted normalized coords (the space all geometry uses)."""
        return self.undistort_norm(self.to_norm(pts_pixel))

    def distort_points(self, pts_undist: np.ndarray) -> np.ndarray:
        """Undistorted normalized coords -> distorted pixels."""
        return self.to_pixel(self.distort_norm(pts_undist))

    def undistort_image(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        s = max(w, h) / 2.0
        cx, cy = w / 2.0, h / 2.0
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        xn, yn = (xx - cx) / s, (yy - cy) / s
        # inverse map: for each *output* (undistorted) pixel find the source.
        # x_u = x_d/(1+k1 r_d^2) is inverted numerically by fixed-point iteration.
        xd, yd = xn.copy(), yn.copy()
        for _ in range(12):
            r2 = xd * xd + yd * yd
            xd = xn * (1.0 + self.k1 * r2)
            yd = yn * (1.0 + self.k1 * r2)
        return cv2.remap(
            img, (xd * s + cx).astype(np.float32), (yd * s + cy).astype(np.float32),
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
        )


IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def background_frame(source: str | Path, n_samples: int = 60) -> np.ndarray:
    """Per-pixel median over sampled frames - people are transient, so this
    leaves the static scene, which is what the plumb-line fit needs.

    Accepts a video file or a directory of image frames, since exported CCTV
    often arrives as stills.
    """
    src = Path(source)
    frames = []
    if src.is_dir():
        files = sorted(f for f in src.iterdir() if f.suffix.lower() in IMG_EXT)
        if not files:
            raise RuntimeError(f"no images found in {src}")
        for i in np.linspace(0, len(files) - 1, min(n_samples, len(files))).astype(int):
            img = cv2.imread(str(files[i]))
            if img is not None:
                frames.append(img)
    else:
        cap = cv2.VideoCapture(str(src))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in np.linspace(0, max(total - 1, 0), n_samples).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok:
                frames.append(fr)
        cap.release()
    if not frames:
        raise RuntimeError(f"could not read frames from {source}")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def _straightness_score(img_gray: np.ndarray, min_len_frac: float = 0.06) -> float:
    """Total squared length of long line segments. Barrel distortion breaks a
    real straight edge into several short segments; correcting it merges them,
    so this score peaks at the right k1. Squaring rewards merging over count."""
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_ADV)
    lines = lsd.detect(img_gray)[0]
    if lines is None:
        return 0.0
    seg = lines.reshape(-1, 4)
    lengths = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    min_len = min_len_frac * max(img_gray.shape)
    keep = lengths[lengths >= min_len]
    return float((keep ** 2).sum())


def estimate_radial_k1(
    bg: np.ndarray, lo: float = -0.60, hi: float = 0.15, coarse: int = 26
) -> tuple[float, list[tuple[float, float]]]:
    """Scan k1 for maximum detectable straight-line length, then refine.

    Returns ``(k1, curve)`` where curve is the (k1, score) trace - worth
    inspecting, because a flat curve means the scene lacked usable straight
    edges and the estimate should not be trusted.
    """
    h, w = bg.shape[:2]
    curve: list[tuple[float, float]] = []

    def score(k1: float) -> float:
        m = RadialModel(k1, w, h)
        und = m.undistort_image(bg)
        g = cv2.cvtColor(und, cv2.COLOR_BGR2GRAY)
        # ignore the black border introduced by undistortion
        valid = (g > 0).astype(np.uint8)
        er = cv2.erode(valid, np.ones((9, 9), np.uint8))
        s = _straightness_score(cv2.bitwise_and(g, g, mask=er))
        curve.append((k1, s))
        return s

    grid = np.linspace(lo, hi, coarse)
    scores = [score(float(k)) for k in grid]
    best = float(grid[int(np.argmax(scores))])
    step = (hi - lo) / (coarse - 1)
    fine = np.linspace(best - step, best + step, 9)
    fscores = [score(float(k)) for k in fine]
    best = float(fine[int(np.argmax(fscores))])
    curve.sort()
    return best, curve


# --------------------------------------------------------------------------
# 2. pedestrian observations
# --------------------------------------------------------------------------


@dataclass
class Observations:
    """Foot and head points in undistorted normalized homogeneous coords."""

    foot: np.ndarray  # (N, 3)
    head: np.ndarray  # (N, 3)
    track_id: np.ndarray  # (N,)
    frame: np.ndarray  # (N,)

    def __len__(self) -> int:
        return len(self.track_id)


def collect_observations(
    traj: dict[int, list],
    model: RadialModel,
    min_track_len: int = 15,
    min_box_h: float = 80.0,
    min_aspect: float = 1.5,
    border_px: int = 6,
    stride: int = 5,
    dedup_cell_px: float = 0.0,
) -> Observations:
    """Tracklet boxes -> (foot, head) pairs, keeping only usable observations.

    The filters all encode "this box is a whole, upright, standing person":
    boxes clipped by the frame edge have no true head or foot point, and a
    squat aspect ratio means the person is sitting or crouching, which breaks
    the constant-stature assumption the calibration rests on.

    ``dedup_cell_px`` keeps at most one observation per (track, floor cell), so
    the fit is driven by how many *distinct* viewpoints a person was seen from
    rather than by how long they stood still. Without it a single person seated
    at a desk for twenty minutes supplies thousands of near-identical samples
    and wins the RANSAC inlier vote outright, tilting the horizon to fit one
    chair. Measured on an office camera: horizon slope +27.4 deg -> +8 deg.
    Set to 0 to disable.
    """
    W, H = model.width, model.height
    f_pts, h_pts, tids, frames = [], [], [], []
    for tid, rows in traj.items():
        if len(rows) < min_track_len:
            continue
        for r in rows[::stride]:
            fr, x1, y1, x2, y2 = r
            bw, bh = x2 - x1, y2 - y1
            if bh < min_box_h or bw <= 0 or bh / bw < min_aspect:
                continue
            if (x1 < border_px or y1 < border_px
                    or x2 > W - border_px or y2 > H - border_px):
                continue
            cx = (x1 + x2) / 2.0
            f_pts.append([cx, y2])
            h_pts.append([cx, y1])
            tids.append(tid)
            frames.append(fr)
    if not f_pts:
        raise RuntimeError("no usable pedestrian observations")

    if dedup_cell_px > 0:
        seen: set[tuple[int, int, int]] = set()
        keep = []
        for i, (fp, tid) in enumerate(zip(f_pts, tids)):
            cell = (int(tid), int(fp[0] // dedup_cell_px), int(fp[1] // dedup_cell_px))
            if cell in seen:
                continue
            seen.add(cell)
            keep.append(i)
        f_pts = [f_pts[i] for i in keep]
        h_pts = [h_pts[i] for i in keep]
        tids = [tids[i] for i in keep]
        frames = [frames[i] for i in keep]

    def hom(pts):
        u = model.undistort_points(np.array(pts))
        return np.hstack([u, np.ones((len(u), 1))])

    return Observations(hom(f_pts), hom(h_pts), np.array(tids), np.array(frames))


# --------------------------------------------------------------------------
# 3-4. vanishing point and horizon
# --------------------------------------------------------------------------


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def vertical_vanishing_point(
    obs: Observations, iters: int = 800, thresh: float = 3e-3, rng_seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """RANSAC the common intersection of all head->foot lines.

    Returns ``(vz, inlier_mask)``. Residual is the algebraic distance
    ``|l . v|`` with both normalized, which for near-image points is
    proportional to the point-line distance in normalized units.
    """
    lines = np.cross(obs.foot, obs.head)
    lines = lines / (np.linalg.norm(lines[:, :2], axis=1, keepdims=True) + 1e-12)
    n = len(lines)
    rng = np.random.default_rng(rng_seed)

    def fit(sel: np.ndarray) -> np.ndarray:
        _, _, vt = np.linalg.svd(lines[sel])
        return _normalize(vt[-1])

    best_v, best_in = None, np.zeros(n, dtype=bool)
    for _ in range(iters):
        sel = rng.choice(n, size=2, replace=False)
        v = fit(sel)
        res = np.abs(lines @ v)
        inl = res < thresh
        if inl.sum() > best_in.sum():
            best_v, best_in = v, inl
    if best_in.sum() >= 2:  # refit on all inliers
        best_v = fit(np.where(best_in)[0])
        best_in = np.abs(lines @ best_v) < thresh
    return best_v, best_in


def dwell_weights(obs: Observations, cell_norm: float = 0.04) -> np.ndarray:
    """Per-observation weight that discounts standing still.

    Observations are binned by (track, floor cell) and each is weighted by the
    reciprocal of its bin's population, so a track contributes about the same
    total weight per *place it was seen* regardless of how long it lingered
    there. This is what keeps a seated worker from deciding the calibration of
    a room; see ``fit_height_field``.

    ``cell_norm`` is the bin size in normalized image units (0.04 is ~25 px on
    a 1280-wide frame).
    """
    f = obs.foot[:, :2] / obs.foot[:, 2:]
    keys = np.stack([obs.track_id,
                     np.floor(f[:, 0] / cell_norm),
                     np.floor(f[:, 1] / cell_norm)], axis=1)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return 1.0 / counts[inv]


def fit_height_field(
    obs: Observations,
    iters: int = 3000,
    rel_thresh: float = 0.10,
    rng_seed: int = 0,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Horizon from the *apparent height field*, which is linear in the image.

    A person of fixed stature standing on a plane projects to an apparent
    pixel height ``h(u, v) = a*u + b*v + c`` that is exactly linear in the
    image coordinates of their feet (in the limit of a distant vertical
    vanishing point, which is precisely the regime where the head->foot line
    intersection used by classical pedestrian calibration breaks down). The
    horizon is where that height vanishes, so ``l = (a, b, c)``.

    This is the well-conditioned way to get the horizon in a close-range
    indoor view: three parameters fitted over hundreds of observations,
    instead of intersecting near-parallel lines. RANSAC handles the residual
    contamination from crouching, occluded and mis-detected boxes; the fit is
    then refined by iteratively reweighted least squares on the inliers.

    ``weights`` scales each observation's contribution to the RANSAC vote and
    to the refinement. It exists because the vote is otherwise decided by how
    *long* people stood somewhere rather than by how many distinct places they
    were seen in: one person seated at a desk for twenty minutes can outvote
    everybody who actually walked, and the horizon tilts to fit their chair.
    Passing per-observation weights of 1/(samples in that track's floor cell)
    removes the dwell-time bias without throwing away data, which matters on
    cameras that are sample-starved to begin with.

    Returns ``(horizon, inlier_mask, rel_rms)``. ``rel_rms`` is the relative
    residual - if it is not small, the planar/fixed-stature assumption did not
    hold and the calibration downstream should not be trusted.
    """
    foot = obs.foot[:, :2] / obs.foot[:, 2:]
    head = obs.head[:, :2] / obs.head[:, 2:]
    h = foot[:, 1] - head[:, 1]          # apparent height, normalized units
    A = np.column_stack([foot[:, 0], foot[:, 1], np.ones(len(h))])
    n = len(h)
    rng = np.random.default_rng(rng_seed)
    w_obs = (np.ones(n) if weights is None
             else np.asarray(weights, dtype=np.float64).reshape(n))

    best_l, best_in, best_score = None, np.zeros(n, dtype=bool), -1.0
    for _ in range(iters):
        sel = rng.choice(n, size=3, replace=False)
        try:
            l = np.linalg.solve(A[sel], h[sel])
        except np.linalg.LinAlgError:
            continue
        pred = A @ l
        inl = np.abs(pred - h) < rel_thresh * h
        score = float(w_obs[inl].sum())
        if score > best_score:
            best_l, best_in, best_score = l, inl, score
    if best_l is None or best_in.sum() < 10:
        raise RuntimeError("height-field RANSAC failed")

    # IRLS refinement (Huber) on the inlier set
    l = best_l
    idx = np.where(best_in)[0]
    for _ in range(20):
        r = A[idx] @ l - h[idx]
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        w = np.clip(1.345 * s / (np.abs(r) + 1e-12), None, 1.0) * w_obs[idx]
        Aw = A[idx] * w[:, None]
        l = np.linalg.lstsq(Aw, h[idx] * w, rcond=None)[0]
    best_in = np.abs(A @ l - h) < rel_thresh * h
    rel_rms = float(np.sqrt(np.mean(((A[best_in] @ l - h[best_in]) / h[best_in]) ** 2)))

    # Return as a proper line: h(u,v)=0 is the horizon. Normalize so that the
    # sign convention matches "height grows as you move away from the horizon".
    horizon = np.array([l[0], l[1], l[2]], dtype=np.float64)
    horizon = horizon / np.linalg.norm(horizon[:2])
    return horizon, best_in, rel_rms


def focal_from_stature_consistency(
    obs: Observations,
    horizon: np.ndarray,
    mask: np.ndarray | None = None,
    f_lo: float = 0.25,
    f_hi: float = 3.0,
    n_scan: int = 200,
) -> tuple[float, float, tuple[float, float], float]:
    """Recover focal length by demanding that stature not depend on position.

    The horizon fixes the plane's orientation but not the focal length. The
    missing constraint is available from the people themselves: whatever their
    individual heights, a person must not appear *taller because they stood
    closer to the camera*. So we scan f and keep the value that minimizes the
    spread of recovered statures across the whole floor. At the wrong f the
    error is systematic - stature drifts with range - which is both why it is
    detectable and why leaving it uncorrected quietly destroys stature as an
    identity cue, since it then encodes position rather than person.

    Self-contained: no depth model, no scene assumptions beyond a flat floor
    and people who stand upright, so it transfers to any camera.

    Returns ``(f, rel_spread, (f_lo95, f_hi95), near_far_ratio)`` where the
    interval spans the focal lengths within 5% of the best spread - wide means
    weakly determined, which is worth knowing before trusting metric output.
    """
    idx = np.arange(len(obs)) if mask is None else np.where(mask)[0]
    foot, head = obs.foot[idx], obs.head[idx]
    l = horizon / np.linalg.norm(horizon[:2])

    def gains(f: float) -> np.ndarray:
        K = np.diag([f, f, 1.0])
        Kinv = np.diag([1.0 / f, 1.0 / f, 1.0])
        r3 = _normalize(np.array([f * l[0], f * l[1], l[2]]))
        B = K @ r3
        out = np.full(len(foot), np.nan)
        for i in range(len(foot)):
            ray = Kinv @ foot[i]
            den = r3 @ ray
            if abs(den) < 1e-9:
                continue
            A = K @ (ray / den)
            cbt = np.cross(B, head[i])
            d2 = cbt @ cbt
            if d2 >= 1e-12:
                out[i] = -(np.cross(A, head[i]) @ cbt) / d2
        return out

    fs = np.linspace(f_lo, f_hi, n_scan)
    spreads = np.full(n_scan, np.inf)
    for i, f in enumerate(fs):
        g = gains(float(f))
        ok = np.isfinite(g) & (g != 0)
        if ok.sum() < 50:
            continue
        gn = g[ok] / np.median(g[ok])
        spreads[i] = float(np.subtract(*np.percentile(gn, [75, 25])))
    if not np.isfinite(spreads).any():
        raise RuntimeError("focal scan failed: no usable observations")
    best_i = int(np.argmin(spreads))
    f = float(fs[best_i])
    within = fs[spreads <= spreads[best_i] * 1.05]

    # report the residual range bias at the chosen f, the thing we minimized
    g = gains(f)
    ok = np.isfinite(g) & (g != 0)
    K = np.diag([f, f, 1.0])
    Kinv = np.diag([1.0 / f, 1.0 / f, 1.0])
    r3 = _normalize(np.array([f * l[0], f * l[1], l[2]]))
    rays = foot[ok] @ Kinv.T
    den = rays @ r3
    X = rays / np.where(np.abs(den) > 1e-9, den, np.nan)[:, None]
    tmp = np.array([1.0, 0.0, 0.0]) if abs(r3[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = _normalize(np.cross(r3, tmp))
    e2 = _normalize(np.cross(r3, e1))
    xy = np.stack([X @ e1, X @ e2], axis=1)
    rng = np.linalg.norm(xy - np.nanmedian(xy, axis=0), axis=1)
    gg = g[ok]
    near = np.nanmedian(gg[rng < np.nanpercentile(rng, 33)])
    far = np.nanmedian(gg[rng > np.nanpercentile(rng, 67)])
    return f, float(spreads[best_i]), (float(within.min()), float(within.max())), \
        float(near / far) if far else float("nan")


def focal_from_depth(
    depth: np.ndarray,
    foot_norm: np.ndarray,
    horizon: np.ndarray,
    model: RadialModel,
    f_lo: float = 0.35,
    f_hi: float = 4.0,
    n_scan: int = 240,
) -> tuple[float, np.ndarray, float, float]:
    """Recover focal length by making a metric depth map agree with the horizon.

    The height field pins down the ground plane's *vanishing line* robustly but
    not the focal length (with the vertical vanishing point at infinity that
    equation degenerates). A metric depth map supplies exactly the missing
    degree of freedom: back-project the observed foot points - which are known
    floor samples, so no floor segmentation is needed - and pick the focal
    length whose fitted 3D plane reproduces the horizon we already trust.

    Returns ``(f, normal, cam_height_m, angle_err_deg)`` with the plane in
    camera coordinates and the camera height in metres straight from the depth
    model's scale.
    """
    H, W = depth.shape[:2]
    s = max(model.width, model.height) / 2.0
    px = foot_norm * s + np.array([model.width / 2.0, model.height / 2.0])
    u = np.clip(px[:, 0].astype(int), 0, W - 1)
    v = np.clip(px[:, 1].astype(int), 0, H - 1)
    Z = depth[v, u].astype(np.float64)
    good = np.isfinite(Z) & (Z > 0.2) & (Z < 50.0)
    if good.sum() < 20:
        raise RuntimeError("depth map gave too few valid floor samples")
    pn, Z = foot_norm[good], Z[good]

    best = None
    for f in np.linspace(f_lo, f_hi, n_scan):
        rays = np.column_stack([pn[:, 0] / f, pn[:, 1] / f, np.ones(len(pn))])
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        P = rays * Z[:, None]
        c = P.mean(axis=0)
        _, _, vt = np.linalg.svd(P - c)
        nrm = vt[-1]
        d = float(nrm @ c)
        if d < 0:
            nrm, d = -nrm, -d
        # vanishing line of this plane, compared with the trusted horizon
        l_pred = np.array([nrm[0] / f, nrm[1] / f, nrm[2]])
        l_pred /= np.linalg.norm(l_pred[:2]) + 1e-12
        ang = np.degrees(np.arccos(np.clip(
            abs(l_pred @ horizon) / (np.linalg.norm(l_pred) * np.linalg.norm(horizon)),
            0, 1)))
        if best is None or ang < best[3]:
            best = (float(f), nrm, d, float(ang))
    return best


# --------------------------------------------------------------------------
# 5. the ground plane
# --------------------------------------------------------------------------


@dataclass
class GroundPlane:
    """Metric ground plane recovered from Vz + horizon.

    All inputs/outputs in normalized image coords (see module docstring).
    ``floor_xy`` returns metres on the floor in an arbitrary but *fixed*
    orthonormal basis - orientation and origin are unrecoverable from a single
    view and irrelevant, since only distances and speeds are ever used.
    """

    f: float                  # focal length, normalized units
    vz: np.ndarray            # vertical vanishing point (3,)
    horizon: np.ndarray       # (3,)
    cam_height_m: float       # camera height above the floor
    r3: np.ndarray            # world "up" in camera coords, unit
    e1: np.ndarray = field(default_factory=lambda: np.zeros(3))
    e2: np.ndarray = field(default_factory=lambda: np.zeros(3))
    width: int = 0
    height: int = 0
    k1: float = 0.0

    # -- construction ------------------------------------------------------
    @staticmethod
    def from_horizon(
        horizon: np.ndarray, f: float, obs: Observations,
        model: RadialModel, mean_stature_m: float = MEAN_STATURE_M,
        mask: np.ndarray | None = None,
    ) -> "GroundPlane":
        """Build the plane from its vanishing line plus a focal length.

        The horizon is the ground plane's vanishing line, so the plane normal
        in camera coordinates is ``K^T l`` - that alone fixes the camera's
        orientation relative to the floor. The remaining unknown is the metre
        scale, which is set by requiring the median observed person to be
        ``mean_stature_m`` tall.
        """
        l = horizon / np.linalg.norm(horizon[:2])
        f = float(f)
        K = np.diag([f, f, 1.0])
        Kinv = np.diag([1.0 / f, 1.0 / f, 1.0])
        r3 = _normalize(np.array([f * l[0], f * l[1], l[2]]))
        vz = _normalize(K @ r3)

        # Scale: head Z is linear in camera height, so one global stature
        # assumption fixes the metre scale for the whole scene.
        idx = np.arange(len(obs)) if mask is None else np.where(mask)[0]

        def stature_gains(up: np.ndarray) -> np.ndarray:
            """Per-observation stature at unit camera height, for a given 'up'."""
            out = []
            B = K @ up
            for i in idx:
                b, t = obs.foot[i], obs.head[i]
                ray = Kinv @ b
                denom = up @ ray
                if abs(denom) < 1e-9:
                    continue
                A = K @ (ray / denom)     # foot 3D at unit camera height
                cbt = np.cross(B, t)
                d2 = cbt @ cbt
                if d2 < 1e-12:
                    continue
                out.append(-(np.cross(A, t) @ cbt) / d2)
            arr = np.array(out)
            return arr[np.isfinite(arr)]

        # Note the sign convention: `g` is invariant under r3 -> -r3 (both the
        # foot ray and the vertical flip together), so the free sign lives in
        # the camera height, not the normal. We therefore keep cam_height_m
        # signed and let it come out negative when the plane normal points away
        # from the floor - every consumer multiplies by it, so the result stays
        # consistent and statures come out positive either way.
        g = stature_gains(r3)
        if len(g) < 5 or abs(np.median(g)) < 1e-9:
            raise RuntimeError("could not establish metric scale from observations")
        cam_h = float(mean_stature_m / np.median(g))

        # orthonormal basis of the floor plane
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(r3 @ tmp) > 0.9:
            tmp = np.array([0.0, 1.0, 0.0])
        e1 = _normalize(np.cross(r3, tmp))
        e2 = _normalize(np.cross(r3, e1))
        return GroundPlane(f, vz, l, cam_h, r3, e1, e2,
                           model.width, model.height, model.k1)

    # -- use ---------------------------------------------------------------
    @property
    def K(self) -> np.ndarray:
        return np.diag([self.f, self.f, 1.0])

    def _ray(self, pts_norm_h: np.ndarray) -> np.ndarray:
        Kinv = np.diag([1.0 / self.f, 1.0 / self.f, 1.0])
        return pts_norm_h @ Kinv.T

    def floor_xy(self, foot_norm_h: np.ndarray) -> np.ndarray:
        """Foot points (N,3 homogeneous, undistorted normalized) -> metres (N,2).

        Points at or above the horizon have no floor intersection and come back
        as NaN rather than a plausible-looking wrong number.
        """
        foot_norm_h = np.atleast_2d(foot_norm_h)
        rays = self._ray(foot_norm_h)
        denom = rays @ self.r3
        X = np.where(
            np.abs(denom)[:, None] > 1e-9,
            self.cam_height_m * rays / np.where(np.abs(denom) > 1e-9, denom, np.nan)[:, None],
            np.nan,
        )
        return np.stack([X @ self.e1, X @ self.e2], axis=1)

    def pixels_from_floor(self, xy: np.ndarray) -> np.ndarray:
        """Inverse of ``floor_xy``: floor metres (N,2) -> distorted pixels (N,2).

        Used to draw a saved plan back onto the image. Points that project
        behind the camera come back as NaN rather than wrapping around.
        """
        xy = np.atleast_2d(np.asarray(xy, float))
        # Camera at the origin; the floor is X · r3 = cam_height_m.
        X = (self.cam_height_m * self.r3
             + xy[:, 0:1] * self.e1 + xy[:, 1:2] * self.e2)
        z = X[:, 2]
        und = np.full((len(xy), 2), np.nan)
        ok = np.abs(z) > 1e-9
        und[ok, 0] = self.f * X[ok, 0] / z[ok]
        und[ok, 1] = self.f * X[ok, 1] / z[ok]
        return self.radial_model().distort_points(und)

    def stature_m(self, foot_h: np.ndarray, head_h: np.ndarray) -> np.ndarray:
        """Metric height of the upright whose base is `foot_h` and top `head_h`.

        This is the clothing-independent, back-facing-compatible identity cue:
        it needs only that the person is standing, not that their face or their
        clothes are visible.
        """
        foot_h, head_h = np.atleast_2d(foot_h), np.atleast_2d(head_h)
        K, Kinv = self.K, np.diag([1.0 / self.f, 1.0 / self.f, 1.0])
        rays = foot_h @ Kinv.T
        denom = rays @ self.r3
        out = np.full(len(foot_h), np.nan)
        B = K @ self.r3
        for i in range(len(foot_h)):
            if abs(denom[i]) < 1e-9:
                continue
            Xb = self.cam_height_m * rays[i] / denom[i]
            A = K @ Xb
            cbt = np.cross(B, head_h[i])
            d2 = cbt @ cbt
            if d2 < 1e-12:
                continue
            out[i] = -(np.cross(A, head_h[i]) @ cbt) / d2
        return out

    def px_per_m(self, foot_h: np.ndarray) -> np.ndarray:
        """Local image scale, for sanity checks and for converting legacy
        pixel-space thresholds into metric ones."""
        foot_h = np.atleast_2d(foot_h)
        s = max(self.width, self.height) / 2.0
        # apparent pixel height of a 1 m upright standing at this foot point
        K = self.K
        Kinv = np.diag([1.0 / self.f, 1.0 / self.f, 1.0])
        rays = foot_h @ Kinv.T
        denom = rays @ self.r3
        out = np.full(len(foot_h), np.nan)
        for i in range(len(foot_h)):
            if abs(denom[i]) < 1e-9:
                continue
            Xb = self.cam_height_m * rays[i] / denom[i]
            top = K @ (Xb + 1.0 * self.r3)
            base = K @ Xb
            if abs(top[2]) < 1e-9 or abs(base[2]) < 1e-9:
                continue
            out[i] = np.linalg.norm(top[:2] / top[2] - base[:2] / base[2]) * s
        return out

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "f_norm": self.f,
            "f_px": self.f * max(self.width, self.height) / 2.0,
            "vz": self.vz.tolist(),
            "horizon": self.horizon.tolist(),
            "cam_height_m": self.cam_height_m,
            "r3": self.r3.tolist(),
            "e1": self.e1.tolist(),
            "e2": self.e2.tolist(),
            "width": self.width,
            "height": self.height,
            "k1": self.k1,
        }

    @staticmethod
    def from_dict(d: dict) -> "GroundPlane":
        return GroundPlane(
            f=d["f_norm"], vz=np.array(d["vz"]), horizon=np.array(d["horizon"]),
            cam_height_m=d["cam_height_m"], r3=np.array(d["r3"]),
            e1=np.array(d["e1"]), e2=np.array(d["e2"]),
            width=d["width"], height=d["height"], k1=d["k1"],
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load(path: str | Path) -> "GroundPlane":
        return GroundPlane.from_dict(json.loads(Path(path).read_text()))

    def radial_model(self) -> RadialModel:
        return RadialModel(self.k1, self.width, self.height)
