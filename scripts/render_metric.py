"""Render the metric scene over footage - the calibration, shown working.

    python scripts/render_metric.py --name gunsan_test --video videos/gunsan_test.mp4

Draws, on every frame: the 1 m floor grid, the derived occluder map, and each
tracked person annotated with their measured height in metres and their
position on the floor - beside a live top-down plan of the room.

It consumes tracklets rather than re-detecting, so what you watch is exactly
the data the benchmark scored: no second tracker, no quiet re-tuning between
the numbers and the picture.

Height is reported bias-corrected, and refused outright when the feet are
occluded. A blank is the honest output there; a confident wrong number is what
made stature useless as a cue in the first place.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from reid.scene_depth import SceneModel, StatureField, cam_point, localize  # noqa: E402
from reid.scene_geometry import GroundPlane, RadialModel  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
TRAIL = 45                      # frames of path history to draw
PANEL = 720                     # top-down panel, px


def project(gp: GroundPlane, X: np.ndarray) -> np.ndarray:
    """Camera-coords metres -> undistorted pixel coords."""
    X = np.atleast_2d(X)
    s = max(gp.width, gp.height) / 2.0
    z = np.where(np.abs(X[:, 2]) > 1e-9, X[:, 2], np.nan)
    return np.stack([gp.f * X[:, 0] / z * s + gp.width / 2.0,
                     gp.f * X[:, 1] / z * s + gp.height / 2.0], axis=1)


def undistort_maps(model: RadialModel) -> tuple[np.ndarray, np.ndarray]:
    """Same inverse map as RadialModel.undistort_image, computed once."""
    h, w = model.height, model.width
    s, cx, cy = max(w, h) / 2.0, w / 2.0, h / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn, yn = (xx - cx) / s, (yy - cy) / s
    xd, yd = xn.copy(), yn.copy()
    for _ in range(12):
        r2 = xd * xd + yd * yd
        xd, yd = xn * (1.0 + model.k1 * r2), yn * (1.0 + model.k1 * r2)
    return (xd * s + cx).astype(np.float32), (yd * s + cy).astype(np.float32)


def norm_h(model: RadialModel, px: np.ndarray) -> np.ndarray:
    """Distorted pixels -> undistorted normalized homogeneous."""
    u = model.undistort_norm(model.to_norm(np.atleast_2d(px)))
    return np.column_stack([u, np.ones(len(u))])


def undist_px(model: RadialModel, px: np.ndarray) -> np.ndarray:
    """Distorted pixels -> undistorted pixels."""
    return model.to_pixel(model.undistort_norm(model.to_norm(np.atleast_2d(px))))


def static_overlay(gp: GroundPlane, scene: SceneModel,
                   step_over_m: float = 0.25) -> tuple[np.ndarray, np.ndarray, list]:
    """Floor/obstacle tint and the 1 m grid, in undistorted image space.

    The grid comes back as polylines rather than baked into the tint, so it can
    be drawn at full opacity afterwards - blending it away is what made the
    metre scale hard to see, which is the one thing this render exists to show.
    """
    H, W = gp.height, gp.width
    tint = np.zeros((H, W, 3), np.uint8)
    floor = scene.floor_mask
    obst = np.isfinite(scene.height) & (scene.height > step_over_m) & (scene.height < 3.0)
    tint[floor] = (70, 165, 70)
    tint[obst] = (55, 55, 175)
    mask = (floor | obst)[..., None]

    grid = []
    xy = scene.xy[floor]
    if len(xy) > 100:
        lo = np.nanpercentile(xy, 2, axis=0)
        hi = np.nanpercentile(xy, 98, axis=0)
        a0, a1 = int(np.floor(lo[0])), int(np.ceil(hi[0]))
        b0, b1 = int(np.floor(lo[1])), int(np.ceil(hi[1]))
        for a in range(a0, a1 + 1):
            grid.append(project(gp, cam_point(gp, np.column_stack(
                [np.full(60, a), np.linspace(b0, b1, 60)]))))
        for b in range(b0, b1 + 1):
            grid.append(project(gp, cam_point(gp, np.column_stack(
                [np.linspace(a0, a1, 60), np.full(60, b)]))))
    return tint, mask, grid


def draw_poly(img, pts, color, w) -> None:
    ok = np.isfinite(pts).all(axis=1)
    seg = []
    for p, good in zip(pts, ok):
        if not good:
            seg = []
            continue
        seg.append(p)
        if len(seg) == 2:
            cv2.line(img, tuple(np.int32(seg[0])), tuple(np.int32(seg[1])),
                     color, w, cv2.LINE_AA)
            seg = [seg[1]]


def plan_view(scene: SceneModel, people: dict, trails: dict) -> np.ndarray:
    """Top-down: obstacles, plus where everybody is standing, in metres."""
    obs = scene.obstacle
    ny, nx = obs.shape
    img = np.full((PANEL, PANEL, 3), 22, np.uint8)
    sc = min(PANEL * 0.86 / nx, PANEL * 0.86 / ny)
    ox, oy = (PANEL - nx * sc) / 2, (PANEL - ny * sc) / 2

    # obstacle HEIGHT as a colour ramp, not a blocked/free binary: how tall the
    # thing is decides whether it hides feet, and that is the whole use of it
    seen = np.isfinite(obs)
    h = np.clip(np.nan_to_num(obs, nan=0.0) / 2.0, 0, 1)
    grid = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    grid[~seen] = (26, 26, 26)
    free = seen & (obs <= 0.25)
    grid[free] = (90, 165, 90)
    big = cv2.resize(grid, (int(nx * sc), int(ny * sc)), interpolation=cv2.INTER_NEAREST)
    img[int(oy):int(oy) + big.shape[0], int(ox):int(ox) + big.shape[1]] = big

    def to_panel(xy):
        gx = (xy[0] - scene.origin[0]) / scene.cell_m
        gy = (xy[1] - scene.origin[1]) / scene.cell_m
        return int(ox + gx * sc), int(oy + gy * sc)

    # 1 m ticks, so the panel is readable as metres and not as pixels
    for m in range(0, int(nx * scene.cell_m) + 1):
        x = int(ox + (m / scene.cell_m) * sc)
        cv2.line(img, (x, int(oy)), (x, int(oy + ny * sc)), (44, 44, 44), 1)
    for m in range(0, int(ny * scene.cell_m) + 1):
        y = int(oy + (m / scene.cell_m) * sc)
        cv2.line(img, (int(ox), y), (int(ox + nx * sc), y), (44, 44, 44), 1)

    for tid, tr in trails.items():
        col = colour(tid)
        for i in range(1, len(tr)):
            cv2.line(img, to_panel(tr[i - 1]), to_panel(tr[i]),
                     tuple(int(c * 0.55) for c in col), 1, cv2.LINE_AA)
    for tid, (xy, st, vis) in people.items():
        p = to_panel(xy)
        col = colour(tid)
        cv2.circle(img, p, 9, col, -1 if vis else 2, cv2.LINE_AA)
        cv2.putText(img, str(tid), (p[0] + 12, p[1] + 5), FONT, 0.5, col, 1, cv2.LINE_AA)

    cv2.putText(img, "TOP-DOWN PLAN  (metres)", (16, 30), FONT, 0.6,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "green = standable floor   bright = tall obstacle   1 m squares",
                (16, PANEL - 40), FONT, 0.45, (140, 140, 140), 1, cv2.LINE_AA)
    cv2.putText(img, "hollow dot = feet occluded, position inferred from head",
                (16, PANEL - 18), FONT, 0.45, (140, 140, 140), 1, cv2.LINE_AA)
    return img


def colour(tid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(int(tid) * 9781 + 5)
    c = rng.integers(70, 255, 3)
    return int(c[0]), int(c[1]), int(c[2])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--calib-dir", default=None)
    ap.add_argument("--traj", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--no-plan", action="store_true",
                    help="drop the top-down panel; camera view only")
    ap.add_argument("--big-ids", action="store_true",
                    help="ID-check mode: boxes + large bold track IDs")
    ap.add_argument("--keep-scene", action="store_true",
                    help="keep the grid + occluder tint in --big-ids mode")
    ap.add_argument("--id-scale", type=float, default=1.0,
                    help="multiplier on the big-ID text size")
    args = ap.parse_args()

    cal = Path(args.calib_dir or f"calib/{args.name}")
    traj_p = Path(args.traj or f"runs/traj/{args.name}.json")
    for p in (cal / "scene.json", cal / "scene_depth.npz", traj_p, Path(args.video)):
        if not p.exists():
            raise SystemExit(f"missing {p}")

    gp = GroundPlane.load(cal / "scene.json")
    scene = SceneModel.load(cal / "scene_depth.npz", gp)
    fld_p = cal / "stature_field.json"
    field = (StatureField.from_dict(json.loads(fld_p.read_text()))
             if fld_p.exists() else None)
    model = gp.radial_model()
    d = json.loads(traj_p.read_text())

    # frame -> [(tid, box), ...]
    per_frame: dict[int, list] = defaultdict(list)
    for tid, rows in d["traj"].items():
        for fr, x1, y1, x2, y2 in rows:
            per_frame[int(fr)].append((int(tid), (x1, y1, x2, y2)))

    print(f"[in  ] {args.video}")
    print(f"[cal ] f={gp.f*max(gp.width,gp.height)/2:.0f}px  "
          f"cam={abs(gp.cam_height_m):.2f}m  k1={gp.k1:+.3f}  "
          f"depth resid={scene.floor_resid_m*100:.0f}cm")

    mapx, mapy = undistort_maps(model)
    tint, tint_m, grid = static_overlay(gp, scene)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    # Without the plan panel, keep native resolution -- the IDs stay crisp.
    fw, fh = (int(gp.width), int(gp.height)) if args.no_plan else (1280, 720)
    suffix = "ids" if args.big_ids else "metric"
    out_p = Path(args.out or f"runs/render/{args.name}_{suffix}.mp4")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    size = (fw, fh) if args.no_plan else (fw + PANEL, max(fh, PANEL))
    vw = cv2.VideoWriter(str(out_p), cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps, size)

    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=TRAIL))
    hist: dict[int, list] = defaultdict(list)
    i = n_meas = 0
    while True:
        ok, fr = cap.read()
        if not ok or (args.max_frames and i >= args.max_frames):
            break
        u = cv2.remap(fr, mapx, mapy, cv2.INTER_LINEAR)
        # In ID-check mode the scene overlays are noise -- the point is the IDs.
        if not args.big_ids or args.keep_scene:
            u = np.where(tint_m, cv2.addWeighted(u, 0.84, tint, 0.16, 0),
                         u).astype(np.uint8)
            for pts in grid:
                draw_poly(u, pts, (0, 200, 255), 1)

        people = {}
        rows = per_frame.get(i, [])
        if rows:
            boxes = np.array([r[1] for r in rows], float)
            tids = [r[0] for r in rows]
            foot = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 3]])
            head = np.column_stack([(boxes[:, 0] + boxes[:, 2]) / 2, boxes[:, 1]])
            fh_n, hh_n = norm_h(model, foot), norm_h(model, head)
            foot_u, head_u = undist_px(model, foot), undist_px(model, head)
            xy, st, vis = localize(gp, scene, foot_u, fh_n, hh_n)
            if field is not None:
                st = field.apply(foot_u, st, gp)

            for k, tid in enumerate(tids):
                col = colour(tid)
                fp, hp = foot_u[k], head_u[k]
                if np.isfinite(st[k]):
                    hist[tid].append(float(st[k]))
                    n_meas += 1
                med = float(np.median(hist[tid])) if hist[tid] else np.nan

                if args.big_ids:
                    b = undist_px(model, boxes[k].reshape(2, 2)).reshape(-1)
                    x1, y1, x2, y2 = [int(v) for v in b]
                    cv2.rectangle(u, (x1, y1), (x2, y2), col, 3)
                    # Size the ID with the box so distant people don't get
                    # labels that cover their neighbours.
                    fs = float(np.clip((y2 - y1) / 190.0, 1.0, 3.2)) * args.id_scale
                    th_ = max(2, int(round(fs * 2.6)))
                    txt = str(tid)
                    (tw, tht), _ = cv2.getTextSize(txt, FONT, fs, th_)
                    tx = int(np.clip((x1 + x2) // 2 - tw // 2, 2, u.shape[1] - tw - 2))
                    ty = y1 - 10 if y1 - 10 - tht > 0 else min(y2 + tht + 8,
                                                               u.shape[0] - 4)
                    cv2.rectangle(u, (tx - 6, ty - tht - 8), (tx + tw + 6, ty + 8),
                                  (15, 15, 15), -1)
                    cv2.putText(u, txt, (tx, ty), FONT, fs, (0, 0, 0),
                                th_ + 4, cv2.LINE_AA)
                    cv2.putText(u, txt, (tx, ty), FONT, fs, col, th_, cv2.LINE_AA)
                    if np.isfinite(med):
                        cv2.putText(u, f"{med:.2f}m", (x1, min(y2 + 22, u.shape[0] - 4)),
                                    FONT, 0.6, col, 2, cv2.LINE_AA)
                else:
                    cv2.line(u, (int(fp[0]), int(fp[1])), (int(hp[0]), int(hp[1])),
                             col, 2, cv2.LINE_AA)
                    cv2.ellipse(u, (int(fp[0]), int(fp[1])), (26, 9), 0, 0, 360,
                                col if vis[k] else (0, 165, 255), 2, cv2.LINE_AA)
                    lab = f"#{tid}"
                    lab += f"  {med:.2f}m" if np.isfinite(med) else "  ---"
                    if not vis[k]:
                        lab += "  feet hidden"
                    (tw, th), _ = cv2.getTextSize(lab, FONT, 0.55, 1)
                    y0 = max(int(hp[1]) - 12, th + 6)
                    cv2.rectangle(u, (int(hp[0]) - 4, y0 - th - 6),
                                  (int(hp[0]) + tw + 6, y0 + 4), (20, 20, 20), -1)
                    cv2.putText(u, lab, (int(hp[0]), y0), FONT, 0.55, col, 1,
                                cv2.LINE_AA)

                if np.isfinite(xy[k]).all():
                    people[tid] = (xy[k], med, bool(vis[k]))
                    trails[tid].append(xy[k])

        left = u if (u.shape[1], u.shape[0]) == (fw, fh) else cv2.resize(u, (fw, fh))
        cv2.rectangle(left, (0, 0), (fw, 34), (18, 18, 18), -1)
        banner = (f"{args.name}   frame {i}   track IDs (colour + number are the "
                  f"same identity)" if args.big_ids else
                  f"{args.name}   frame {i}   "
                  f"1 m grid + derived occluder map + metric stature")
        cv2.putText(left, banner, (12, 23), FONT, 0.6, (210, 210, 210), 1,
                    cv2.LINE_AA)
        if args.no_plan:
            canvas = left
        else:
            right = plan_view(scene, people, trails)
            canvas = np.zeros((max(fh, PANEL), fw + PANEL, 3), np.uint8)
            canvas[:fh, :fw] = left
            canvas[:PANEL, fw:] = right
        vw.write(canvas)

        i += 1
        if i % 250 == 0:
            print(f"       frame {i}  tracks {len(people)}", flush=True)

    cap.release()
    vw.release()
    print(f"[save] {out_p}  ({i} frames, {n_meas} metric height measurements, "
          f"{len(hist)} tracks)")


if __name__ == "__main__":
    main()
