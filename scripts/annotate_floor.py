"""
Draw the walkable floor and obstacle footprints for one camera.

Clicks are pixels; what gets saved is metres on the ground plane, so the plan
survives a resolution change and drops straight into `reachability` without
re-projection. The conversion uses the commissioned `scene.json` -- run
`scripts/commission_scene.py` first, and sanity-check `08_metre_check.png`
before drawing, because every polygon inherits that metre scale.

    python scripts/annotate_floor.py \
        --video clip.mp4 \
        --scene calib/office_cam1 \
        --out   calib/office_cam1/floor_plan.json

Controls
    left click      add a vertex to the polygon in progress
    w               close it as WALKABLE   (floor a person can stand on)
    b               close it as BLOCKED    (a body in here is invisible to this camera)
    u               undo the last vertex, or the last polygon if none pending
    c               clear everything
    [ / ]           step the preview frame back / forward
    s               save and keep going
    q / Esc         save and quit

Draw one loose WALKABLE boundary around the room first, then punch out each
piece of furniture with BLOCKED. Anything outside every walkable polygon
becomes blocked, which is the point -- it replaces "unknown, traversable at a
penalty" with a real wall.

Click the point where the floor meets the object, not the top of it. A point
at or above the horizon has no floor intersection; the tool refuses those
rather than saving a plausible-looking wrong number.

If <scene>/scene_depth.npz exists (it does once you've run commission_scene.py),
closing a BLOCKED polygon also reads its height back from the depth model's
calibrated height field and reports low/waist/tall -- the footprint says
where, depth says how tall, and together they say whether it hides a whole
person or just their feet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from reid.floor_plan import FloorPlan, WALKABLE, BLOCKED_KIND
from reid.scene_depth import SceneModel
from reid.scene_geometry import GroundPlane

COL = {WALKABLE: (90, 220, 90), BLOCKED_KIND: (60, 60, 235)}
PENDING = (240, 200, 40)


# A click near (not at) the horizon still returns a finite floor_xy, because
# the ray is nearly parallel to the floor - a tiny pixel error there becomes
# tens of metres on the ground. floor_xy() only guards the exactly-degenerate
# case (>=horizon), so this guards the near-degenerate one: a floor point this
# far from every other vertex clicked so far is treated as a near-horizon
# misclick, not real geometry. 25 m is generous for any indoor room; it exists
# to catch the horizon blow-up, not to constrain legitimate obstacle size.
REJECT_DIST_M = 25.0


class Annotator:
    def __init__(self, frame: np.ndarray, gp: GroundPlane, camera: str, scene=None):
        self.base = frame
        self.gp = gp
        self.model = gp.radial_model()
        self.scene = scene   # SceneModel or None; enables the height readout on 'b'
        self.plan = FloorPlan(camera=camera)
        self.px: list[list[tuple[int, int]]] = []   # pixel copy, for drawing only
        self.pending: list[tuple[int, int]] = []
        self.pending_xy: list[np.ndarray] = []       # floor metres, parallel to pending
        self.msg = ""

    # ── geometry ────────────────────────────────────────────────────────────
    def to_floor(self, pts_px) -> np.ndarray:
        p = np.asarray(pts_px, float).reshape(-1, 2)
        u = self.model.undistort_points(p)
        return self.gp.floor_xy(np.hstack([u, np.ones((len(u), 1))]))

    def to_px(self, xy_m) -> list[tuple[int, int]]:
        px = self.gp.pixels_from_floor(xy_m)
        return [(int(round(x)), int(round(y)))
                for x, y in px if np.isfinite(x) and np.isfinite(y)]

    # ── editing ─────────────────────────────────────────────────────────────
    def click(self, x: int, y: int) -> None:
        xy = self.to_floor([[x, y]])[0]
        if not np.isfinite(xy).all():
            self.msg = "point is at/above the horizon - no floor there"
            return
        if np.linalg.norm(xy) > REJECT_DIST_M or any(
            np.linalg.norm(xy - p) > REJECT_DIST_M for p in self.pending_xy
        ):
            self.msg = (f"REJECTED: ({xy[0]:+.1f}, {xy[1]:+.1f}) m is implausibly far - "
                        f"click landed too close to the horizon. Click lower, on the floor.")
            return
        self.pending.append((x, y))
        self.pending_xy.append(xy)
        self.msg = f"vertex {len(self.pending)}  ({xy[0]:+.2f}, {xy[1]:+.2f}) m"

    def close(self, kind: str) -> None:
        if len(self.pending) < 3:
            self.msg = "need at least 3 vertices"
            return
        xy = self.to_floor(self.pending)
        try:
            self.plan.add(kind, xy)
        except ValueError as e:
            self.msg = str(e)
            return
        self.px.append(list(self.pending))
        self.pending.clear()
        self.pending_xy.clear()
        self.msg = f"closed {kind} polygon ({len(self.plan.polygons)} total)"
        if kind == BLOCKED_KIND and self.scene is not None:
            ht = self.plan.obstacle_heights(self.scene)[-1]
            if ht["cls"] == "unseen":
                self.msg += "  | depth never saw this spot -- height unknown"
            else:
                self.msg += (f"  | depth says: {ht['cls']}  "
                             f"(p50={ht['height_p50_m']:.2f}m p90={ht['height_p90_m']:.2f}m)")

    def undo(self) -> None:
        if self.pending:
            self.pending.pop()
            self.pending_xy.pop()
            self.msg = "vertex removed"
        elif self.plan.polygons:
            self.plan.polygons.pop()
            self.px.pop()
            self.msg = "polygon removed"

    def clear(self) -> None:
        self.pending.clear()
        self.pending_xy.clear()
        self.plan.polygons.clear()
        self.px.clear()
        self.msg = "cleared"

    # ── drawing ─────────────────────────────────────────────────────────────
    def render(self) -> np.ndarray:
        img = self.base.copy()
        overlay = img.copy()
        for poly, meta in zip(self.px, self.plan.polygons):
            pts = np.array(poly, np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(overlay, [pts], COL[meta["kind"]])
            cv2.polylines(img, [pts], True, COL[meta["kind"]], 2)
        cv2.addWeighted(overlay, 0.28, img, 0.72, 0, img)

        for i, p in enumerate(self.pending):
            cv2.circle(img, p, 4, PENDING, -1)
            if i:
                cv2.line(img, self.pending[i - 1], p, PENDING, 2)

        n_w = len(self.plan.of_kind(WALKABLE))
        n_b = len(self.plan.of_kind(BLOCKED_KIND))
        bar = [
            f"walkable {n_w}   blocked {n_b}   pending {len(self.pending)}",
            "click=vertex  w=walkable  b=blocked  u=undo  c=clear  [ ]=frame  s=save  q=quit",
        ]
        if self.msg:
            bar.append(self.msg)
        y = 24
        for line in bar:
            cv2.putText(img, line, (11, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
            cv2.putText(img, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y += 24
        return img


def grab(cap, idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    return f if ok else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="clip to draw on")
    ap.add_argument("--scene", required=True, help="commissioned calib dir with scene.json")
    ap.add_argument("--out", default=None, help="default: <scene>/floor_plan.json")
    ap.add_argument("--frame", type=int, default=0, help="starting frame")
    ap.add_argument("--step", type=int, default=30, help="frames per [ or ] press")
    args = ap.parse_args()

    sdir = Path(args.scene)
    out = Path(args.out) if args.out else sdir / "floor_plan.json"
    gp = GroundPlane.load(sdir / "scene.json")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = max(0, min(args.frame, total - 1))
    frame = grab(cap, idx)
    if frame is None:
        sys.exit("cannot read a frame")

    h, w = frame.shape[:2]
    if (w, h) != (gp.width, gp.height):
        print(f"[WARN] video is {w}x{h} but scene.json was calibrated at "
              f"{gp.width}x{gp.height}; clicks will project wrong.")

    scene = None
    depth_path = sdir / "scene_depth.npz"
    if depth_path.exists():
        scene = SceneModel.load(depth_path, gp)
        print(f"[INFO] scene depth loaded ({depth_path}); 'b' polygons will "
              f"show a height readout (floor_resid_m={scene.floor_resid_m:.2f})")
    else:
        print(f"[INFO] no {depth_path.name} in {sdir} -- height readout on 'b' disabled")

    ann = Annotator(frame, gp, camera=sdir.name, scene=scene)
    if out.exists():
        ann.plan = FloorPlan.load(out)
        ann.px = [ann.to_px(np.asarray(p["xy"], float)) for p in ann.plan.polygons]
        print(f"[INFO] loaded {len(ann.plan.polygons)} existing polygons from {out}")

    win = f"floor plan - {sdir.name}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(w, 1600), min(h, 900))
    cv2.setMouseCallback(
        win, lambda e, x, y, f, p: ann.click(x, y) if e == cv2.EVENT_LBUTTONDOWN else None
    )

    print(__doc__.split("Controls")[1].split("Draw one")[0])
    while True:
        cv2.imshow(win, ann.render())
        k = cv2.waitKey(20) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("w"):
            ann.close(WALKABLE)
        elif k == ord("b"):
            ann.close(BLOCKED_KIND)
        elif k == ord("u"):
            ann.undo()
        elif k == ord("c"):
            ann.clear()
        elif k in (ord("["), ord("]")):
            idx = max(0, min(idx + (args.step if k == ord("]") else -args.step), total - 1))
            f = grab(cap, idx)
            if f is not None:
                ann.base = f
                ann.msg = f"frame {idx}"
        elif k == ord("s"):
            ann.plan.save(out)
            ann.msg = f"saved -> {out}"
            print(f"[save] {out}")

    cap.release()
    cv2.destroyAllWindows()
    if not ann.plan.polygons:
        print("[INFO] nothing drawn, nothing saved")
        return
    ann.plan.save(out)
    print(f"[save] {out}  ({len(ann.plan.of_kind(WALKABLE))} walkable, "
          f"{len(ann.plan.of_kind(BLOCKED_KIND))} blocked)")
    print(f"[next] python scripts/track_rtdetr_db.py --scene {sdir} --floor-plan {out} ...")


if __name__ == "__main__":
    main()
