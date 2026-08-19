"""
Step 10 — "appearance only works if we compare the same pose"

The user's claim, in full: *"if we track appearance only the model is not as
versatile, and appearance works well only if we know the pose - compared exact
pose. Generally we need to understand the space and the movement of the human
inside the vid in order to track them."*

The first half is testable. If the embedding is largely encoding *which side of
the person the camera can see*, then same-person pairs observed from a similar
viewing aspect should agree far better than same-person pairs seen from
opposite aspects - and the aspect can be recovered for free, because step 9
already gives us metric ground-plane position and velocity.

Viewing aspect here is the angle between where the person is *walking* and the
direction the camera is looking at them from. The camera's own floor position is
the origin of `GroundPlane.floor_xy`, so the camera->person direction is just the
normalized floor position.

    aspect ~ 0    walking away from camera   -> we see their back
    aspect ~ pi/2 walking across the view    -> profile
    aspect ~ pi   walking toward the camera  -> we see their face

Run:
    python bench/10_eval_pose_conditioning.py --scene calib/gunsan_test \\
        --traj runs/traj/gunsan_test.json --labels labels/gunsan_test.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from importlib import import_module

from reid.scene_depth import SceneModel, floor_from_boxes
from reid.scene_geometry import GroundPlane

embed_all = import_module("03_eval_embeddings").embed_all
DATA = Path(__file__).resolve().parent / "data"

VEL_WIN = 5          # frames each side used for the heading estimate
MIN_SPEED_MS = 0.15  # below this the heading is noise, not a direction


def headings(gp, scene, traj):
    """{track_id: {frame: aspect_radians}} — NaN where the person is too slow."""
    out = {}
    for tid, recs in traj.items():
        recs = sorted(recs, key=lambda r: r[0])
        f = np.array([r[0] for r in recs], dtype=float)
        boxes = np.array([r[1:5] for r in recs], dtype=float)
        xy, _vis = floor_from_boxes(gp, scene, boxes)
        asp = {}
        for i in range(len(f)):
            a, b = max(0, i - VEL_WIN), min(len(f) - 1, i + VEL_WIN)
            d, dt = xy[b] - xy[a], (f[b] - f[a]) / 15.0
            if dt <= 0 or not np.all(np.isfinite(d)) or not np.all(np.isfinite(xy[i])):
                continue
            v = d / dt
            if np.hypot(*v) < MIN_SPEED_MS:
                continue           # standing still has no heading
            look = xy[i] / (np.linalg.norm(xy[i]) + 1e-9)   # camera -> person
            h = v / (np.linalg.norm(v) + 1e-9)
            asp[int(f[i])] = float(np.arccos(np.clip(h @ look, -1, 1)))
        out[int(tid)] = asp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="calib/gunsan_test")
    ap.add_argument("--traj", default="runs/traj/gunsan_test.json")
    ap.add_argument("--labels", default="labels/gunsan_test.json")
    ap.add_argument("--reid", default="models/clip_market1501.pt")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    scene_dir = Path(args.scene)
    gp = GroundPlane.load(scene_dir / "scene.json")
    scene = SceneModel.load(scene_dir / "scene_depth.npz", gp)
    traj = json.load(open(args.traj))["traj"]
    labels = {int(k): v for k, v in json.load(open(args.labels)).items()}

    asp = headings(gp, scene, traj)

    man = json.load(open(DATA / "manifest.json"))
    rows = [r for r in man
            if r["trk"] in labels and r["frame"] in asp.get(r["trk"], {})]
    print(f"[INFO] {len(rows)} crops with both a person label and a heading "
          f"(the rest were standing still or off the floor)")

    embs = embed_all(args.reid, [DATA / "crops" / f"crop_{r['cid']:06d}.jpg"
                                 for r in rows], args.device)
    trk = np.array([r["trk"] for r in rows])
    per = np.array([labels[r["trk"]] for r in rows])
    ang = np.array([asp[r["trk"]][r["frame"]] for r in rows])
    D = 1.0 - embs @ embs.T

    same_x, diff_x = [], []   # (embedding distance, aspect difference)
    for i, j in itertools.combinations(range(len(rows)), 2):
        da = abs(float(ang[i] - ang[j]))
        rec = (float(D[i, j]), da)
        if per[i] != per[j]:
            diff_x.append(rec)
        elif trk[i] != trk[j]:
            same_x.append(rec)
    same_x, diff_x = np.array(same_x), np.array(diff_x)

    def auc(pos, neg):
        x = np.concatenate([-pos, -neg])
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
        return (r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

    print("\n=== does matching the viewing pose rescue the embedding? ===")
    print("  SAME = same person, different tracklets (the case that matters).")
    print(f"  {'aspect diff':<16}{'n same':>8}{'med d':>8}{'n diff':>9}"
          f"{'med d':>8}{'AUC':>8}")
    bins = [(0, 30), (30, 60), (60, 90), (90, 180)]
    for lo, hi in bins:
        l, h = np.deg2rad(lo), np.deg2rad(hi)
        s = same_x[(same_x[:, 1] >= l) & (same_x[:, 1] < h)][:, 0]
        d = diff_x[(diff_x[:, 1] >= l) & (diff_x[:, 1] < h)][:, 0]
        if len(s) < 20 or len(d) < 20:
            continue
        print(f"  {f'{lo}-{hi} deg':<16}{len(s):>8}{np.median(s):>8.3f}"
              f"{len(d):>9}{np.median(d):>8.3f}{auc(s, d):>8.3f}")
    print(f"  {'all':<16}{len(same_x):>8}{np.median(same_x[:,0]):>8.3f}"
          f"{len(diff_x):>9}{np.median(diff_x[:,0]):>8.3f}"
          f"{auc(same_x[:,0], diff_x[:,0]):>8.3f}")

    # ── CONTROL ────────────────────────────────────────────────────────────
    # The table above is flat, which admits two readings: either pose does not
    # drive the embedding, or the aspect estimate is noise. Within ONE tracklet
    # the person and the outfit are fixed, so aspect is the only variable left.
    # If pose matters at all, it must show here — and if it does not, the
    # instrument is broken and the table above means nothing.
    ctl = []
    for i, j in itertools.combinations(range(len(rows)), 2):
        if trk[i] == trk[j]:
            ctl.append((float(D[i, j]), abs(float(ang[i] - ang[j]))))
    ctl = np.array(ctl)
    print("\n=== control: same tracklet — same person, same outfit ===")
    print(f"  {'aspect diff':<16}{'n':>8}{'med d':>8}")
    for lo, hi in bins:
        m = (ctl[:, 1] >= np.deg2rad(lo)) & (ctl[:, 1] < np.deg2rad(hi))
        if m.sum() > 20:
            print(f"  {f'{lo}-{hi} deg':<16}{m.sum():>8}{np.median(ctl[m,0]):>8.3f}")
    print(f"  correlation(aspect diff, distance) = "
          f"{np.corrcoef(ctl[:,1], ctl[:,0])[0,1]:+.3f}")
    print("\n  Control rising while the cross-tracklet table stays flat means")
    print("  pose is real but second-order: across tracklets the clothing")
    print("  change is the larger effect and buries it. Pose gating would help")
    print("  short-gap matching and not the re-entry case it was wanted for.")


if __name__ == "__main__":
    main()
