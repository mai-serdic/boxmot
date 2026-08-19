"""
Step 11 — how much identity is in metric stature alone?

Steps 10 and 9 leave one clothing-invariant cue built but unused: `StatureField`
gives a metric height per observation, corrected for the position-dependent bias
that made raw stature useless. The ghost pool has a `w_height` term, but it
scores *pixel bbox height*, which is a function of where the person stands as
much as who they are — so the metric version is not reaching the association
logic at all, the same failure step 9 fixed for geometry.

Before touching the scorer: does stature actually separate these people? A cue
that cannot tell 4 people apart is not worth wiring in, no matter how
clothing-invariant it is.

Reported the same way as the appearance benchmark so the two are comparable:
per-tracklet estimate, then pairwise SAME (same person, different tracklets) vs
DIFF (different people) discrimination.

Run:
    python bench/11_eval_stature_cue.py --scene calib/gunsan_test \\
        --traj runs/traj/gunsan_test.json --labels labels/gunsan_test.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reid.scene_depth import SceneModel, StatureField, localize
from reid.scene_geometry import GroundPlane

MIN_OBS = 8          # tracklets with fewer usable statures are not estimated


def tracklet_statures(gp, scene, field, traj, stature_prior_m=1.70):
    """{track_id: (median_stature_m, n_used, iqr)} using feet-visible frames only."""
    model = gp.radial_model()
    out = {}
    for tid, recs in traj.items():
        recs = sorted(recs, key=lambda r: r[0])
        b = np.array([r[1:5] for r in recs], dtype=float)
        if not len(b):
            continue
        cx = 0.5 * (b[:, 0] + b[:, 2])
        foot, head = np.column_stack([cx, b[:, 3]]), np.column_stack([cx, b[:, 1]])

        def _nh(px):
            u = model.undistort_norm(model.to_norm(px))
            return np.column_stack([u, np.ones(len(u))])

        foot_u = model.to_pixel(model.undistort_norm(model.to_norm(foot)))
        _xy, st, vis = localize(gp, scene, foot_u, _nh(foot), _nh(head),
                                stature_prior_m)
        st = field.apply(foot_u, st, gp)          # position-bias corrected
        # Feet must be visible: an inferred foot point makes stature circular,
        # since the head is what was used to place the foot in the first place.
        good = vis & np.isfinite(st) & (st > 0.9) & (st < 2.4)
        if good.sum() < MIN_OBS:
            continue
        v = st[good]
        out[int(tid)] = (float(np.median(v)), int(good.sum()),
                         float(np.percentile(v, 75) - np.percentile(v, 25)))
    return out


def auc(pos, neg):
    """pos = same person (want SMALL difference), so score on the negated diff."""
    x = np.concatenate([-np.asarray(pos), -np.asarray(neg)])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(1, len(x) + 1)
    return (r[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="calib/gunsan_test")
    ap.add_argument("--traj", default="runs/traj/gunsan_test.json")
    ap.add_argument("--labels", default="labels/gunsan_test.json")
    args = ap.parse_args()

    sdir = Path(args.scene)
    gp = GroundPlane.load(sdir / "scene.json")
    scene = SceneModel.load(sdir / "scene_depth.npz", gp)
    field = StatureField.from_dict(json.load(open(sdir / "stature_field.json")))
    traj = json.load(open(args.traj))["traj"]
    labels = {int(k): v for k, v in json.load(open(args.labels)).items()}

    est = tracklet_statures(gp, scene, field, traj)
    est = {k: v for k, v in est.items() if k in labels}
    print(f"[INFO] {len(est)} of {len(labels)} labelled tracklets got a stature "
          f"(the rest had <{MIN_OBS} feet-visible frames)")

    byp = defaultdict(list)
    for t, (m, n, iqr) in est.items():
        byp[labels[t]].append((t, m, n, iqr))

    print("\n=== per person ===")
    print(f"  {'person':<9}{'tracklets':>10}{'median h':>10}{'spread':>9}"
          f"{'per-tracklet IQR':>19}")
    for p in sorted(byp):
        v = byp[p]
        ms = np.array([x[1] for x in v])
        print(f"  {p:<9}{len(v):>10}{np.median(ms):>10.3f}"
              f"{ms.max()-ms.min():>9.3f}{np.median([x[3] for x in v]):>19.3f}")
    print("  'spread' = disagreement between tracklets of the SAME person, i.e.")
    print("  the noise floor. It has to be smaller than the gaps between people")
    print("  for stature to be usable at all.")

    same, diff = [], []
    ts = sorted(est)
    for a, b in itertools.combinations(ts, 2):
        d = abs(est[a][0] - est[b][0])
        (same if labels[a] == labels[b] else diff).append(d)
    print(f"\n=== pairwise, same protocol as the appearance benchmark ===")
    print(f"  {'set':<34}{'n':>7}{'p50 |dh|':>10}{'p90':>8}")
    print(f"  {'SAME person, across tracklets':<34}{len(same):>7}"
          f"{np.median(same):>10.3f}{np.percentile(same,90):>8.3f}")
    print(f"  {'DIFFERENT people':<34}{len(diff):>7}"
          f"{np.median(diff):>10.3f}{np.percentile(diff,90):>8.3f}")
    a = auc(same, diff)
    print(f"\n  stature AUC (same vs different) : {a:.3f}")
    print(f"  appearance AUC, same regime     : 0.803   (step 10)")
    print("\n  Stature is independent of clothing and of pose, so even a weaker")
    print("  AUC is worth having *if* it fails on different pairs than")
    print("  appearance does — that is the question the scorer cares about,")
    print("  not which single cue is stronger.")

    out = Path(__file__).resolve().parent / "data" / "stature_identity.json"
    json.dump({"est": {str(k): v for k, v in est.items()},
               "same": same, "diff": diff, "auc": a}, open(out, "w"))
    print(f"\n[INFO] wrote {out}")


if __name__ == "__main__":
    main()
