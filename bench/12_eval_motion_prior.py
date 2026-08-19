"""
Step 12 — does predicting *movement* beat predicting *reachability*?

Step 9's prior answers "could they be there". The user asked for something
stronger: "predict the movement and direction of the tracked human". Those are
different questions, and only the first is currently implemented — after 0.6 s of
straight-line lookahead the geodesic belief is an isotropic blob.

`motion_prior.MotionPrior` replaces that with a real forward simulation: the
state is (cell, heading), so momentum is represented, and the turn distribution
is learned per cell from the tracker's own output, so the room's traffic lanes
steer the walker. Propagating it for the actual elapsed time gives P(cell | last
position, last heading, elapsed) instead of a distance-decay function.

Same ranking protocol as step 8, so the three priors are directly comparable:
one ghost, one elapsed time, the candidates alive in that frame, pick one.

Run:
    python bench/12_eval_motion_prior.py --scene calib/gunsan_test \\
        --traj runs/traj/gunsan_test.json --labels labels/gunsan_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reid.ghost_pool import ScoringWeights, _spatial_prior_px
from reid.motion_prior import MotionPrior, heading_of
from reid.reachability import ReachParams, Reachability, rebind_prior
from reid.scene_depth import SceneModel
from reid.scene_geometry import GroundPlane

_b8 = import_module("08_eval_geodesic_prior")
build_tracks, make_query, make_cand = _b8.build_tracks, _b8.make_query, _b8.make_cand

BANDS = ((5, 20), (20, 60), (60, 150), (150, 600))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="calib/gunsan_test")
    ap.add_argument("--traj", default="runs/traj/gunsan_test.json")
    ap.add_argument("--labels", default="labels/gunsan_test.json")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--gap-min", type=int, default=5)
    ap.add_argument("--gap-max", type=int, default=600)
    args = ap.parse_args()

    sdir = Path(args.scene)
    gp = GroundPlane.load(sdir / "scene.json")
    scene = SceneModel.load(sdir / "scene_depth.npz", gp)
    tracks = build_tracks(json.load(open(args.traj)), gp, scene)
    labels = {int(k): v for k, v in json.load(open(args.labels)).items()}
    labels = {k: v for k, v in labels.items() if k in tracks}

    reach = Reachability.build(scene)
    for t in tracks.values():
        ok = t["vis"] & np.all(np.isfinite(t["xy"]), axis=1)
        if ok.any():
            reach.observe(t["xy"][ok])
    print(f"[INFO] {reach.summary()}")

    # The flow field is fitted on tracklets, which are *not* the thing being
    # predicted here — we predict re-entry across tracklet boundaries, and a
    # tracklet by construction contains no boundary. So this is not training on
    # the test set, but it is the same clip, and that is stated in the report.
    mp = MotionPrior.fit(reach, [(t["f"], t["xy"]) for t in tracks.values()],
                         args.fps)
    print(f"[INFO] {mp.summary()}")

    tids = sorted(tracks)
    same = []
    for i in tids:
        for j in tids:
            if i == j or labels.get(i) is None or labels.get(i) != labels.get(j):
                continue
            a, b = tracks[i], tracks[j]
            if b["f"][0] <= a["f"][-1]:
                continue
            k = len(a["f"]) - 1
            for m in range(min(len(b["f"]), 40)):
                if args.gap_min <= b["f"][m] - a["f"][k] <= args.gap_max:
                    same.append((a, k, b, m))
    # Honesty about sample size: each re-entry event contributes up to 40 pairs
    # because the candidate is sampled at 40 different offsets into the new
    # tracklet. The pair count is therefore NOT the evidence count, and every
    # accuracy figure in this file and in step 8 rests on the smaller number.
    ev = len({(id(a), id(b)) for a, _k, b, _m in same})
    print(f"[INFO] {len(same)} cross-tracklet pairs from {ev} independent "
          f"re-entry events ({len(same)/max(ev,1):.0f}x resampling)")

    # How far do people actually go while they are lost? If the answer is
    # "nowhere", a movement model has nothing to predict.
    net = defaultdict(list)
    for tg, k, tc, m in same:
        gap = int(tc["f"][m]) - int(tg["f"][k])
        b = next((bd for bd in BANDS if bd[0] <= gap < bd[1]), None)
        pa, pb = tg["xy"][k], tc["xy"][m]
        if b and np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
            net[b].append(float(np.hypot(*(pb - pa))))
    print("\n=== how far does a lost person actually travel? ===")
    for b in BANDS:
        if net[b]:
            print(f"  {b[0]:>3}-{b[1]:>3}f ({b[0]/args.fps:>4.1f}-{b[1]/args.fps:>4.1f}s): "
                  f"median {np.median(net[b]):.2f} m, p90 {np.percentile(net[b],90):.2f} m")
    print(f"  for scale, a 0.6 m/s walker unimpeded for 40 s covers 24 m.")

    by_frame = defaultdict(list)
    for tid in tids:
        for i, f in enumerate(tracks[tid]["f"]):
            by_frame[int(f)].append((tid, i))
    person_of = {id(tracks[t]): labels.get(t) for t in tids}

    w, par = ScoringWeights(), ReachParams()

    def score_all(g, c, cur, gap_s):
        """(pixel, geodesic, motion) prior for one ghost/candidate pair."""
        px = _spatial_prior_px(g, c, cur, w)
        geo, mot = np.nan, np.nan
        if g.last_xy is not None and c.xy is not None:
            r = rebind_prior(reach, g.last_xy, g.last_vel_xy, c.xy, gap_s, par)
            geo = r["prior"] if r["feasible"] else 0.0
            h = heading_of(g.last_vel_xy or (0.0, 0.0))
            a, b = reach.cell_of(g.last_xy), reach.cell_of(c.xy)
            if a >= 0 and b >= 0 and r["feasible"]:
                p = mp.predict(reach, a, h if h >= 0 else 0, gap_s)
                # Normalise against the best cell so the number is a *relative*
                # plausibility like the other two, not an absolute probability
                # that shrinks as the belief spreads.
                mot = float(p[b] / max(p.max(), 1e-12))
            else:
                mot = 0.0
        return px, geo, mot

    wins = {k: defaultdict(lambda: [0, 0]) for k in ("pixel", "geodesic", "motion")}
    for tg, k, tc, m in same:
        cur = int(tc["f"][m])
        gap = cur - int(tg["f"][k])
        band = next((b for b in BANDS if b[0] <= gap < b[1]), None)
        if band is None:
            continue
        others = [(tid, i) for tid, i in by_frame.get(cur, [])
                  if tracks[tid] is not tg and tracks[tid] is not tc
                  and labels.get(tid) != person_of.get(id(tg))]
        if not others:
            continue
        g = make_query(tg, k, args.fps)
        gap_s = gap / args.fps
        true_s = score_all(g, make_cand(tc, m), cur, gap_s)
        dis = [score_all(g, make_cand(tracks[tid], i), cur, gap_s)
               for tid, i in others]
        for n, name in enumerate(("pixel", "geodesic", "motion")):
            best = max((d[n] for d in dis), default=-np.inf)
            v = true_s[n]
            if not np.isfinite(v):
                continue
            wins[name][band][1] += 1
            wins[name][band][0] += int(v > best)

    counts = {b: wins["geodesic"][b][1] for b in BANDS}
    tot = sum(counts.values()) or 1
    print("\n=== ranking: true candidate vs distractors alive in the same frame ===")
    print("  gap mix: " + "  ".join(f"{lo}-{hi}f {100*counts[(lo,hi)]/tot:.0f}%"
                                    for lo, hi in BANDS))
    hdr = "".join(f"{f'{lo}-{hi}f':>10}" for lo, hi in BANDS)
    print(f"\n  {'spatial prior':<16}{hdr}{'weighted':>11}")
    for name in ("pixel", "geodesic", "motion"):
        row, acc = "", 0.0
        for b in BANDS:
            hit, n = wins[name][b]
            r = hit / n if n else float("nan")
            row += f"{100*r:>9.1f}%"
            acc += (r if n else 0.0) * counts[b] / tot
        print(f"  {name:<16}{row}{100*acc:>10.1f}%")
    print("\n  'motion' = learned (cell, heading) chain propagated for the actual")
    print("  elapsed time. 'geodesic' = what ships today.")


if __name__ == "__main__":
    main()
