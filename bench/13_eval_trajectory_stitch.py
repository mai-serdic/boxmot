"""
Step 13 — is a whole-path model better than a chain of pairwise guesses?

The user: *"iou only track consecutive frames is really limited, it should track
the whole motion path of one person."*

`trajectory_stitch` implements that: tracklets become nodes in a time-ordered
graph, and each unit of min-cost flow from source to sink is one person's path
through the space. Costs come from the step 9 geodesic prior, so the scene model
is what holds the paths together, and the assignment is solved globally rather
than edge by edge.

Measured against the three things a greedy matcher structurally cannot do —
transitivity, exclusivity, revision — using the labels as ground truth.

Metrics, all tracklet-weighted by observation count so a 200-frame tracklet
counts more than a 10-frame one:
  * identities found vs the true 4
  * fragmentation: identities per real person, 1.0 is perfect
  * purity: are the tracklets grouped under one identity really one person
  * IDF1: the standard single number, harmonic mean of ID-precision/recall

Run:
    python bench/13_eval_trajectory_stitch.py --scene calib/gunsan_test \\
        --traj runs/traj/gunsan_test.json --labels labels/gunsan_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from importlib import import_module
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reid.reachability import ReachParams, Reachability
from reid.scene_depth import SceneModel
from reid.scene_geometry import GroundPlane
from reid.trajectory_stitch import Tracklet, stitch_global, stitch_greedy

build_tracks = import_module("08_eval_geodesic_prior").build_tracks


def evaluate(owner: dict[int, int], labels: dict[int, str],
             weight: dict[int, int]) -> dict:
    """Cluster quality against labels, weighted by observations."""
    ids = [t for t in owner if t in labels]
    if not ids:
        return {}
    by_pred, by_true = defaultdict(list), defaultdict(list)
    for t in ids:
        by_pred[owner[t]].append(t)
        by_true[labels[t]].append(t)
    W = lambda ts: sum(weight[t] for t in ts)

    # Purity: within each predicted identity, the weight of its majority person.
    pure = sum(max(Counter({p: W([t for t in ts if labels[t] == p])
                            for p in {labels[t] for t in ts}}).values())
               for ts in by_pred.values())
    total = W(ids)

    # IDF1 over a greedy best-match between predicted and true identities.
    tp = 0
    used = set()
    for _p, ts in sorted(by_true.items(), key=lambda kv: -W(kv[1])):
        best, bw = None, 0
        for q, qs in by_pred.items():
            if q in used:
                continue
            w = W(set(ts) & set(qs))
            if w > bw:
                best, bw = q, w
        if best is not None:
            used.add(best); tp += bw
    idp = tp / total
    idr = tp / total
    return {
        "n_ident": len(by_pred),
        "frag": len(by_pred) / len(by_true),
        "purity": pure / total,
        "idf1": 2 * idp * idr / (idp + idr) if idp + idr else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="calib/gunsan_test")
    ap.add_argument("--traj", default="runs/traj/gunsan_test.json")
    ap.add_argument("--labels", default="labels/gunsan_test.json")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--birth", type=float, default=2.0)
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

    tl = [Tracklet(tid, np.asarray(t["f"]), t["xy"]) for tid, t in tracks.items()]
    weight = {tid: len(t["f"]) for tid, t in tracks.items()}
    n_people = len(set(labels.values()))
    print(f"[INFO] {len(tl)} tracklets, {len(labels)} labelled, "
          f"{n_people} real people")

    par = ReachParams()
    rows = []
    rows.append(("no stitching (raw tracklets)",
                 {t.tid: t.tid for t in tl}))
    rows.append((f"greedy pairwise (ghost pool)",
                 stitch_greedy(reach, tl, args.fps, par)))
    rows.append((f"global min-cost flow",
                 stitch_global(reach, tl, args.fps, par, birth_cost=args.birth)))

    print(f"\n=== stitching tracklets into whole paths ===")
    print(f"  {'method':<32}{'identities':>12}{'frag':>8}{'purity':>9}{'IDF1':>8}")
    for name, owner in rows:
        m = evaluate(owner, labels, weight)
        print(f"  {name:<32}{m['n_ident']:>12}{m['frag']:>8.2f}"
              f"{100*m['purity']:>8.1f}%{100*m['idf1']:>7.1f}%")
    print(f"  {'(perfect)':<32}{n_people:>12}{1.0:>8.2f}{100.0:>8.1f}%{100.0:>7.1f}%")

    print("\n  'frag' = identities per real person; 1.0 means each person was")
    print("  recovered as exactly one path. 'purity' = weight of the majority")
    print("  person inside each predicted identity — it falls when stitching")
    print("  merges two different people, which is the failure that matters.")

    print("\n=== birth-cost sensitivity (the one knob) ===")
    print(f"  {'birth':>7}{'identities':>12}{'frag':>8}{'purity':>9}{'IDF1':>8}")
    for b in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        m = evaluate(stitch_global(reach, tl, args.fps, par, birth_cost=b),
                     labels, weight)
        print(f"  {b:>7.1f}{m['n_ident']:>12}{m['frag']:>8.2f}"
              f"{100*m['purity']:>8.1f}%{100*m['idf1']:>7.1f}%")


if __name__ == "__main__":
    main()
