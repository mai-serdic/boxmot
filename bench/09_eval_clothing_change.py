"""
Step 9 — does the appearance embedding survive a clothing change?

The user's observation: in this clip people change clothes, and real workers do
it constantly (jackets on/off, PPE, shift uniforms). If true, the embedding is
not measuring identity, it is measuring garments — and the *embedding veto* in
ghost_pool, which kills any rebind whose cosine distance exceeds
`embedding_veto_max_dist`, would be actively blocking correct re-identifications
rather than blocking impostors.

That is a claim about a shipped default, so it needs measuring, not assuming.

Method: with tracklet-level person labels we can split same-person crop pairs by
whether they sit inside one tracklet (seconds apart, same outfit by
construction) or across two tracklets (minutes apart, outfit unknown). If the
embedding tracks identity, the two distributions are similar. If it tracks
clothing, the cross-tracklet one splits in two: a low-distance mode (same
outfit) and a high-distance mode (changed).

Run:
    python bench/09_eval_clothing_change.py --labels labels/gunsan_test.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from importlib import import_module

_m3 = import_module("03_eval_embeddings")
embed_all = _m3.embed_all

DATA = Path(__file__).resolve().parent / "data"


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="labels/gunsan_test.json")
    ap.add_argument("--reid", default="models/clip_market1501.pt")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--veto", type=float, default=0.55,
                    help="ghost_pool ScoringWeights.embedding_veto_max_dist")
    args = ap.parse_args()

    man = json.load(open(DATA / "manifest.json"))
    labels = {int(k): v for k, v in json.load(open(args.labels)).items()}

    rows = [r for r in man if r["trk"] in labels]
    print(f"[INFO] {len(rows)} crops on {len(set(r['trk'] for r in rows))} "
          f"labelled tracklets, {len(set(labels[r['trk']] for r in rows))} people")

    paths = [DATA / "crops" / f"crop_{r['cid']:06d}.jpg" for r in rows]
    embs = embed_all(args.reid, paths, args.device)

    trk = np.array([r["trk"] for r in rows])
    per = np.array([labels[r["trk"]] for r in rows])
    frm = np.array([r["frame"] for r in rows])

    D = 1.0 - embs @ embs.T

    within, cross, diff = [], [], []
    n = len(rows)
    for i, j in itertools.combinations(range(n), 2):
        d = float(D[i, j])
        if per[i] != per[j]:
            diff.append(d)
        elif trk[i] == trk[j]:
            within.append(d)
        else:
            cross.append((d, abs(int(frm[i] - frm[j])), per[i]))

    cross_d = np.array([c[0] for c in cross])
    within = np.array(within)
    diff = np.array(diff)

    print("\n=== cosine distance, same identity vs different ===")
    print(f"  {'set':<34}{'n':>7}{'p10':>8}{'p50':>8}{'p90':>8}")
    for name, x in (("SAME person, within one tracklet", within),
                    ("SAME person, across tracklets", cross_d),
                    ("DIFFERENT people", diff)):
        print(f"  {name:<34}{len(x):>7}{pct(x,10):>8.3f}"
              f"{pct(x,50):>8.3f}{pct(x,90):>8.3f}")

    print("\n=== is the cross-tracklet mode split? ===")
    print("  If the embedding encoded identity, cross-tracklet SAME would sit")
    print("  near within-tracklet SAME. A second mode up in DIFFERENT territory")
    print("  means those pairs are the same body in different clothes.")
    thr = pct(diff, 10)
    hi = cross_d >= thr
    print(f"  DIFFERENT-people p10 = {thr:.3f}  (below this, pairs look alike)")
    print(f"  cross-tracklet SAME pairs at/above it: "
          f"{100*hi.mean():.1f} %  <- indistinguishable from strangers")

    print(f"\n=== what the embedding veto (>{args.veto:.2f}) does ===")
    for name, x in (("SAME within-tracklet", within),
                    ("SAME across tracklets", cross_d),
                    ("DIFFERENT people", diff)):
        print(f"  {name:<24} vetoed: {100*(x > args.veto).mean():>5.1f} %")
    print("  The middle row is the cost: true re-entries the veto refuses.")
    print("  The last row is the benefit: impostors it blocks.")

    print("\n=== per person, worst cross-tracklet pair ===")
    byp = defaultdict(list)
    for d, gap, p in cross:
        byp[p].append((d, gap))
    for p in sorted(byp):
        v = sorted(byp[p], reverse=True)
        print(f"  person {p}: {len(v):>4} pairs, max d={v[0][0]:.3f} "
              f"@ {v[0][1]} frames apart, median d={np.median([x[0] for x in v]):.3f}")

    out = DATA / "clothing_change.json"
    json.dump({"within": within.tolist(), "cross": cross_d.tolist(),
               "diff": diff.tolist(), "veto": args.veto}, open(out, "w"))
    print(f"\n[INFO] wrote {out}")


if __name__ == "__main__":
    main()
