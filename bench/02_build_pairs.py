"""
Step 2: turn tracklets into a labelled verification set.

POSITIVE pair = two crops of the SAME tracklet, at least --min-gap frames apart
                (so they differ in pose / viewpoint / side — the failure modes
                we actually care about, not near-duplicate adjacent frames).

NEGATIVE pair = two crops from DIFFERENT tracklets that were active in the SAME
                frame at least once (co-active ⇒ two bodies on screen at once ⇒
                guaranteed different people). We deliberately do NOT pair two
                non-overlapping tracklets, because those *could* be the same
                person fragmented — that would inject label noise.

Balanced: negatives are subsampled to the positive count. Seeded for
reproducibility.
"""

import argparse
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


def run(args):
    d = Path(args.datadir)
    manifest = json.loads((d / "manifest.json").read_text())
    active = {int(k): set(v) for k, v in
              json.loads((d / "active_frames.json").read_text()).items()}
    rng = np.random.default_rng(args.seed)

    # cid -> record; trk -> [ (cid, frame) ]
    by_trk = defaultdict(list)
    for m in manifest:
        by_trk[m["trk"]].append((m["cid"], m["frame"]))

    # ── positives ────────────────────────────────────────────────────────────
    pos = []
    pos_gaps = []
    for trk, crops in by_trk.items():
        crops.sort(key=lambda x: x[1])
        valid = []
        for (ca, fa), (cb, fb) in combinations(crops, 2):
            if abs(fb - fa) >= args.min_gap:
                valid.append((ca, cb, abs(fb - fa)))
        if not valid:
            continue
        if len(valid) > args.max_pos_per_trk:
            idx = rng.choice(len(valid), args.max_pos_per_trk, replace=False)
            valid = [valid[i] for i in idx]
        for ca, cb, g in valid:
            pos.append([ca, cb])
            pos_gaps.append(g)

    # ── co-active track pairs (guaranteed-different) ─────────────────────────
    trks = list(active.keys())
    coactive = set()
    for a, b in combinations(trks, 2):
        if active[a] & active[b]:
            coactive.add((a, b))

    # negatives: for each co-active track pair, pair one crop from each
    neg_candidates = []
    for a, b in coactive:
        if not by_trk[a] or not by_trk[b]:
            continue
        ca = by_trk[a][rng.integers(len(by_trk[a]))][0]
        cb = by_trk[b][rng.integers(len(by_trk[b]))][0]
        neg_candidates.append([ca, cb])
    # top up with more crop combos from co-active pairs if we need more negatives
    extra = []
    for a, b in coactive:
        for (ca, _), (cb, _) in [(x, y) for x in by_trk[a][:4] for y in by_trk[b][:4]]:
            extra.append([ca, cb])
    rng.shuffle(extra)
    neg_pool = neg_candidates + extra

    n_target = len(pos)
    if len(neg_pool) > n_target:
        idx = rng.choice(len(neg_pool), n_target, replace=False)
        neg = [neg_pool[i] for i in idx]
    else:
        neg = neg_pool

    (d / "pairs.json").write_text(json.dumps({"pos": pos, "neg": neg}))

    gaps = np.array(pos_gaps) if pos_gaps else np.array([0])
    print(f"[PAIRS] tracks total={len(by_trk)}  co-active track-pairs={len(coactive)}")
    print(f"[PAIRS] positives={len(pos)}  negatives={len(neg)}")
    print(f"[PAIRS] positive frame-gap: median={np.median(gaps):.0f}  "
          f"p90={np.percentile(gaps, 90):.0f}  max={gaps.max():.0f} frames "
          f"(@15fps → median {np.median(gaps)/15:.1f}s, max {gaps.max()/15:.1f}s apart)")
    print(f"[PAIRS] wrote {d/'pairs.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", required=True)
    p.add_argument("--min-gap", type=int, default=30,
                   help="min frame separation for a positive pair (30f@15fps=2s)")
    p.add_argument("--max-pos-per-trk", type=int, default=40,
                   help="cap positives per track so long tracks don't dominate")
    p.add_argument("--seed", type=int, default=0)
    run(p.parse_args())
