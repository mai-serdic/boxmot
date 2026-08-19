"""
Step 8 — does the geodesic floor prior beat the pixel Gaussian for rebinding?

The question is not "is the map pretty" but "does it change a decision, and in
the right direction". Two label-free ground truths make that measurable on
unannotated footage:

  SAME  — two observations of one continuous tracklet, separated by a gap.
          Definitionally the same person. A prior that vetoes these is doing
          harm: it would refuse a true rebind.
  DIFF  — observations of two tracklets that are alive *in the same frame*.
          Definitionally different people, since one person cannot be two
          boxes at once. A prior that admits these is doing nothing: it leaves
          the impossible candidates for appearance to sort out, which on a
          uniformed workforce it cannot.

Both are then posed as a rebind query: ghost last seen at frame f, candidate
born at frame f + gap. The pixel model and the geodesic model score the same
queries, so the comparison is like-for-like.

    python bench/08_eval_geodesic_prior.py --scene calib/gunsan_test \\
        --traj runs/traj/gunsan_test.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reid.ghost_pool import (GhostTrack, NewTrackInfo, ScoringWeights,  # noqa: E402
                        _spatial_prior_px, score_rebind)
from reid.reachability import Reachability, ReachParams, rebind_prior  # noqa: E402
from reid.scene_depth import SceneModel, floor_from_boxes  # noqa: E402
from reid.scene_geometry import GroundPlane  # noqa: E402


BANDS = ((5, 20), (20, 60), (60, 150), (150, 600))


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC; 0.5 = the score carries no information."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = np.empty(len(allv))
    order = np.argsort(allv, kind="mergesort")
    sv = allv[order]
    i = 0
    while i < len(sv):                      # average ranks over ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def build_tracks(traj: dict, gp, scene) -> dict:
    out = {}
    for tid, rows in traj["traj"].items():
        a = np.asarray(rows, float)
        if len(a) < 10:
            continue
        xy, vis = floor_from_boxes(gp, scene, a[:, 1:5])
        out[int(tid)] = {"f": a[:, 0].astype(int), "box": a[:, 1:5],
                         "xy": xy, "vis": vis}
    return out


def velocity(t: dict, k: int, fps: float, win: int = 5):
    """Smoothed pixel and metric velocity at sample k, from preceding samples."""
    j = max(k - win, 0)
    n = max(k - j, 1)
    bc = lambda i: (0.5 * (t["box"][i, 0] + t["box"][i, 2]),
                    0.5 * (t["box"][i, 1] + t["box"][i, 3]))
    df = max(t["f"][k] - t["f"][j], 1)
    (x0, y0), (x1, y1) = bc(j), bc(k)
    vpx = ((x1 - x0) / df, (y1 - y0) / df)              # px/frame
    dxy = t["xy"][k] - t["xy"][j]
    vxy = tuple(dxy * fps / df) if np.all(np.isfinite(dxy)) else (0.0, 0.0)
    return vpx, vxy


def make_query(t: dict, k: int, fps: float):
    vpx, vxy = velocity(t, k, fps)
    box = t["box"][k]
    g = GhostTrack(
        trk_id=0, last_frame=int(t["f"][k]),
        last_bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
        last_velocity=vpx, embeddings=[np.zeros(4, np.float32)],
        avg_height=float(box[3] - box[1]), avg_width=float(box[2] - box[0]),
        last_xy=tuple(t["xy"][k]) if np.all(np.isfinite(t["xy"][k])) else None,
        last_vel_xy=vxy,
    )
    return g


def make_cand(t: dict, k: int):
    box = t["box"][k]
    return NewTrackInfo(
        bbox=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
        embedding=np.zeros(4, np.float32),
        height=float(box[3] - box[1]), width=float(box[2] - box[0]),
        xy=tuple(t["xy"][k]) if np.all(np.isfinite(t["xy"][k])) else None,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--gap-min", type=int, default=5)
    ap.add_argument("--gap-max", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-pairs", type=int, default=4000)
    ap.add_argument(
        "--labels", type=str, default=None,
        help="labels.json from scripts/label_tracklets.py, {track_id: person}. "
             "Without it SAME pairs can only come from *within* one tracklet, "
             "which is the one case the long-gap rebind path never sees; with "
             "it, SAME becomes a real cross-tracklet re-entry.")
    args = ap.parse_args()

    sdir = Path(args.scene)
    gp = GroundPlane.load(sdir / "scene.json")
    scene = SceneModel.load(sdir / "scene_depth.npz", gp)
    traj = json.load(open(args.traj))
    tracks = build_tracks(traj, gp, scene)
    print(f"[INFO] {len(tracks)} tracklets, {sum(len(t['f']) for t in tracks.values())} obs")

    reach = Reachability.build(scene)
    print(f"[INFO] cold  {reach.summary()}")
    for t in tracks.values():
        ok = t["vis"] & np.all(np.isfinite(t["xy"]), axis=1)
        if ok.any():
            reach.observe(t["xy"][ok])
    print(f"[INFO] warm  {reach.summary()}")

    rng = np.random.default_rng(args.seed)
    w = ScoringWeights()
    par = ReachParams()
    tids = sorted(tracks)

    labels = {}
    if args.labels:
        labels = {int(k): v for k, v in json.load(open(args.labels)).items()}
        labels = {k: v for k, v in labels.items() if k in tracks}
        print(f"[INFO] {len(labels)} labelled tracklets, "
              f"{len(set(labels.values()))} distinct people")

    # ── SAME: one tracklet sampled twice across a gap ───────────────────────
    same = []
    cand_t = [i for i in tids if len(tracks[i]["f"]) > args.gap_min + 2]
    while len(same) < args.n_pairs and cand_t:
        t = tracks[cand_t[rng.integers(len(cand_t))]]
        n = len(t["f"])
        k = int(rng.integers(1, n - 1))
        gap = int(rng.integers(args.gap_min, args.gap_max + 1))
        m = np.searchsorted(t["f"], t["f"][k] + gap)
        if m >= n:
            continue
        same.append((t, k, t, int(m)))

    # ── SAME, cross-tracklet: needs labels, and is the only construction that
    # exercises the case this feature exists for — a person who left the
    # tracker's hands entirely and came back as a brand-new track.
    if labels:
        cross = []
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
                        cross.append((a, k, b, m))
        if cross:
            print(f"[INFO] {len(cross)} cross-tracklet SAME pairs — replacing "
                  f"the within-tracklet proxy")
            same = cross
        else:
            print("[WARN] labels give no re-entry pairs; keeping the proxy")

    # ── DIFF: tracklets known distinct — by label where available, else by
    # co-existing in a frame (one person cannot be two boxes at once) ────────
    coexist = []
    for i in range(len(tids)):
        for j in range(i + 1, len(tids)):
            ii, jj = tids[i], tids[j]
            if labels.get(ii) is not None and labels.get(jj) is not None:
                if labels[ii] != labels[jj]:
                    coexist.append((ii, jj))
                continue
            if np.intersect1d(tracks[ii]["f"], tracks[jj]["f"]).size:
                coexist.append((ii, jj))
    print(f"[INFO] {len(coexist)} tracklet pairs known distinct "
          f"({'labels' if labels else 'co-existence'})")
    diff = []
    while len(diff) < args.n_pairs and coexist:
        i, j = coexist[rng.integers(len(coexist))]
        if rng.random() < 0.5:
            i, j = j, i
        a, b = tracks[i], tracks[j]
        k = int(rng.integers(1, len(a["f"]) - 1))
        gap = int(rng.integers(args.gap_min, args.gap_max + 1))
        m = np.searchsorted(b["f"], a["f"][k] + gap)
        if m >= len(b["f"]) or b["f"][m] <= a["f"][k]:
            continue
        diff.append((a, k, b, int(m)))

    def evaluate(pairs):
        px, geo, feas = [], [], []
        for tg, k, tc, m in pairs:
            g = make_query(tg, k, args.fps)
            c = make_cand(tc, m)
            cur = int(tc["f"][m])
            px.append(_spatial_prior_px(g, c, cur, w))
            if g.last_xy is None or c.xy is None:
                geo.append(np.nan); feas.append(True); continue
            r = rebind_prior(reach, g.last_xy, g.last_vel_xy, c.xy,
                             (cur - g.last_frame) / args.fps, par)
            geo.append(r["prior"]); feas.append(r["feasible"])
        return np.array(px), np.array(geo), np.array(feas, bool)

    ps, gs, fs = evaluate(same)
    pd_, gd, fd = evaluate(diff)
    print(f"\n[INFO] {len(ps)} SAME pairs, {len(pd_)} DIFF pairs "
          f"(gap {args.gap_min}-{args.gap_max} frames)")

    print("\n=== reachability veto: could they have walked there at all? ===")
    print(f"  SAME wrongly vetoed  : {100 * (~fs).mean():5.1f} %   <- cost, want ~0")
    print(f"  DIFF correctly vetoed: {100 * (~fd).mean():5.1f} %   <- benefit")
    for lo, hi in ((5, 20), (20, 60), (60, 150)):
        a = np.array([int(tc['f'][mm]) - int(tg['f'][kk]) for tg, kk, tc, mm in same])
        b = np.array([int(tc['f'][mm]) - int(tg['f'][kk]) for tg, kk, tc, mm in diff])
        sa, sb = (a >= lo) & (a < hi), (b >= lo) & (b < hi)
        if sa.sum() < 20 or sb.sum() < 20:
            continue
        print(f"    {lo:3d}-{hi:3d}f: DIFF vetoed {100 * (~fd[sb]).mean():5.1f} %   "
              f"SAME vetoed {100 * (~fs[sa]).mean():4.1f} %")

    # ── ranking, the decision the pipeline actually makes ───────────────────
    # A pooled same-vs-different AUC pools across gap lengths, and the two
    # priors decay with gap at completely different rates - so pooling scores
    # a short-gap impostor against a long-gap true match, a comparison that
    # never happens in deployment. What does happen is: one ghost, one elapsed
    # time, several candidates on screen, pick one. So rank the true candidate
    # against the distractors alive in that same frame, and report per gap
    # band, weighted by how often each band actually occurs.
    print("\n=== ranking: true candidate vs distractors alive in the same frame ===")
    by_frame: dict[int, list] = {}
    for tid, t in tracks.items():
        for i, f in enumerate(t["f"]):
            by_frame.setdefault(int(f), []).append((tid, i))

    # Empirical gap distribution: for every new-track birth, the time since the
    # most recent death of a *different* track - the query the pool really gets.
    span = {tid: (int(t["f"][0]), int(t["f"][-1])) for tid, t in tracks.items()}
    real_gaps = []
    for tid, (b, _e) in span.items():
        prev = [e for o, (_s, e) in span.items() if o != tid and e < b]
        if prev:
            real_gaps.append(b - max(prev))
    real_gaps = np.array(real_gaps)
    weight = {band: float(np.mean((real_gaps >= band[0]) & (real_gaps < band[1])))
              for band in BANDS}
    print("  observed gap mix: " + "  ".join(
        f"{lo}-{hi}f {100 * weight[(lo, hi)]:.0f}%" for lo, hi in BANDS))

    px_only = ScoringWeights(px_trust_frames=10 ** 9)   # forces the legacy prior
    tally = {k: {b: [0, 0] for b in BANDS} for k in ("pixel", "geodesic")}
    person_of = {id(tracks[t]): labels.get(t) for t in tids} if labels else {}
    for tg, k, tc, m in same:
        cur = int(tc["f"][m])
        # Distractors must be *other people*; without labels that can only be
        # approximated by "a different tracklet".
        others = [(tid, i) for tid, i in by_frame.get(cur, [])
                  if tracks[tid] is not tg and tracks[tid] is not tc
                  and (not labels
                       or labels.get(tid) != person_of.get(id(tg)))]
        if not others:
            continue
        band = next((b for b in BANDS if b[0] <= cur - int(tg["f"][k]) < b[1]), None)
        if band is None:
            continue
        g = make_query(tg, k, args.fps)
        cands = [make_cand(tc, m)] + [make_cand(tracks[t], i) for t, i in others]
        for name, ww, rr in (("pixel", px_only, None), ("geodesic", w, reach)):
            sc_ = np.array([score_rebind(g, c, cur, ww, reach=rr, fps=args.fps,
                                         reach_params=par)[1]["spatial"]
                            for c in cands])
            tally[name][band][0] += 1
            tally[name][band][1] += int((sc_[1:] >= sc_[0]).sum() == 0)

    print(f"\n  {'spatial prior':14s}" + "".join(f"{lo}-{hi}f".rjust(10) for lo, hi in BANDS)
          + "   weighted")
    for name in ("pixel", "geodesic"):
        row, num, den = f"  {name:14s}", 0.0, 0.0
        for b in BANDS:
            n, t1 = tally[name][b]
            row += (f"{100 * t1 / n:9.1f}%" if n else "        - ")
            if n:
                num += weight[b] * 100 * t1 / n
                den += weight[b]
        print(row + f"{num / den:11.1f}%" if den else row)
    print("\n  'pixel' = the prior this replaces; 'geodesic' = what ships "
          "(pixel below\n  the crossover, max(pixel, geodesic) above, "
          "reachability veto throughout).")


if __name__ == "__main__":
    main()
