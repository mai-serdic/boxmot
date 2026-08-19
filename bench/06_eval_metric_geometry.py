"""Step 6 — does metric geometry beat pixel geometry for identity association?

Two measurements, both on `videos/gunsan_test.mp4`, both reusing the weak-label
methodology of steps 01-03 so the numbers are directly comparable with the
CLIP / OSNet / face AUCs already in REPORT.md.

A. Reachability discrimination
   The pixel-space probe asked "does this track birth have *a* plausible
   predecessor?" and answered yes 26/30 times - which sounds good but is the
   wrong question. A loose budget makes everything feasible; what association
   actually needs is a *small* candidate set. So we count candidates per birth
   under a pixel budget vs a physical 1.8 m/s limit on the recovered floor.
   Fewer candidates = less work left for appearance to do.

B. Stature as a ReID cue
   Metric height is clothing-independent and, unlike a face, fully visible
   from behind - the two properties that body appearance and face respectively
   lack on this footage. Scored on the same pair construction as step 02:
   positives are windows of one tracklet >=30 frames apart, negatives are
   windows of tracklets co-active in one frame (physically different people).

    python bench/06_eval_metric_geometry.py --traj runs/traj/gunsan_test.json \
        --calib calib/gunsan_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reid.scene_depth import SceneModel, StatureField  # noqa: E402
from reid.scene_geometry import GroundPlane, collect_observations  # noqa: E402

FPS = 15.0
WALK_MS = 1.8          # generous upper bound on a working person's speed
SLACK_M = 0.35         # foot-point / detector jitter
GAP_MAX_S = 20.0
EDGE = 0.08


def auc_eer(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float, float]:
    """AUC = P(a positive pair scores lower than a negative pair); EER at the
    best balanced threshold. Same convention as bench/03."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv)
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(len(allv))
    r_pos = ranks[: len(pos)].sum()
    auc = 1.0 - (r_pos - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))
    # EER: the threshold where the fragment rate (positives called different)
    # and the merge rate (negatives called same) cross.
    taus = np.unique(allv)
    frag = np.array([(pos > t).mean() for t in taus])
    merge = np.array([(neg <= t).mean() for t in taus])
    i = int(np.argmin(np.abs(frag - merge)))
    return float(auc), float((frag[i] + merge[i]) / 2), float(taus[i])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True)
    ap.add_argument("--calib", required=True)
    ap.add_argument("--scene", default=None, help="scene_depth.npz from commission_scene.py")
    ap.add_argument("--out", default="bench/data/metric_geometry.json")
    args = ap.parse_args()

    gp = GroundPlane.load(args.calib)
    model = gp.radial_model()
    d = json.loads(Path(args.traj).read_text())
    W, H = d["W"], d["H"]
    traj = {int(k): sorted(v) for k, v in d["traj"].items()}
    tracks = {k: v for k, v in traj.items() if len(v) >= 5}
    scene = SceneModel.load(args.scene, gp) if args.scene else None
    print(f"[data] {len(tracks)} tracklets >=5 frames"
          f"{'' if scene else '  (no scene depth: occlusion tests skipped)'}\n")

    # ---------------------------------------------------------------- A ----
    def foot_px(b):
        return np.array([(b[1] + b[3]) / 2.0, b[4]])

    def to_floor(px):
        u = model.undistort_points(px.reshape(1, 2))
        return gp.floor_xy(np.hstack([u, [[1.0]]]))[0]

    births, deaths = [], []
    for k, v in sorted(tracks.items()):
        for arr, row in ((births, v[0]), (deaths, v[-1])):
            p = foot_px(row)
            arr.append({
                "tid": k, "frame": row[0], "px": p, "xy": to_floor(p),
                "h": row[4] - row[2],
            })

    def on_edge(p):
        return (p[0] < EDGE * W or p[0] > (1 - EDGE) * W
                or p[1] < EDGE * H or p[1] > (1 - EDGE) * H)

    mid = [b for b in births if not on_edge(b["px"])]
    print(f"[A] mid-scene births: {len(mid)}/{len(births)}")

    n_px, n_m = [], []
    for b in mid:
        cp = cm = 0
        for dd in deaths:
            if dd["tid"] == b["tid"] or dd["frame"] >= b["frame"]:
                continue
            dt = (b["frame"] - dd["frame"]) / FPS
            if dt > GAP_MAX_S:
                continue
            # pixel-space budget (what the old probe used)
            reach_px = WALK_MS * max(dt, 0.2) * (b["h"] / 1.7) + 0.5 * b["h"]
            if np.linalg.norm(b["px"] - dd["px"]) <= reach_px:
                cp += 1
            # metric budget: an actual speed limit on the floor
            if np.isfinite(b["xy"]).all() and np.isfinite(dd["xy"]).all():
                if np.linalg.norm(b["xy"] - dd["xy"]) <= WALK_MS * max(dt, 0.2) + SLACK_M:
                    cm += 1
        n_px.append(cp)
        n_m.append(cm)
    n_px, n_m = np.array(n_px), np.array(n_m)

    def summarize(n, label):
        print(f"    {label:<8} any-candidate {100*(n>0).mean():5.1f}%   "
              f"UNAMBIGUOUS (exactly 1) {100*(n==1).mean():5.1f}%   "
              f"mean candidates {n.mean():.2f}")
        return {"any": float((n > 0).mean()), "unique": float((n == 1).mean()),
                "mean_candidates": float(n.mean())}

    res_px = summarize(n_px, "pixel")
    res_m = summarize(n_m, "metric")
    print()

    # ---------------------------------------------------------------- B ----
    obs = collect_observations(traj, model, stride=3)
    st_raw = gp.stature_m(obs.foot, obs.head)
    s_img = max(W, H) / 2.0
    fpx = obs.foot[:, :2] / obs.foot[:, 2:] * s_img + np.array([W / 2, H / 2])
    span = {k: (v[0][0], v[-1][0]) for k, v in tracks.items()}

    def score(st: np.ndarray, label: str) -> dict:
        """Same pair construction as step 02, so the AUCs are comparable."""
        ok = np.isfinite(st) & (st > 0.9) & (st < 2.4)
        S, tid, fr = st[ok], obs.track_id[ok], obs.frame[ok]
        WIN = 10
        wins = []
        for t in np.unique(tid):
            m = tid == t
            s_t, f_t = S[m], fr[m]
            o = np.argsort(f_t)
            s_t, f_t = s_t[o], f_t[o]
            for i in range(0, len(s_t) - WIN + 1, WIN):
                wins.append({"tid": int(t), "frame": float(f_t[i:i + WIN].mean()),
                             "stature": float(np.median(s_t[i:i + WIN]))})
        pos, neg = [], []
        for i in range(len(wins)):
            for j in range(i + 1, len(wins)):
                a, b = wins[i], wins[j]
                if a["tid"] == b["tid"]:
                    if abs(a["frame"] - b["frame"]) >= 30:
                        pos.append(abs(a["stature"] - b["stature"]))
                elif a["tid"] in span and b["tid"] in span:
                    sa, sb = span[a["tid"]], span[b["tid"]]
                    if min(sa[1], sb[1]) >= max(sa[0], sb[0]):   # co-active => different
                        neg.append(abs(a["stature"] - b["stature"]))
        pos, neg = np.array(pos), np.array(neg)
        if len(pos) < 10 or len(neg) < 10:
            print(f"    {label:<40} too few pairs ({len(pos)}/{len(neg)})")
            return {}
        auc, eer, tau = auc_eer(pos, neg)
        print(f"    {label:<40} AUC {auc:.3f}  EER {eer:.3f}   "
              f"same {np.median(pos)*100:4.1f}cm  diff {np.median(neg)*100:4.1f}cm  "
              f"({len(pos)}+/{len(neg)}-)")
        return {"auc": auc, "eer": eer, "tau_m": tau, "n_pos": len(pos),
                "n_neg": len(neg), "pos_median_m": float(np.median(pos)),
                "neg_median_m": float(np.median(neg))}

    print("[B] stature as a ReID cue")
    variants = {"raw": score(st_raw, "1. raw foot-box stature")}

    if scene is not None:
        vis = ~scene.foot_occluded(fpx)
        st_gate = np.where(vis, st_raw, np.nan)
        variants["occlusion_gated"] = score(st_gate, "2. + occlusion gating (feet visible)")

        # The bias field is fitted on tracklets, so evaluating it on the same
        # tracklets would leak. Fit on one fold, score the other.
        uid = np.unique(obs.track_id)
        fold = np.isin(obs.track_id, uid[::2])
        st_field = np.full(len(st_raw), np.nan)
        for sel in (fold, ~fold):
            try:
                f = StatureField.fit(fpx[~sel], st_gate[~sel], obs.track_id[~sel], gp)
            except RuntimeError:
                continue
            st_field[sel] = f.apply(fpx[sel], st_gate[sel], gp)
        variants["bias_corrected"] = score(st_field, "3. + position-bias field (held-out)")
        st = st_field
    else:
        st = st_raw
        vis = np.ones(len(st_raw), bool)

    print("    (compare: CLIP body AUC 0.842 / EER 0.227, face AUC 0.907 / EER 0.127)")
    auc = variants.get("bias_corrected", variants["raw"]).get("auc", float("nan"))
    fin = np.isfinite(st)
    pos = np.array([]); neg = np.array([])
    per_track = {int(t): float(np.median(st[fin & (obs.track_id == t)]))
                 for t in np.unique(obs.track_id[fin])
                 if (fin & (obs.track_id == t)).sum() >= 10}
    print("\n    per-tracklet median stature after correction (m):")
    for t, v in sorted(per_track.items(), key=lambda kv: kv[1]):
        print(f"      trk {t:>3}: {v:.2f}   n={int((fin & (obs.track_id==t)).sum())}")

    # ------------------------------------------------------------- plots ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    mx = max(n_px.max(), n_m.max())
    bins = np.arange(-0.5, mx + 1.5)
    ax[0].hist([n_px, n_m], bins=bins, label=["pixel budget", "metric 1.8 m/s"])
    ax[0].set_xlabel("feasible predecessors per mid-scene birth")
    ax[0].set_ylabel("count"); ax[0].legend()
    ax[0].set_title("Candidate-set size (smaller = better)")
    names = [k for k in variants if variants[k]]
    ax[1].bar(range(len(names)), [variants[k]["auc"] for k in names], color="steelblue")
    ax[1].axhline(0.842, color="darkorange", ls="--", label="CLIP body")
    ax[1].axhline(0.907, color="crimson", ls="--", label="face")
    ax[1].axhline(0.5, color="k", ls=":", label="chance")
    ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels(["raw", "+occl.\ngating", "+bias\nfield"][:len(names)])
    ax[1].set_ylim(0.4, 1.0); ax[1].set_ylabel("AUC"); ax[1].legend(fontsize=8)
    ax[1].set_title("Stature separability")
    plt.tight_layout()
    Path("bench/data").mkdir(parents=True, exist_ok=True)
    plt.savefig("bench/data/metric_geometry.png", dpi=120)
    plt.close()

    Path(args.out).write_text(json.dumps({
        "reachability": {"n_mid_scene_births": len(mid), "pixel": res_px, "metric": res_m},
        "stature": {**variants, "feet_visible_frac": float(vis.mean()),
                    "per_track_median_m": per_track},
    }, indent=2))
    print(f"\n[save] {args.out}  +  bench/data/metric_geometry.png")


if __name__ == "__main__":
    main()
