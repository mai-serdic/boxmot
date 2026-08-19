"""
Step 3: score each ReID embedding on the labelled verification set.

For every model we embed each crop once, then for every labelled pair compute
cosine distance d = 1 - cos. Two views of the result:

  1. Threshold-FREE (the fair, model-agnostic comparison):
       AUC   — P(random positive is closer than random negative). 1.0 = perfect.
       EER   — error rate at the threshold where miss == false-match.
     These don't depend on any hand-picked cutoff, so they compare OSNet and
     CLIP fairly even though their distance scales differ.

  2. At the pipeline's OPERATING threshold tau (PersonDB match_threshold):
       fragment_rate = P(pos_dist > tau)  → same person splits into a new ID
       merge_rate    = P(neg_dist < tau)  → two workers collapse to one ID
     Reported at the shared tau AND at each model's own EER-optimal tau.

Outputs: metrics.json + dist_hist.png in the data dir.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
from boxmot.reid.core import ReID  # noqa: E402

MODELS = {
    "osnet_x1_0":  "models/osnet_x1_0_msmt17.pt",
    "osnet_x0_25": "models/osnet_x0_25_msmt17.pt",
    "clip_market": "models/clip_market1501.pt",
}


def embed_all(weight_path, crop_paths, device, batch=256):
    # Use the ReID wrapper (callable): it applies each model's own input_shape
    # and mean/std normalization to a list of BGR crops — the fair way to
    # compare models with different preprocessing.
    reid = ReID(weights=weight_path, device=device, half=False)
    out = None
    for i in range(0, len(crop_paths), batch):
        chunk = crop_paths[i:i + batch]
        crops = [cv2.imread(str(p)) for p in chunk]
        feats = np.asarray(reid(crops), dtype=np.float32)  # ReID.__call__ → normalized
        if out is None:
            out = np.zeros((len(crop_paths), feats.shape[1]), dtype=np.float32)
        out[i:i + len(chunk)] = feats
    # re-normalize defensively
    out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
    return out


def pair_dists(embs, pairs, cid2row):
    a = embs[[cid2row[p[0]] for p in pairs]]
    b = embs[[cid2row[p[1]] for p in pairs]]
    cos = np.sum(a * b, axis=1)
    return 1.0 - cos  # cosine distance in [0, 2]


def auc_eer(pos, neg):
    """AUC via Mann-Whitney rank statistic; EER + optimal threshold by sweep.

    A pair is "same" when its distance is small, so we score by -distance
    (higher score = more same) and rank; positives should end up high-ranked.
    """
    labels = np.concatenate([np.ones_like(pos), np.zeros_like(neg)])  # 1 = same
    scores = -np.concatenate([pos, neg])                              # higher = more same
    r = scores.argsort().argsort() + 1                               # 1..N ascending ranks
    n_pos, n_neg = len(pos), len(neg)
    auc = (r[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

    thr = np.unique(np.concatenate([pos, neg]))
    best = None
    for t in thr:
        miss = np.mean(pos > t)          # positive judged different (fragment)
        fa = np.mean(neg <= t)           # negative judged same (merge)
        if best is None or abs(miss - fa) < best[0]:
            best = (abs(miss - fa), t, (miss + fa) / 2)
    _, eer_t, eer = best
    return float(auc), float(eer), float(eer_t)


def run(args):
    d = Path(args.datadir)
    manifest = json.loads((d / "manifest.json").read_text())
    pairs = json.loads((d / "pairs.json").read_text())
    device = torch.device(args.device)

    cid2path = {m["cid"]: d / "crops" / f"crop_{m['cid']:06d}.jpg" for m in manifest}
    used_cids = sorted({c for pr in (pairs["pos"] + pairs["neg"]) for c in pr})
    cid2row = {c: i for i, c in enumerate(used_cids)}
    crop_paths = [cid2path[c] for c in used_cids]
    print(f"[EVAL] crops used={len(used_cids)}  pos={len(pairs['pos'])}  neg={len(pairs['neg'])}")

    tau = args.tau
    results = {}
    dists_store = {}
    for name, rel in MODELS.items():
        wp = PROJECT_ROOT / rel
        if not wp.exists():
            print(f"[EVAL] {name}: weight missing ({wp}) — skipped")
            continue
        t0 = time.time()
        embs = embed_all(wp, crop_paths, device)
        pos = pair_dists(embs, pairs["pos"], cid2row)
        neg = pair_dists(embs, pairs["neg"], cid2row)
        auc, eer, eer_t = auc_eer(pos, neg)
        frag_tau = float(np.mean(pos > tau))
        merge_tau = float(np.mean(neg < tau))
        frag_eer = float(np.mean(pos > eer_t))
        merge_eer = float(np.mean(neg < eer_t))
        results[name] = {
            "dim": int(embs.shape[1]),
            "embed_s": round(time.time() - t0, 1),
            "pos_median": round(float(np.median(pos)), 4),
            "neg_median": round(float(np.median(neg)), 4),
            "separation": round(float(np.median(neg) - np.median(pos)), 4),
            "auc": round(auc, 4),
            "eer": round(eer, 4),
            "eer_threshold": round(eer_t, 4),
            f"fragment_rate@tau{tau}": round(frag_tau, 4),
            f"merge_rate@tau{tau}": round(merge_tau, 4),
            "fragment_rate@eer_thr": round(frag_eer, 4),
            "merge_rate@eer_thr": round(merge_eer, 4),
        }
        dists_store[name] = (pos, neg)
        print(f"[EVAL] {name:12s} dim={embs.shape[1]:4d} AUC={auc:.4f} EER={eer:.4f} "
              f"(thr={eer_t:.3f})  sep={results[name]['separation']:+.3f}  "
              f"frag@{tau}={frag_tau:.3f} merge@{tau}={merge_tau:.3f}")

    (d / "metrics.json").write_text(json.dumps({"tau": tau, "models": results}, indent=2))

    # ── plot ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(dists_store)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
        bins = np.linspace(0, 1.2, 60)
        for ax, (name, (pos, neg)) in zip(axes[0], dists_store.items()):
            ax.hist(pos, bins=bins, alpha=0.6, label="same (positive)", color="#2a9d8f", density=True)
            ax.hist(neg, bins=bins, alpha=0.6, label="diff (negative)", color="#e76f51", density=True)
            ax.axvline(tau, color="k", ls="--", lw=1, label=f"tau={tau}")
            r = results[name]
            ax.set_title(f"{name}  AUC={r['auc']:.3f}  EER={r['eer']:.3f}")
            ax.set_xlabel("cosine distance")
            ax.legend(fontsize=8)
        fig.suptitle("Same-person vs different-person distance (gunsan_test, weak tracklet labels)")
        fig.tight_layout()
        fig.savefig(d / "dist_hist.png", dpi=120)
        print(f"[EVAL] wrote {d/'dist_hist.png'}")
    except Exception as e:
        print(f"[EVAL] plot skipped: {e}")

    print(f"[EVAL] wrote {d/'metrics.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--tau", type=float, default=0.25,
                   help="operating cosine-distance threshold (PersonDB match_threshold)")
    run(p.parse_args())
