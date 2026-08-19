"""
Step 4 (quick win): sweep PersonDB match_threshold (tau) on CLIP distances.

Picks an operating point on the fragment/merge trade-off. For a PERSISTENT
gallery, a wrong merge (two people share an identity, cascades) costs more than
a fragment (a throwaway new ID, recoverable) — so we also report the tau that
holds merge under a small budget, not just the EER-balanced point.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
from boxmot.reid.core import ReID  # noqa: E402


def run(args):
    d = Path(args.datadir)
    manifest = json.loads((d / "manifest.json").read_text())
    pairs = json.loads((d / "pairs.json").read_text())
    device = torch.device(args.device)

    cid2path = {m["cid"]: d / "crops" / f"crop_{m['cid']:06d}.jpg" for m in manifest}
    used = sorted({c for pr in (pairs["pos"] + pairs["neg"]) for c in pr})
    row = {c: i for i, c in enumerate(used)}

    reid = ReID(weights=PROJECT_ROOT / "models/clip_market1501.pt", device=device, half=False)
    embs = None
    paths = [cid2path[c] for c in used]
    for i in range(0, len(paths), 256):
        chunk = paths[i:i + 256]
        f = np.asarray(reid([cv2.imread(str(p)) for p in chunk]), dtype=np.float32)
        if embs is None:
            embs = np.zeros((len(paths), f.shape[1]), dtype=np.float32)
        embs[i:i + len(chunk)] = f
    embs /= np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12

    def dists(prs):
        a = embs[[row[p[0]] for p in prs]]
        b = embs[[row[p[1]] for p in prs]]
        return 1.0 - np.sum(a * b, axis=1)

    pos, neg = dists(pairs["pos"]), dists(pairs["neg"])

    print(f"CLIP  n_pos={len(pos)} n_neg={len(neg)}  pos_med={np.median(pos):.3f} neg_med={np.median(neg):.3f}\n")
    print(f"{'tau':>5} {'fragment':>9} {'merge':>7}   note")
    taus = np.round(np.arange(0.22, 0.56, 0.02), 2)
    best_eer = None
    for t in taus:
        frag = float(np.mean(pos > t))
        merge = float(np.mean(neg < t))
        note = ""
        if best_eer is None or abs(frag - merge) < best_eer[0]:
            best_eer = (abs(frag - merge), t, frag, merge)
        marks = []
        if abs(t - 0.25) < 1e-9:
            marks.append("← current")
        note = " ".join(marks)
        print(f"{t:>5.2f} {frag:>8.1%} {merge:>7.1%}   {note}")

    print()
    _, et, ef, em = best_eer
    print(f"EER-balanced       : tau={et:.2f}  fragment={ef:.1%}  merge={em:.1%}")
    for budget in (0.02, 0.05, 0.10):
        ok = [t for t in taus if float(np.mean(neg < t)) <= budget]
        if ok:
            t = max(ok)
            print(f"merge<= {budget:>4.0%} (safe): tau={t:.2f}  fragment={np.mean(pos>t):.1%}  merge={np.mean(neg<t):.1%}")

    # curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fine = np.linspace(0.15, 0.65, 200)
        fr = [np.mean(pos > t) for t in fine]
        mg = [np.mean(neg < t) for t in fine]
        plt.figure(figsize=(7, 4.5))
        plt.plot(fine, fr, label="fragment rate (same→new ID)", color="#2a9d8f")
        plt.plot(fine, mg, label="merge rate (diff→same ID)", color="#e76f51")
        plt.axvline(0.25, color="k", ls=":", lw=1, label="current tau=0.25")
        plt.axvline(et, color="gray", ls="--", lw=1, label=f"EER tau={et:.2f}")
        plt.xlabel("match_threshold (tau)")
        plt.ylabel("rate")
        plt.title("CLIP: fragment vs merge trade-off (gunsan_test)")
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(d / "threshold_sweep.png", dpi=120)
        print(f"\nwrote {d/'threshold_sweep.png'}")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", required=True)
    p.add_argument("--device", default="cuda:0")
    run(p.parse_args())
