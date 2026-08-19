"""
Step 5: does pose-gated face anchoring break the appearance ceiling?

Step 3/4 proved the body-appearance ceiling: CLIP floors at ~0.22 EER, so no
tau makes both fragmentation and merges small. This step measures the proposed
fix on the same 715+715 weak-labelled pairs:

  * COVERAGE  — how often do both crops of a pair carry a pose-gated face?
    (face is an opportunistic anchor; it cannot help pairs it never sees)
  * PRECISION — on covered pairs, how sharply does ArcFace separate same vs
    different people compared to CLIP?
  * FUSION    — the pipeline's actual decision rule: face verdict overrides
    body when both sides have a gated face, body-CLIP at tau=0.35 otherwise.
    Reported as overall fragment/merge vs the body-only baseline.

Inputs: bench/data produced by steps 1-2, plus faces.npz (raw SCRFD+ArcFace
observations per crop, gate-free so gates can be swept here).

Usage:
    python bench/05_eval_face_anchor.py --datadir bench/data
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
from reid.face_anchor import pose_gate  # noqa: E402


def auc_eer(pos: np.ndarray, neg: np.ndarray) -> tuple[float, float, float]:
    """Threshold-free AUC + EER (and the tau achieving it) on distance arrays."""
    auc = float(np.mean(pos[:, None] < neg[None, :]))
    taus = np.linspace(0.0, 1.5, 601)
    frag = np.array([np.mean(pos > t) for t in taus])
    merge = np.array([np.mean(neg < t) for t in taus])
    i = int(np.argmin(np.abs(frag - merge)))
    return auc, float((frag[i] + merge[i]) / 2), float(taus[i])


def run(args):
    d = Path(args.datadir)
    manifest = json.loads((d / "manifest.json").read_text())
    pairs = json.loads((d / "pairs.json").read_text())
    faces = np.load(d / "faces.npz")
    device = torch.device(args.device)

    # ── gated face embedding per crop ───────────────────────────────────────
    face_emb: dict[int, np.ndarray] = {}
    n_raw = len(faces["cids"])
    for i in range(n_raw):
        ok, _size, _front = pose_gate(
            faces["bboxes"][i], faces["kps"][i], float(faces["det_scores"][i]),
            det_score_min=args.det_score_min, min_face_px=args.min_face_px,
        )
        if ok:
            face_emb[int(faces["cids"][i])] = faces["embs"][i]
    n_crops = len(manifest)
    print(f"[FACE] raw faces={n_raw}/{n_crops} crops; gated={len(face_emb)} "
          f"({len(face_emb) / n_crops:.1%} of all crops)")

    # ── body (CLIP) embedding per used crop ─────────────────────────────────
    cid2path = {m["cid"]: d / "crops" / f"crop_{m['cid']:06d}.jpg" for m in manifest}
    used = sorted({c for pr in (pairs["pos"] + pairs["neg"]) for c in pr})
    row = {c: i for i, c in enumerate(used)}
    reid = ReID(weights=PROJECT_ROOT / "models/clip_market1501.pt", device=device, half=False)
    body = None
    paths = [cid2path[c] for c in used]
    for i in range(0, len(paths), 256):
        chunk = paths[i:i + 256]
        f = np.asarray(reid([cv2.imread(str(p)) for p in chunk]), dtype=np.float32)
        if body is None:
            body = np.zeros((len(paths), f.shape[1]), dtype=np.float32)
        body[i:i + len(chunk)] = f
    body /= np.linalg.norm(body, axis=1, keepdims=True) + 1e-12

    def body_dist(a: int, b: int) -> float:
        return 1.0 - float(np.dot(body[row[a]], body[row[b]]))

    # ── per-pair distances ──────────────────────────────────────────────────
    def split(prs):
        both, rest = [], []
        for a, b in prs:
            (both if (a in face_emb and b in face_emb) else rest).append((a, b))
        return both, rest

    pos_f, pos_nf = split(pairs["pos"])
    neg_f, neg_nf = split(pairs["neg"])
    n_pos, n_neg = len(pairs["pos"]), len(pairs["neg"])
    print(f"[FACE] pair coverage: pos {len(pos_f)}/{n_pos} ({len(pos_f)/n_pos:.1%})  "
          f"neg {len(neg_f)}/{n_neg} ({len(neg_f)/n_neg:.1%})")

    fpos = np.array([1.0 - float(np.dot(face_emb[a], face_emb[b])) for a, b in pos_f])
    fneg = np.array([1.0 - float(np.dot(face_emb[a], face_emb[b])) for a, b in neg_f])
    bpos_f = np.array([body_dist(a, b) for a, b in pos_f])
    bneg_f = np.array([body_dist(a, b) for a, b in neg_f])

    if len(fpos) and len(fneg):
        fa, fe, ft = auc_eer(fpos, fneg)
        ba, be, _ = auc_eer(bpos_f, bneg_f)
        print(f"\n[COVERED PAIRS] face  : AUC={fa:.4f} EER={fe:.4f} (tau={ft:.2f})  "
              f"pos_med={np.median(fpos):.3f} neg_med={np.median(fneg):.3f}")
        print(f"[COVERED PAIRS] body  : AUC={ba:.4f} EER={be:.4f}  "
              f"pos_med={np.median(bpos_f):.3f} neg_med={np.median(bneg_f):.3f}")
    else:
        ft = args.face_tau
        print("[COVERED PAIRS] not enough covered pairs to score")

    # ── fusion: face overrides body where available ─────────────────────────
    face_tau = args.face_tau if args.face_tau is not None else ft
    body_tau = args.body_tau

    def override_errors():
        """Face verdict replaces body wherever both sides have a gated face."""
        frag = sum(fd > face_tau for fd in fpos)
        merge = sum(fd < face_tau for fd in fneg)
        frag += sum(body_dist(a, b) > body_tau for a, b in pos_nf)
        merge += sum(body_dist(a, b) < body_tau for a, b in neg_nf)
        return frag / n_pos, merge / n_neg

    def confirm_only_errors():
        """The pipeline's rule: a strict face match can only ADD a same-person
        verdict (rescue a fragment / re-anchor); a face mismatch never rejects
        a body match. pair is 'same' iff body says same OR face says same."""
        frag = merge = 0
        for (a, b), fd in zip(pos_f, fpos):
            frag += not (body_dist(a, b) <= body_tau or fd <= face_tau)
        frag += sum(body_dist(a, b) > body_tau for a, b in pos_nf)
        for (a, b), fd in zip(neg_f, fneg):
            merge += (body_dist(a, b) <= body_tau or fd <= face_tau)
        merge += sum(body_dist(a, b) < body_tau for a, b in neg_nf)
        return frag / n_pos, merge / n_neg

    def body_only_errors():
        frag = sum(body_dist(a, b) > body_tau for a, b in pairs["pos"])
        merge = sum(body_dist(a, b) < body_tau for a, b in pairs["neg"])
        return frag / n_pos, merge / n_neg

    bf, bm = body_only_errors()
    of, om = override_errors()
    cf, cm = confirm_only_errors()
    print(f"\n[DECISION @ body_tau={body_tau:.2f}, face_tau={face_tau:.2f}]")
    print(f"  body-only     : fragment={bf:.1%}  merge={bm:.1%}")
    print(f"  face-override : fragment={of:.1%}  merge={om:.1%}")
    print(f"  confirm-only  : fragment={cf:.1%}  merge={cm:.1%}   ← pipeline rule")

    out = {
        "gated_crops": len(face_emb), "n_crops": n_crops,
        "pos_covered": len(pos_f), "neg_covered": len(neg_f),
        "face_tau": float(face_tau), "body_tau": float(body_tau),
        "body_only": {"fragment": bf, "merge": bm},
        "face_override": {"fragment": of, "merge": om},
        "confirm_only": {"fragment": cf, "merge": cm},
    }
    if len(fpos) and len(fneg):
        out["covered"] = {
            "face": {"auc": fa, "eer": fe, "eer_tau": ft},
            "body": {"auc": ba, "eer": be},
        }
    (d / "face_metrics.json").write_text(json.dumps(out, indent=2))
    print(f"\n[DONE] wrote {d / 'face_metrics.json'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--datadir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--det-score-min", type=float, default=0.5)
    p.add_argument("--min-face-px", type=float, default=24.0)
    p.add_argument("--face-tau", type=float, default=None,
                   help="face cosine-distance threshold; default = face EER point")
    p.add_argument("--body-tau", type=float, default=0.35,
                   help="body threshold for uncovered pairs (pipeline default)")
    run(p.parse_args())
