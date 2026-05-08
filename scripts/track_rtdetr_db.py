"""
RT-DETRv4 (ONNX) + BoxMOT BoT-SORT + Persistent Person Database
----------------------------------------------------------------
Two-tier pipeline:
  1. Short-term: BoT-SORT does frame-to-frame association with its own trk_id.
  2. Long-term:  IdentityResolver maps trk_id → persistent Person_XXX via a
     gallery-backed PersonDB that survives across runs/sessions.

Embeddings are extracted once per frame (OSNet x1.0 by default) and fed into
BoT-SORT via the `embs=` parameter, so we don't double-compute them.

Usage:
    python track_rtdetr_db.py \\
        --onnx  models/20260504_rtv4_hgnetv2_m.onnx \\
        --reid  osnet_x1_0_msmt17.pt \\
        --input videos/gunsan_test.mp4 \\
        --output runs/track/rtdetr_db.mp4 \\
        --gallery gallery/persons.npz
"""

import argparse
import os
import sys
from pathlib import Path

# Allow `from person_db import …` when running this script from boxmot/scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # boxmot/
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import onnxruntime as ort

from boxmot.trackers.botsort.botsort import BotSort
from boxmot.reid.core import ReID

from person_db import PersonDB, IdentityResolver


# ─────────────────────────────────────────────────────────────────────────────
# ONNX detector wrapper (same as track_rtdetr.py, kept self-contained)
# ─────────────────────────────────────────────────────────────────────────────
class RTDetrOnnx:
    INPUT_SIZE = (640, 640)

    def __init__(self, onnx_path: str, device: str = "cuda:0"):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "cuda" in device
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        active = self.session.get_providers()
        print(f"[INFO] ORT providers active: {active}")
        if "cuda" in device and "CUDAExecutionProvider" not in active:
            print("[WARN] Requested CUDA but ORT fell back to CPU.")
        self.input_name = self.session.get_inputs()[0].name
        self.size_name = self.session.get_inputs()[1].name

    def predict(self, frame_bgr: np.ndarray, conf_thresh: float = 0.4, person_class: int = 0):
        orig_h, orig_w = frame_bgr.shape[:2]
        h, w = self.INPUT_SIZE

        img = cv2.resize(frame_bgr, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis]
        orig_size = np.array([[orig_w, orig_h]], dtype=np.int64)

        labels, boxes, scores = self.session.run(
            None, {self.input_name: img, self.size_name: orig_size}
        )
        labels = labels[0]
        boxes = boxes[0]
        scores = scores[0]

        mask = (labels == person_class) & (scores >= conf_thresh)
        if mask.sum() == 0:
            return np.empty((0, 6), dtype=np.float32)
        return np.column_stack(
            [boxes[mask], scores[mask], labels[mask].astype(float)]
        ).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Color palette (deterministic per person_id)
# ─────────────────────────────────────────────────────────────────────────────
def color_for(pid: int):
    rng = np.random.default_rng(pid * 9_973 + 17)
    c = rng.integers(64, 255, size=3, dtype=np.int64)
    return int(c[0]), int(c[1]), int(c[2])


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def run(args):
    device_str = args.device
    torch_device = torch.device(device_str)

    # ── 1. Detector ──────────────────────────────────────────────────────────
    print(f"[INFO] Loading ONNX detector: {args.onnx}")
    detector = RTDetrOnnx(args.onnx, device=device_str)

    # ── 2. ReID ──────────────────────────────────────────────────────────────
    reid_path = Path(args.reid)
    if not reid_path.is_absolute():
        reid_path = PROJECT_ROOT / args.reid
    print(f"[INFO] Loading ReID model: {reid_path}")
    reid_backend = ReID(weights=reid_path, device=torch_device, half=False).get_backend()

    # ── 3. Video I/O ─────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video {orig_w}x{orig_h} @ {fps:.1f} fps  ({total} frames)")

    # ── 4. Tracker (short-term) ──────────────────────────────────────────────
    tracker = BotSort(
        reid_model=reid_backend,
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.7,
        track_buffer=600,
        match_thresh=0.8,
        proximity_thresh=0.9,
        appearance_thresh=0.4,
        cmc_method="ecc",
        frame_rate=int(round(fps)),
        with_reid=True,
    )

    # ── 5. Person DB + resolver (long-term) ─────────────────────────────────
    gallery_path = Path(args.gallery)
    if not gallery_path.is_absolute():
        gallery_path = PROJECT_ROOT / args.gallery
    db = PersonDB(
        gallery_path,
        k_per_person=args.k_per_person,
        match_threshold=args.match_threshold,
    )
    resolver = IdentityResolver(
        db,
        decision_frames=args.decision_frames,
        quality_min_conf=args.quality_min_conf,
        ratio_threshold=args.ratio_threshold,
        gallery_update_max_dist=args.gallery_update_max_dist,
    )
    print(
        f"[INFO] DB loaded with {db.n_persons} persons "
        f"({db.n_embeddings} embeddings); next_id={db.next_id}"
    )

    # ── 6. Output writer ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (orig_w, orig_h)
    )

    # ── 7. Main loop ────────────────────────────────────────────────────────
    print("[INFO] Tracking with persistent person DB …")
    frame_idx = 0
    frame_shape = (orig_h, orig_w, 3)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            dets = detector.predict(
                frame, conf_thresh=args.conf, person_class=args.person_class
            )

            # Pre-extract embeddings ONCE (BoT-SORT will use these via embs=)
            if len(dets) > 0:
                embs = reid_backend.get_features(dets[:, :4], frame)
                embs = np.asarray(embs, dtype=np.float32)
            else:
                embs = np.empty((0, 0), dtype=np.float32)

            tracks = tracker.update(dets, frame, embs=embs if len(dets) else None)
            # tracks: (M, 8) → x1 y1 x2 y2 trk_id conf cls det_ind

            # Mutual-exclusion set: which Person_XXX are already attached to
            # *other* still-active tracks on this frame? A pending tracklet
            # cannot resolve to one of these — that would put two boxes with
            # the same label on screen at once.
            active_trk_ids = {int(t[4]) for t in tracks}
            in_use_pids: set[int] = {
                pid for tid, pid in resolver.trk_to_person.items()
                if tid in active_trk_ids
            }

            for t in tracks:
                x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
                trk_id = int(t[4])
                conf = float(t[5])
                det_ind = int(t[7])

                if det_ind < 0 or det_ind >= len(embs):
                    pid = resolver.trk_to_person.get(trk_id)
                else:
                    # Exclude pids in use by *other* trks; this trk's own pid
                    # (if already resolved) must remain in scope so the
                    # resolved-track gallery-update path still works.
                    own_pid = resolver.trk_to_person.get(trk_id)
                    excl = in_use_pids - ({own_pid} if own_pid is not None else set())
                    pid = resolver.resolve(
                        trk_id,
                        embs[det_ind],
                        (x1, y1, x2, y2),
                        conf,
                        frame_idx,
                        frame_shape,
                        exclude_pids=excl,
                    )

                # If this call just committed a new mapping, lock that pid
                # for the rest of this frame's iteration.
                if pid is not None:
                    in_use_pids.add(pid)

                if pid is not None:
                    label = f"Person_{pid:03d}"
                    color = color_for(pid)
                else:
                    label = f"trk{trk_id}?"
                    color = (180, 180, 180)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(
                    frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1
                )
                cv2.putText(
                    frame, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA,
                )

            # HUD: gallery summary
            hud = (
                f"frame {frame_idx}/{total}  "
                f"persons={db.n_persons}  "
                f"enrolled={resolver.n_enrolled}  matched={resolver.n_matched}  "
                f"refused_amb={resolver.n_refused_ambiguous}"
            )
            cv2.rectangle(frame, (0, 0), (orig_w, 28), (0, 0, 0), -1)
            cv2.putText(
                frame, hud, (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
            )

            writer.write(frame)
            frame_idx += 1
            if frame_idx % 100 == 0:
                pct = frame_idx / total * 100 if total else 0
                print(
                    f"  frame {frame_idx}/{total} ({pct:.1f}%)  "
                    f"persons={db.n_persons} "
                    f"(enrolled={resolver.n_enrolled} matched={resolver.n_matched})"
                )
    finally:
        cap.release()
        writer.release()
        db.save()

    print(f"\n[DONE] Saved video: {args.output}")
    print(
        f"[DONE] DB final state: {db.n_persons} persons, "
        f"{db.n_embeddings} embeddings  ({db.db_path})"
    )

    fixed = args.output.replace(".mp4", "_h264.mp4")
    os.system(
        f'ffmpeg -y -i "{args.output}" -vcodec libx264 -crf 23 "{fixed}" '
        f'-loglevel quiet'
    )
    if os.path.exists(fixed):
        print(f"[DONE] H.264 copy: {fixed}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="RT-DETRv4 + BoT-SORT + persistent person DB"
    )
    p.add_argument("--onnx", default="models/20260504_rtv4_hgnetv2_m.onnx")
    p.add_argument("--reid", default="osnet_x1_0_msmt17.pt")
    p.add_argument("--input", default="videos/gunsan_test.mp4")
    p.add_argument("--output", default="runs/track/rtdetr_db.mp4")
    p.add_argument("--gallery", default="gallery/persons.npz")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--person-class", type=int, default=0)
    # DB / resolver knobs
    p.add_argument("--k-per-person", type=int, default=10)
    p.add_argument(
        "--match-threshold", type=float, default=0.25,
        help="cosine distance ceiling for declaring a match (lower = stricter)",
    )
    p.add_argument(
        "--ratio-threshold", type=float, default=0.85,
        help="Lowe-style ratio: best/second-best must be < this to commit a match",
    )
    p.add_argument(
        "--gallery-update-max-dist", type=float, default=0.20,
        help="per-frame gallery updates are skipped if cur_dist > this",
    )
    p.add_argument("--decision-frames", type=int, default=8)
    p.add_argument("--quality-min-conf", type=float, default=0.7)
    args = p.parse_args()
    run(args)
