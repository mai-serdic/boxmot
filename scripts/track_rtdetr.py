"""
RT-DETRv4 (ONNX) + BoxMOT Person Tracking Script
--------------------------------------------------
Runs the exported ONNX model for detection and feeds
only 'person' boxes into BoxMOT's BoT-SORT tracker.

Usage:
    conda activate boxmot_env
    python track_rtdetr.py \\
        --onnx  models/20260504_rtv4_hgnetv2_m.onnx \\
        --input videos/gunsan_test.mp4 \\
        --output runs/track/rtdetr_tracked.mp4
"""

import argparse
import os
import sys
from pathlib import Path

# Make the repo root importable so `boxmot` and `reid` resolve when this
# script is run as scripts/track_rtdetr.py.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import onnxruntime as ort

from boxmot.trackers.botsort.botsort import BotSort
from boxmot.reid.core import ReID


# ─────────────────────────────────────────────────────────────────────────────
# ONNX Detector wrapper
# ─────────────────────────────────────────────────────────────────────────────

class RTDetrOnnx:
    """Loads the RT-DETRv4 ONNX and runs inference frame-by-frame."""

    INPUT_SIZE = (640, 640)  # h, w — must match what was used during export

    def __init__(self, onnx_path: str, device: str = "cuda:0"):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "cuda" in device
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.size_name = self.session.get_inputs()[1].name

    def predict(self, frame_bgr: np.ndarray, conf_thresh: float = 0.4, person_class: int = 0):
        """
        Returns detections as numpy array: (N, 6) → [x1, y1, x2, y2, conf, cls]
        Coordinates are in original frame pixel space.
        """
        orig_h, orig_w = frame_bgr.shape[:2]
        h, w = self.INPUT_SIZE

        # Pre-process
        img = cv2.resize(frame_bgr, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis]  # (1, 3, H, W)
        orig_size = np.array([[orig_w, orig_h]], dtype=np.int64)

        # Inference
        labels, boxes, scores = self.session.run(
            None,
            {self.input_name: img, self.size_name: orig_size}
        )
        # labels: (1, N), boxes: (1, N, 4), scores: (1, N)
        labels = labels[0]   # (N,)
        boxes  = boxes[0]    # (N, 4) in orig pixel coords
        scores = scores[0]   # (N,)

        # Filter by class and confidence
        mask = (labels == person_class) & (scores >= conf_thresh)
        if mask.sum() == 0:
            return np.empty((0, 6), dtype=np.float32)

        dets = np.column_stack([
            boxes[mask],
            scores[mask],
            labels[mask].astype(float),
        ]).astype(np.float32)
        return dets


# ─────────────────────────────────────────────────────────────────────────────
# Main tracking loop
# ─────────────────────────────────────────────────────────────────────────────

def run(args):
    device_str = args.device
    torch_device = torch.device(device_str)

    # ── 1. Detector ───────────────────────────────────────────────────────────
    print(f"[INFO] Loading ONNX detector: {args.onnx}")
    detector = RTDetrOnnx(args.onnx, device=device_str)

    # ── 2. ReID model ─────────────────────────────────────────────────────────
    project_root = Path(__file__).resolve().parent.parent  # boxmot/
    reid_path = Path(args.reid)
    if not reid_path.is_absolute():
        reid_path = project_root / args.reid
    print(f"[INFO] Loading ReID model: {reid_path}")
    reid_model = ReID(weights=reid_path, device=torch_device, half=False).get_backend()

    # ── 3. Video I/O (opened first so the tracker gets the real fps) ─────────
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.input}")

    fps       = cap.get(cv2.CAP_PROP_FPS) or 25
    orig_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Video {orig_w}x{orig_h} @ {fps:.1f} fps  ({total} frames)")

    # ── 4. Tracker ───────────────────────────────────────────────────────────
    tracker = BotSort(
        reid_model=reid_model,
        track_high_thresh=0.5,
        track_low_thresh=0.1,
        new_track_thresh=0.7,
        track_buffer=600,        # 600 frames @ 15fps ≈ 40s lost-track memory
        match_thresh=0.8,
        proximity_thresh=0.9,    # let ReID rescue spatially-far re-appearances
        appearance_thresh=0.4,   # looser cosine ceiling for OSNet x0.25
        cmc_method="ecc",
        frame_rate=int(round(fps)),
        with_reid=True,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    writer = cv2.VideoWriter(
        args.output,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (orig_w, orig_h)
    )

    # ── 5. Loop ───────────────────────────────────────────────────────────────
    print("[INFO] Tracking …")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets   = detector.predict(frame, conf_thresh=args.conf, person_class=args.person_class)
        tracks = tracker.update(dets, frame)
        # tracks: (M, 8) → x1 y1 x2 y2 id conf cls det_ind

        if len(tracks):
            for t in tracks:
                x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
                tid = int(t[4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, f"ID {tid}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA,
                )

        writer.write(frame)
        frame_idx += 1
        if frame_idx % 100 == 0:
            pct = frame_idx / total * 100 if total else 0
            print(f"  frame {frame_idx}/{total}  ({pct:.1f}%)  active tracks={len(tracks)}")

    cap.release()
    writer.release()
    print(f"\n[DONE] Saved: {args.output}")

    # Re-encode to H.264 for universal playback
    fixed = args.output.replace(".mp4", "_h264.mp4")
    os.system(f'ffmpeg -y -i "{args.output}" -vcodec libx264 -crf 23 "{fixed}" -loglevel quiet')
    if os.path.exists(fixed):
        print(f"[DONE] H.264 copy: {fixed}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RT-DETRv4 ONNX + BoxMOT person tracking")
    parser.add_argument("--onnx",   default="models/20260504_rtv4_hgnetv2_m.onnx")
    parser.add_argument("--reid",   default="osnet_x0_25_msmt17.pt")
    parser.add_argument("--input",  default="videos/gunsan_test.mp4")
    parser.add_argument("--output", default="runs/track/rtdetr_tracked.mp4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--conf",   type=float, default=0.4)
    parser.add_argument("--person-class", type=int, default=0)
    args = parser.parse_args()
    run(args)
