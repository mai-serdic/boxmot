"""
Step 1 of the ReID embedding benchmark.

Runs RT-DETRv4 + *motion-only* ByteTrack over a video to produce weak identity
labels WITHOUT any appearance model in the loop — so the labels can't be biased
toward whichever embedding we later evaluate.

Two label sources fall out of the tracklets:
  * POSITIVES  — two crops from the SAME tracklet (same trk_id) are the same
                 physical person. Sampling them far apart in time captures the
                 pose / viewpoint / side changes we care about.
  * NEGATIVES  — two tracklets that are active in the SAME frame are, by
                 physics, different people (one body can't be in two places).
                 We record each track's active-frame set so step 2 can build
                 these guaranteed-different pairs.

Outputs (to --outdir):
  crops/crop_XXXXXX.jpg        subsampled person crops (shared by every model)
  manifest.json                [{cid, trk, frame, bbox:[x1,y1,x2,y2], conf}, ...]
  active_frames.json           {trk_id: [frame, frame, ...]}  (co-active lookup)
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import onnxruntime as ort

from boxmot.trackers.bytetrack.bytetrack import ByteTrack


class RTDetrOnnx:
    INPUT_SIZE = (640, 640)

    def __init__(self, onnx_path: str, device: str = "cuda:0"):
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "cuda" in device else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        print(f"[INFO] ORT providers: {self.session.get_providers()}")
        self.input_name = self.session.get_inputs()[0].name
        self.size_name = self.session.get_inputs()[1].name

    def predict(self, frame_bgr, conf_thresh=0.4, person_class=0):
        orig_h, orig_w = frame_bgr.shape[:2]
        h, w = self.INPUT_SIZE
        img = cv2.resize(frame_bgr, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis]
        orig_size = np.array([[orig_w, orig_h]], dtype=np.int64)
        labels, boxes, scores = self.session.run(
            None, {self.input_name: img, self.size_name: orig_size}
        )
        labels, boxes, scores = labels[0], boxes[0], scores[0]
        mask = (labels == person_class) & (scores >= conf_thresh)
        if mask.sum() == 0:
            return np.empty((0, 6), dtype=np.float32)
        return np.column_stack(
            [boxes[mask], scores[mask], labels[mask].astype(float)]
        ).astype(np.float32)


def is_quality_crop(x1, y1, x2, y2, conf, W, H,
                    min_area=60 * 120, min_aspect=1.3, edge=8, min_conf=0.5):
    """Same spirit as IdentityResolver.is_quality_crop — clean crops only, so
    every model is scored on the same well-formed inputs."""
    w, h = x2 - x1, y2 - y1
    if w * h < min_area:
        return False
    if conf < min_conf:
        return False
    if h <= w * min_aspect:
        return False
    if x1 < edge or y1 < edge or x2 > W - edge or y2 > H - edge:
        return False
    return True


def run(args):
    outdir = Path(args.outdir)
    crops_dir = outdir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    detector = RTDetrOnnx(args.onnx, device=args.device)
    tracker = ByteTrack(
        min_conf=0.1,
        track_thresh=0.5,
        match_thresh=0.8,
        track_buffer=int(args.track_buffer),
        frame_rate=int(round(args.fps)),
    )

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(args.input)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    manifest = []
    active_frames: dict[int, list[int]] = {}
    last_saved: dict[int, int] = {}
    saved_count: dict[int, int] = {}
    cid = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        H, W = frame.shape[:2]
        dets = detector.predict(frame, conf_thresh=args.conf, person_class=args.person_class)
        tracks = tracker.update(dets, frame)  # motion only, no embs

        for t in tracks:
            x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
            tid = int(t[4])
            conf = float(t[5])
            active_frames.setdefault(tid, []).append(frame_idx)

            if not is_quality_crop(x1, y1, x2, y2, conf, W, H):
                continue
            if frame_idx - last_saved.get(tid, -10**9) < args.save_stride:
                continue
            if saved_count.get(tid, 0) >= args.max_per_track:
                continue

            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            fn = crops_dir / f"crop_{cid:06d}.jpg"
            cv2.imwrite(str(fn), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            manifest.append({
                "cid": cid, "trk": tid, "frame": frame_idx,
                "bbox": [x1, y1, x2, y2], "conf": round(conf, 4),
            })
            last_saved[tid] = frame_idx
            saved_count[tid] = saved_count.get(tid, 0) + 1
            cid += 1

        frame_idx += 1
        if frame_idx % 250 == 0:
            print(f"  frame {frame_idx}/{total}  tracks={len(active_frames)}  crops={cid}")

    cap.release()
    (outdir / "manifest.json").write_text(json.dumps(manifest))
    (outdir / "active_frames.json").write_text(
        json.dumps({str(k): v for k, v in active_frames.items()})
    )

    # Summary
    per_trk = {}
    for m in manifest:
        per_trk.setdefault(m["trk"], 0)
        per_trk[m["trk"]] += 1
    multi = {k: v for k, v in per_trk.items() if v >= 2}
    print(f"\n[DONE] frames={frame_idx}  total_tracks={len(active_frames)}")
    print(f"[DONE] saved crops={cid}  tracks_with_>=2_crops={len(multi)} "
          f"(these seed the positive pairs)")
    print(f"[DONE] crops/ and manifest.json written to {outdir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default=str(PROJECT_ROOT / "models/20260504_rtv4_hgnetv2_m.onnx"))
    p.add_argument("--input", default=str(PROJECT_ROOT / "videos/gunsan_test.mp4"))
    p.add_argument("--outdir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--person-class", type=int, default=0)
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--track-buffer", type=int, default=30)
    p.add_argument("--save-stride", type=int, default=5,
                   help="min frames between saved crops of the same track")
    p.add_argument("--max-per-track", type=int, default=25)
    run(p.parse_args())
