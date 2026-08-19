"""Turn footage into motion-only tracklets - the one input calibration needs.

    python scripts/make_tracklets.py --input videos/site.mp4 --out runs/traj/site.json
    python scripts/make_tracklets.py --input frames/site/    --out runs/traj/site.json

Accepts a video file or a directory of image frames (sorted by filename, which
is how exported CCTV stills are normally named).

Deliberately **motion-only** ByteTrack, with no appearance model. Calibration
does not need to know who anybody is - it only needs many examples of a person
standing upright in different parts of the room. Leaving ReID out keeps this
step fast, keeps it from inheriting the appearance failure this whole line of
work exists to route around, and means the tracklets are an honest input to a
benchmark that is later used to judge appearance.

Output schema (also what `commission_scene.py` and `bench/06` read):

    {"W": int, "H": int, "frames": int,
     "traj": {"<track_id>": [[frame, x1, y1, x2, y2], ...]}}
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class RTDetrOnnx:
    """RT-DETRv4 ONNX, person class only."""

    def __init__(self, onnx_path: str, device: str = "cuda"):
        import onnxruntime as ort

        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if "cuda" in device else ["CPUExecutionProvider"])
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.size_name = self.sess.get_inputs()[1].name
        print(f"[det ] {onnx_path}  providers={self.sess.get_providers()[0]}")

    def __call__(self, frame: np.ndarray, conf: float, person_class: int) -> np.ndarray:
        h, w = frame.shape[:2]
        img = cv2.resize(frame, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None]
        lab, box, sc = self.sess.run(
            None, {self.in_name: img,
                   self.size_name: np.array([[w, h]], dtype=np.int64)})
        lab, box, sc = lab[0], box[0], sc[0]
        m = (lab == person_class) & (sc >= conf)
        if not m.any():
            return np.empty((0, 6), np.float32)
        return np.column_stack([box[m], sc[m], lab[m].astype(float)]).astype(np.float32)


def frame_source(path: str):
    """Yields (index, frame) from a video file or a directory of images."""
    p = Path(path)
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMG_EXT)
        if not files:
            raise SystemExit(f"no images found in {p}")
        print(f"[in  ] {len(files)} images from {p}")
        for i, f in enumerate(files):
            img = cv2.imread(str(f))
            if img is not None:
                yield i, img
    else:
        cap = cv2.VideoCapture(str(p))
        if not cap.isOpened():
            raise SystemExit(f"cannot open {p}")
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[in  ] video {p} ({n} frames)")
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            yield i, fr
            i += 1
        cap.release()


def build(input_path: str, onnx: str, conf: float = 0.4, fps: float = 15.0,
          device: str = "cuda", person_class: int = 0,
          progress_every: int = 250) -> dict:
    from boxmot.trackers.bytetrack.bytetrack import ByteTrack

    det = RTDetrOnnx(onnx, device)
    trk = ByteTrack(min_conf=0.1, track_thresh=0.5, match_thresh=0.8,
                    track_buffer=30, frame_rate=fps)
    traj: dict[int, list] = {}
    W = H = n = 0
    for i, fr in frame_source(input_path):
        H, W = fr.shape[:2]
        for t in trk.update(det(fr, conf, person_class), fr):
            traj.setdefault(int(t[4]), []).append(
                [i, float(t[0]), float(t[1]), float(t[2]), float(t[3])])
        n = i + 1
        if progress_every and n % progress_every == 0:
            print(f"       frame {n}  tracks so far {len(traj)}", flush=True)
    return {"W": W, "H": H, "frames": n,
            "traj": {str(k): v for k, v in traj.items()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="video file or directory of frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--onnx", default="models/20260504_rtv4_hgnetv2_m.onnx")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--person-class", type=int, default=0)
    args = ap.parse_args()

    out = build(args.input, args.onnx, args.conf, args.fps,
                args.device, args.person_class)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    long = sum(1 for v in out["traj"].values() if len(v) >= 15)
    print(f"[save] {args.out}: {len(out['traj'])} tracklets "
          f"({long} with >=15 frames) over {out['frames']} frames, "
          f"{out['W']}x{out['H']}")
    if long < 8:
        print("[warn] very few usable tracklets - calibration needs people walking "
              "around the room. Use a longer clip or a busier period.")


if __name__ == "__main__":
    main()
