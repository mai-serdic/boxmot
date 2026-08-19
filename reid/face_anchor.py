"""
Pose-gated face anchoring for persistent person re-identification.

Why this exists (see bench/REPORT.md):
  Body-appearance embeddings (CLIP-ReID) floor at ~0.22 EER on this footage —
  no match threshold makes both fragmentation and merges small. Face is the
  clothing-independent second cue: when a worker faces the camera closely
  enough, an ArcFace embedding separates identities far more sharply than any
  body embedding, and it survives clothing changes across days.

The catch is that face embeddings are only trustworthy under a narrow set of
conditions (frontal-ish view, sufficient resolution, confident detection).
Outside those conditions ArcFace degrades into noise that would poison a
persistent gallery. So every observation passes a *pose gate* before it is
allowed to influence identity:

  1. detection confidence  — SCRFD det_score >= det_score_min
  2. resolution            — face min side >= min_face_px
  3. frontality (yaw)      — nose x must sit between the eyes
                             (profile views push the nose past an eye)
  4. frontality (roll/up)  — eye separation must be a reasonable fraction of
                             face width (collapses on profile / extreme tilt)

Only gated observations are returned; callers treat a returned embedding as a
high-precision identity anchor and everything else as "no face evidence"
(which is the common case — on the benchmark footage ~26% of quality person
crops pass the gate).

Model: insightface `buffalo_l` (SCRFD-10G detection + ArcFace-R50 w600k
recognition). Landmark/genderage submodules are not loaded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FaceObservation:
    """A pose-gated face seen inside a person crop."""

    embedding: np.ndarray   # (512,) L2-normalized ArcFace
    det_score: float        # SCRFD detection confidence
    face_size: float        # min(face_w, face_h) in source-image pixels
    frontality: float       # 0..1, 1 = nose perfectly centred between the eyes
    quality: float          # det_score * size * frontality — gallery slot score

    @property
    def is_strong(self) -> bool:
        """Anchor-grade observation: allowed to override a body-appearance
        decision, not just supplement it."""
        return self.det_score >= 0.6 and self.face_size >= 36 and self.frontality >= 0.6


def pose_gate(
    bbox: np.ndarray,
    kps: np.ndarray,
    det_score: float,
    det_score_min: float = 0.5,
    min_face_px: float = 24.0,
    nose_band: tuple[float, float] = (0.2, 0.8),
    min_eye_frac: float = 0.22,
) -> tuple[bool, float, float]:
    """Apply the pose gate to a raw face detection (no model needed).

    bbox: (4,) face box, kps: (5, 2) landmarks in the order
    [l_eye, r_eye, nose, l_mouth, r_mouth].

    Returns (passed, face_size, frontality).
    """
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    face_w, face_h = x2 - x1, y2 - y1
    size = min(face_w, face_h)
    if det_score < det_score_min or size < min_face_px:
        return False, size, 0.0

    eye_dx = float(kps[1, 0] - kps[0, 0])
    if face_w <= 0 or eye_dx < min_eye_frac * face_w:
        return False, size, 0.0

    nose_ratio = float(kps[2, 0] - kps[0, 0]) / eye_dx
    lo, hi = nose_band
    if not (lo < nose_ratio < hi):
        return False, size, 0.0

    # 1.0 when the nose is dead-centre (ratio 0.5), falling to 0 at the
    # band edges — a smooth yaw proxy usable as a quality weight.
    half_band = (hi - lo) / 2.0
    frontality = max(0.0, 1.0 - abs(nose_ratio - 0.5) / half_band)
    return True, size, frontality


class FaceAnchor:
    """Detects + embeds the dominant face in a person crop, applying the pose
    gate described in the module docstring. Returns None when no trustworthy
    face is visible — callers must treat that as "no evidence", not "no match".
    """

    def __init__(
        self,
        det_size: tuple[int, int] = (320, 320),
        det_score_min: float = 0.5,
        min_face_px: float = 24.0,
        nose_band: tuple[float, float] = (0.2, 0.8),
        min_eye_frac: float = 0.22,
        providers: list[str] | None = None,
        ctx_id: int = 0,
    ):
        from insightface.app import FaceAnalysis  # deferred: heavy import

        self.det_score_min = det_score_min
        self.min_face_px = min_face_px
        self.nose_band = nose_band
        self.min_eye_frac = min_eye_frac

        self.app = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        # det_thresh is set below det_score_min on purpose: detection recall is
        # cheap, the gate does the precision work.
        self.app.prepare(ctx_id=ctx_id, det_size=det_size, det_thresh=0.35)

    # ── gate ────────────────────────────────────────────────────────────────
    def gate(
        self,
        bbox: np.ndarray,
        kps: np.ndarray,
        det_score: float,
    ) -> tuple[bool, float, float]:
        """Instance-configured wrapper around :func:`pose_gate`."""
        return pose_gate(
            bbox, kps, det_score,
            det_score_min=self.det_score_min,
            min_face_px=self.min_face_px,
            nose_band=self.nose_band,
            min_eye_frac=self.min_eye_frac,
        )

    # ── extraction ──────────────────────────────────────────────────────────
    def extract(self, person_crop_bgr: np.ndarray) -> FaceObservation | None:
        """Best gated face in a person crop, or None."""
        if person_crop_bgr is None or person_crop_bgr.size == 0:
            return None
        faces = self.app.get(person_crop_bgr)
        best: FaceObservation | None = None
        for f in faces:
            ok, size, frontality = self.gate(f.bbox, f.kps, float(f.det_score))
            if not ok:
                continue
            emb = np.asarray(f.normed_embedding, dtype=np.float32).reshape(-1)
            n = np.linalg.norm(emb)
            if n < 1e-6:
                continue
            obs = FaceObservation(
                embedding=emb / n,
                det_score=float(f.det_score),
                face_size=float(size),
                frontality=float(frontality),
                quality=float(f.det_score) * float(size) * float(frontality),
            )
            if best is None or obs.quality > best.quality:
                best = obs
        return best
