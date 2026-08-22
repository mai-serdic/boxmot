"""
Multi-signal lost-track rebinding for within-session stable IDs.

The problem this solves:
  BYTETracker / BoT-SORT drop a track after `track_buffer` frames of no
  detection. When the same physical person re-appears (walking out from
  behind a wall, a forklift, another worker), the tracker spawns a NEW
  trk_id. Downstream state keyed by trk_id (e.g. PPE-compliance buffers)
  resets, producing a long PENDING window and noisy alerts.

The design:
  We keep a *ghost pool* of recently-lost tracks alongside the live tracker.
  When a brand-new trk_id is born, we compare it against the ghost pool
  using a weighted combination of signals (spatial prior, time decay,
  body embedding, height, optional hat color). If the best ghost scores
  above threshold, we *rebind*: the new trk_id is mapped back to the
  ghost's original trk_id for all downstream purposes.

What this is NOT:
  - Not identity recognition. Stable anonymous IDs only.
  - Not a tracker. Sits on top of one.
  - Not cross-session. Pool is cleared between shifts.

Calibration philosophy:
  Each individual signal is unreliable on its own (body ReID in uniformed
  workforce, spatial prior across long gaps, hat color at collision scale).
  The combined score is robust because the failure modes are independent.
  Permissive threshold is acceptable because a wrong rebind is recoverable
  (offline merge), unlike a wrong identity assignment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# Hard import on purpose. This used to be wrapped in `except Exception: pass`
# "because the module is optional", and when the package layout changed the
# import broke silently: rebind_prior went None, _spatial_prior_geo returned
# None on every call, `feasible` defaulted to True so the reachability veto
# never fired, and summary() still printed "prior=geodesic". The pool ran on
# appearance and a pixel Gaussian while reporting that it was using the floor.
# reachability is a sibling module in this package -- it cannot be absent, and
# if it ever is, that is a broken install and should say so loudly.
from .reachability import rebind_prior


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class GhostTrack:
    """A track that has been lost (no detection for >1 frame) but kept alive
    in the ghost pool for possible rebinding.

    ``embeddings`` holds up to K body-embedding views of this person captured
    across the track's lifetime (selected by quality score). Matching against
    a new track uses the *best* (highest cosine similarity) of these — so a
    track that was last seen back-view can still match a front-view re-emerge
    if a front-view embedding was captured during its lifetime.
    """

    trk_id: int                                 # original tracker ID
    last_frame: int                             # frame where last detected
    last_bbox: tuple[int, int, int, int]        # x1, y1, x2, y2 at loss
    last_velocity: tuple[float, float]          # (vx, vy) in px/frame
    embeddings: list[np.ndarray]                # K-best L2-normalized embeddings
    avg_height: float                           # avg bbox height over lifetime
    avg_width: float                            # avg bbox width over lifetime
    hat_color_hsv: Optional[tuple[float, float, float]] = None  # optional

    # Metric floor state at the moment of loss, in the ground-plane frame.
    # Present only when the camera has been commissioned; the pixel-space
    # fallback still works without it.
    last_xy: Optional[tuple[float, float]] = None        # metres
    last_vel_xy: Optional[tuple[float, float]] = None    # metres/second

    # Internal stats for diagnostics
    lifetime_frames: int = 0
    n_quality_obs: int = 0


@dataclass
class NewTrackInfo:
    """Information about a freshly-spawned track, presented to the pool for
    rebinding lookup."""

    bbox: tuple[int, int, int, int]
    embedding: np.ndarray                       # L2-normalized
    height: float
    width: float
    hat_color_hsv: Optional[tuple[float, float, float]] = None
    xy: Optional[tuple[float, float]] = None             # metres, floor frame


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ScoringWeights:
    """Tunable weights for the multi-signal rebind score.

    All signals are mapped to [0, 1] before weighting. The weights need not
    sum to 1 but doing so makes the final score interpretable as a
    probability-like quantity.
    """
    w_spatial:   float = 0.45
    w_time:      float = 0.10
    w_embedding: float = 0.30
    w_height:    float = 0.10
    w_hat:       float = 0.05

    # Spatial-prior uncertainty: how fast the predicted-position ellipse
    # grows per frame (in pixels of standard deviation). Empirically tied
    # to typical walking-speed jitter and detection noise.
    spatial_sigma_per_frame: float = 4.0

    # Time decay: probability of "still here" halves every this many frames.
    time_decay_half_life: float = 600.0    # 600 frames @15fps = 40 sec

    # Veto: if embedding similarity is *worse* than this, refuse the rebind
    # regardless of the rest of the score. Catches obvious "wrong person".
    embedding_veto_max_dist: float = 0.55  # cosine distance

    # Reachability veto: a candidate the person could not have *walked* to in
    # the time available is physically impossible, not merely unlikely, so it
    # is penalised harder than the embedding veto. Kept as a multiplier rather
    # than a hard zero so the logs still show what was rejected and why.
    reach_veto_scale: float = 0.05

    # Below this gap the pixel Gaussian is still the sharper of the two priors
    # - the box has barely moved and metric localisation noise (~0.2 m here)
    # is larger than the displacement being measured - so the geodesic prior
    # is not allowed to override it. Above it the pixel prior has decayed to
    # noise and the geodesic one carries the decision. Measured crossover on
    # the reference clip is ~20 frames at 15 fps (bench/08).
    px_trust_frames: int = 20


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _spatial_prior_px(ghost: GhostTrack, new: NewTrackInfo,
                      current_frame: int, w: ScoringWeights) -> float:
    """Pixel-space fallback for an uncommissioned camera.

    Gaussian likelihood that the new bbox sits where the ghost should be, with
    sigma growing as sqrt(elapsed). It knows nothing about metres or about
    solid objects, so it happily places a person inside a cabinet and goes
    uniform after a few seconds. Use `_spatial_prior_geo` wherever a scene
    model exists; this remains only so the tracker still runs without one.
    """
    elapsed = max(1, current_frame - ghost.last_frame)
    cx_pred = _bbox_center(ghost.last_bbox)[0] + ghost.last_velocity[0] * elapsed
    cy_pred = _bbox_center(ghost.last_bbox)[1] + ghost.last_velocity[1] * elapsed
    cx_new, cy_new = _bbox_center(new.bbox)

    sigma = w.spatial_sigma_per_frame * math.sqrt(elapsed)
    d2 = (cx_new - cx_pred) ** 2 + (cy_new - cy_pred) ** 2
    # Gaussian likelihood normalized to [0, 1] (peak = 1 at d=0)
    return math.exp(-0.5 * d2 / (sigma ** 2))


def _spatial_prior_geo(ghost: GhostTrack, new: NewTrackInfo,
                       current_frame: int, reach, fps: float,
                       params) -> Optional[dict]:
    """Geodesic prior over the walkable floor. None if either end lacks a
    metric position, so the caller can fall back."""
    if reach is None:
        return None
    if ghost.last_xy is None or new.xy is None:
        return None
    elapsed_s = max(current_frame - ghost.last_frame, 1) / max(fps, 1e-6)
    vel = ghost.last_vel_xy or (0.0, 0.0)
    return rebind_prior(reach, ghost.last_xy, vel, new.xy, elapsed_s, params)


def _time_decay(ghost: GhostTrack, current_frame: int,
                w: ScoringWeights) -> float:
    elapsed = max(0, current_frame - ghost.last_frame)
    return 0.5 ** (elapsed / w.time_decay_half_life)


def _embedding_sim(ghost: GhostTrack, new: NewTrackInfo) -> tuple[float, float]:
    """Best-of-K similarity over the ghost's stored embeddings.

    Returns (similarity_in_[0,1], cosine_distance_in_[0,2]).
    Taking the *max* over K means that if the ghost captured a front-view
    at some point during its lifetime, a future front-view query will match
    against it — even if the most-recent embedding was a back-view at the
    moment the track disappeared.
    """
    if not ghost.embeddings:
        return 0.0, 2.0
    sims = np.array(
        [float(np.dot(e, new.embedding)) for e in ghost.embeddings],
        dtype=np.float32,
    )
    cos = float(sims.max())
    dist = 1.0 - cos
    sim = max(0.0, min(1.0, (1.0 + cos) / 2.0))   # remap [-1,1] -> [0,1]
    return sim, dist


def _height_consistency(ghost: GhostTrack, new: NewTrackInfo) -> float:
    """Closer ratios → higher score. People's apparent height changes with
    camera distance, but two side-by-side workers usually differ by a clear
    margin."""
    h1, h2 = ghost.avg_height, max(new.height, 1.0)
    ratio = min(h1, h2) / max(h1, h2)
    # Map ratio in [0,1] with a soft curve: very forgiving above 0.85,
    # sharper penalty below 0.7.
    return float(ratio ** 2)


def _hat_color_match(ghost: GhostTrack, new: NewTrackInfo) -> float:
    """HSV-space hat color similarity. Returns neutral 0.5 if either side
    lacks hat info — refusing to penalize when the signal is missing."""
    if ghost.hat_color_hsv is None or new.hat_color_hsv is None:
        return 0.5
    h1, s1, v1 = ghost.hat_color_hsv
    h2, s2, v2 = new.hat_color_hsv
    # Hue is circular [0, 180] in OpenCV
    dh = min(abs(h1 - h2), 180.0 - abs(h1 - h2)) / 90.0
    ds = abs(s1 - s2) / 255.0
    dv = abs(v1 - v2) / 255.0
    # Weight hue heavily — colors of helmets are designed to be hue-distinct
    d = 0.7 * dh + 0.2 * ds + 0.1 * dv
    return max(0.0, 1.0 - d)


def score_rebind(ghost: GhostTrack, new: NewTrackInfo, current_frame: int,
                 w: ScoringWeights, reach=None, fps: float = 15.0,
                 reach_params=None) -> tuple[float, dict]:
    """Combined score in [0, 1]. Returns (score, breakdown_for_logging).

    When a commissioned scene is supplied the spatial term is a geodesic
    walking prior over the floor and an unreachable candidate is vetoed;
    otherwise it degrades to the pixel-space Gaussian.
    """
    sim, emb_dist = _embedding_sim(ghost, new)
    px = _spatial_prior_px(ghost, new, current_frame, w)
    geo = _spatial_prior_geo(ghost, new, current_frame, reach, fps, reach_params)
    if geo is None:
        spatial, geo_m, feasible, portal = px, float("nan"), True, 0.0
    else:
        geo_m, feasible, portal = geo["geo_m"], geo["feasible"], geo["portal"]
        # The two priors are not rivals; they have different error models and
        # they fail in different regimes. The pixel Gaussian is sharp while the
        # box has barely moved and collapses to noise after a second or two;
        # the geodesic prior is the only one that still says anything after
        # that, and the only one that knows furniture is solid. Hand over at
        # the measured crossover rather than blending, which loses to both.
        gap = current_frame - ghost.last_frame
        spatial = px if gap < w.px_trust_frames else max(px, geo["prior"])
        if not feasible:
            spatial = 0.0        # physically impossible outranks both
    breakdown = {
        "spatial":   spatial,
        "spatial_px": px,
        "geo_m":     geo_m,
        "reachable": feasible,
        "portal":    portal,
        "time":      _time_decay(ghost, current_frame, w),
        "embedding": sim,
        "height":    _height_consistency(ghost, new),
        "hat":       _hat_color_match(ghost, new),
        "emb_dist":  emb_dist,
    }
    score = (
        w.w_spatial   * breakdown["spatial"] +
        w.w_time      * breakdown["time"] +
        w.w_embedding * breakdown["embedding"] +
        w.w_height    * breakdown["height"] +
        w.w_hat       * breakdown["hat"]
    )
    # Reachability veto — the person could not have walked there in the time
    # available, going around the furniture. Physically impossible beats any
    # amount of appearance agreement, so this is the harsher of the two vetoes.
    if not feasible:
        score *= w.reach_veto_scale
    # Embedding veto — actively contradicting embedding kills the score
    if emb_dist > w.embedding_veto_max_dist:
        score *= 0.1  # heavy penalty rather than hard 0, lets logs show it
    return score, breakdown


# ─────────────────────────────────────────────────────────────────────────────
# Pool
# ─────────────────────────────────────────────────────────────────────────────
class GhostPool:
    """Keeps recently-lost tracks for possible rebinding to brand-new tracks.

    Typical lifecycle per frame:
      1. tracker.update(...) → live tracks
      2. for each trk_id that disappeared since last frame: pool.add(...)
      3. for each brand-new trk_id this frame: pool.best_match(...) → rebind?
      4. pool.cleanup(current_frame) at the end of the frame
    """

    def __init__(
        self,
        ttl_frames: int = 9000,        # 10 min @ 15 fps
        max_size: int = 200,
        weights: Optional[ScoringWeights] = None,
        threshold: float = 0.55,
        reach=None,
        fps: float = 15.0,
        reach_params=None,
    ):
        self.ghosts: dict[int, GhostTrack] = {}
        self.ttl = ttl_frames
        self.max_size = max_size
        self.w = weights or ScoringWeights()
        self.threshold = threshold
        # Optional `reachability.Reachability`. Absent -> pixel-space fallback.
        self.reach = reach
        self.fps = float(fps)
        self.reach_params = reach_params
        self.n_reach_veto = 0

        # Diagnostics
        self.n_added = 0
        self.n_rebound = 0
        self.n_expired = 0

    # ─── lifecycle ───────────────────────────────────────────────────────────
    def add(self, ghost: GhostTrack) -> None:
        """Add a freshly-lost track to the pool."""
        if len(self.ghosts) >= self.max_size:
            # Evict the oldest by last_frame
            oldest = min(self.ghosts.values(), key=lambda g: g.last_frame)
            del self.ghosts[oldest.trk_id]
        self.ghosts[ghost.trk_id] = ghost
        self.n_added += 1

    def cleanup(self, current_frame: int) -> int:
        """Drop ghosts older than ttl. Returns count expired."""
        expired = [tid for tid, g in self.ghosts.items()
                   if current_frame - g.last_frame > self.ttl]
        for tid in expired:
            del self.ghosts[tid]
        self.n_expired += len(expired)
        return len(expired)

    def remove(self, trk_id: int) -> None:
        self.ghosts.pop(trk_id, None)

    # ─── matching ────────────────────────────────────────────────────────────
    def best_match(
        self,
        new: NewTrackInfo,
        current_frame: int,
        exclude_trk_ids: Optional[set[int]] = None,
        verbose: bool = False,
    ) -> Optional[tuple[int, float, dict]]:
        """Return (rebound_trk_id, score, breakdown) or None.

        ``exclude_trk_ids``: ghost trk_ids forbidden from matching (e.g.
        because they're already bound to another live track this frame).
        Mirrors the uniqueness constraint we use in PersonDB.
        """
        if not self.ghosts:
            return None
        excluded = exclude_trk_ids or set()
        scored: list[tuple[float, int, dict]] = []
        for trk_id, ghost in self.ghosts.items():
            if trk_id in excluded:
                continue
            s, br = score_rebind(ghost, new, current_frame, self.w,
                                 reach=self.reach, fps=self.fps,
                                 reach_params=self.reach_params)
            if not br["reachable"]:
                self.n_reach_veto += 1
            scored.append((s, trk_id, br))
        if not scored:
            return None
        scored.sort(reverse=True)
        best_score, best_id, best_br = scored[0]
        if verbose:
            top = scored[: min(3, len(scored))]
            print(f"  [GHOST] top candidates: " +
                  ", ".join(f"trk={t} s={s:.2f} d={br['emb_dist']:.2f}"
                            f" geo={br['geo_m']:.1f}m"
                            f"{'' if br['reachable'] else ' UNREACHABLE'}"
                            f"{' portal' if br['portal'] else ''}"
                            for s, t, br in top))
        if best_score < self.threshold:
            return None
        # Pop the matched ghost out of the pool — it now lives on as the
        # rebound trk_id and shouldn't match anything else.
        del self.ghosts[best_id]
        self.n_rebound += 1
        return best_id, best_score, best_br

    # ─── stats ───────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.ghosts)

    def summary(self) -> str:
        mode = "geodesic" if self.reach is not None else "pixel"
        return (f"GhostPool[live={len(self.ghosts)}, added={self.n_added}, "
                f"rebound={self.n_rebound}, expired={self.n_expired}, "
                f"prior={mode}, reach_veto={self.n_reach_veto}]")


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    def rnorm(d: int = 512) -> np.ndarray:
        v = rng.standard_normal(d).astype(np.float32)
        v /= np.linalg.norm(v)
        return v

    pool = GhostPool(ttl_frames=300, threshold=0.55)

    # Person A walking right at (500, 400), velocity (3, 0)
    emb_a = rnorm()
    # Ghost A captured TWO views during its lifetime: a front view and a back
    # view. This is the key change — matching takes the best of K.
    emb_a_back = (emb_a + 0.7 * rnorm()).astype(np.float32)
    emb_a_back /= np.linalg.norm(emb_a_back)
    ghost_a = GhostTrack(
        trk_id=42,
        last_frame=100,
        last_bbox=(460, 300, 540, 500),
        last_velocity=(3.0, 0.0),
        embeddings=[emb_a, emb_a_back],   # K=2: front + back view
        avg_height=200.0,
        avg_width=80.0,
        hat_color_hsv=(15.0, 200.0, 200.0),  # yellow
    )
    pool.add(ghost_a)

    # 30 frames later, a new track appears where ghost A should be —
    # slightly noisy embedding, slightly noisy position
    new_at_predicted = NewTrackInfo(
        bbox=(550, 300, 630, 500),
        embedding=(emb_a + 0.05 * rnorm()).astype(np.float32),
        height=200.0,
        width=80.0,
        hat_color_hsv=(17.0, 195.0, 205.0),
    )
    new_at_predicted.embedding /= np.linalg.norm(new_at_predicted.embedding)

    match = pool.best_match(new_at_predicted, current_frame=130, verbose=True)
    assert match is not None, "predicted-position rebind should succeed"
    print(f"  → rebound to trk_id={match[0]} (score={match[1]:.3f})")
    print(f"    breakdown={ {k: round(v, 3) for k, v in match[2].items()} }")

    # Re-seed and add another ghost B somewhere else
    pool.add(ghost_a)
    emb_b = rnorm()
    pool.add(GhostTrack(
        trk_id=43, last_frame=120, last_bbox=(50, 50, 130, 250),
        last_velocity=(0.0, 0.0), embeddings=[emb_b],
        avg_height=200.0, avg_width=80.0,
        hat_color_hsv=(110.0, 200.0, 200.0),  # blue
    ))

    # New track appears far from both. Embedding doesn't match either.
    # Should NOT rebind — should return None.
    new_unrelated = NewTrackInfo(
        bbox=(1500, 700, 1580, 900),
        embedding=rnorm(),
        height=200.0, width=80.0,
        hat_color_hsv=(60.0, 200.0, 200.0),
    )
    match = pool.best_match(new_unrelated, current_frame=140, verbose=True)
    assert match is None, "unrelated new track must not rebind"
    print("  → correctly refused (no plausible ghost match)")

    # Embedding veto: new track appears AT ghost A's predicted location and
    # height but with a totally different embedding. Should be refused.
    new_imposter = NewTrackInfo(
        bbox=(550, 300, 630, 500),
        embedding=rnorm(),                   # nothing like emb_a
        height=200.0, width=80.0,
        hat_color_hsv=(15.0, 200.0, 200.0),  # even same hat color
    )
    match = pool.best_match(new_imposter, current_frame=141, verbose=True)
    assert match is None, "embedding veto should refuse imposter"
    print("  → correctly vetoed by embedding (different person at same spot)")

    # Exclusion: ghost A is technically a plausible match but excluded
    # because it's already bound to another live track this frame.
    new_at_predicted2 = NewTrackInfo(
        bbox=(550, 300, 630, 500),
        embedding=(emb_a + 0.05 * rnorm()).astype(np.float32),
        height=200.0, width=80.0,
    )
    new_at_predicted2.embedding /= np.linalg.norm(new_at_predicted2.embedding)
    match = pool.best_match(new_at_predicted2, current_frame=142,
                            exclude_trk_ids={42}, verbose=True)
    print(f"  → with excl={{42}}: match={match}")
    assert match is None, "excluded ghost must not match"

    # Cleanup: advance time beyond ttl
    pool.cleanup(current_frame=500)
    print(f"  → after cleanup at frame 500: {pool.summary()}")

    print("\nOK — ghost_pool.py self-test passed")
