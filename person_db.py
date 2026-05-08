"""
Persistent K-best person gallery for cross-session re-identification.

Storage format (single .npz on disk):
  person_ids:  int64[N]      - which person each embedding row belongs to
  embeddings:  float32[N, D] - L2-normalized
  qualities:   float32[N]    - score used to evict worst slot when at K capacity
  last_seen:   float64[N]    - unix timestamp of last update
  next_id:     int64[1]      - next person_id to hand out
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


class PersonDB:
    def __init__(
        self,
        db_path: str | Path,
        k_per_person: int = 10,
        match_threshold: float = 0.35,
    ):
        self.db_path = Path(db_path)
        self.k = k_per_person
        self.tau = match_threshold

        self.person_ids = np.empty(0, dtype=np.int64)
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.qualities = np.empty(0, dtype=np.float32)
        self.last_seen = np.empty(0, dtype=np.float64)
        self.next_id = 1
        self.load()

    # ─── persistence ─────────────────────────────────────────────────────────
    def load(self) -> None:
        if not self.db_path.exists():
            return
        d = np.load(self.db_path, allow_pickle=False)
        self.person_ids = d["person_ids"].astype(np.int64)
        self.embeddings = d["embeddings"].astype(np.float32)
        self.qualities = d["qualities"].astype(np.float32)
        self.last_seen = d["last_seen"].astype(np.float64)
        self.next_id = int(d["next_id"][0])
        n_people = len(np.unique(self.person_ids)) if len(self.person_ids) else 0
        print(
            f"[DB] Loaded {len(self.person_ids)} embeddings across "
            f"{n_people} persons from {self.db_path}"
        )

    def save(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.db_path.with_suffix(".tmp.npz")
        np.savez(
            tmp,
            person_ids=self.person_ids,
            embeddings=self.embeddings,
            qualities=self.qualities,
            last_seen=self.last_seen,
            next_id=np.array([self.next_id], dtype=np.int64),
        )
        tmp.replace(self.db_path)
        n_people = len(np.unique(self.person_ids)) if len(self.person_ids) else 0
        print(
            f"[DB] Saved {len(self.person_ids)} embeddings across "
            f"{n_people} persons → {self.db_path}"
        )

    # ─── core ops ────────────────────────────────────────────────────────────
    def query(
        self,
        emb: np.ndarray,
        exclude_pids: set[int] | None = None,
    ) -> tuple[int | None, float, float]:
        """
        Top-1 cosine match across the full gallery, with second-best distance
        for a Lowe-style ratio test by callers.

        ``exclude_pids`` removes those person_ids from consideration entirely —
        used to enforce the "no two tracks share an identity in the same frame"
        constraint.

        Returns: (best_pid_if_match_else_None, best_dist, second_best_dist).
        ``second_best_dist`` is +inf if fewer than two persons remain after
        exclusion.
        """
        if self.embeddings.size == 0:
            return None, float("inf"), float("inf")
        excluded = set(int(p) for p in (exclude_pids or set()))
        sims = self.embeddings @ emb.astype(np.float32)
        dists = 1.0 - sims
        per_person_min: list[tuple[float, int]] = []
        for pid in np.unique(self.person_ids):
            if int(pid) in excluded:
                continue
            mask = self.person_ids == pid
            per_person_min.append((float(dists[mask].min()), int(pid)))
        if not per_person_min:
            return None, float("inf"), float("inf")
        per_person_min.sort(key=lambda x: x[0])
        best_dist, best_pid = per_person_min[0]
        second_best = per_person_min[1][0] if len(per_person_min) >= 2 else float("inf")
        if best_dist < self.tau:
            return best_pid, best_dist, second_best
        return None, best_dist, second_best

    def enroll(self, emb: np.ndarray, quality: float) -> int:
        """Create a new person from this embedding. Returns new person_id."""
        pid = self.next_id
        self.next_id += 1
        self._add_row(pid, emb, quality)
        return pid

    def update(self, person_id: int, emb: np.ndarray, quality: float) -> None:
        """Append to a known person; evict worst-quality slot if at K capacity."""
        idxs = np.where(self.person_ids == person_id)[0]
        if len(idxs) < self.k:
            self._add_row(person_id, emb, quality)
            return
        worst_idx = int(idxs[np.argmin(self.qualities[idxs])])
        if quality > self.qualities[worst_idx]:
            self.embeddings[worst_idx] = emb.astype(np.float32)
            self.qualities[worst_idx] = float(quality)
            self.last_seen[worst_idx] = time.time()

    # ─── internals ───────────────────────────────────────────────────────────
    def _add_row(self, pid: int, emb: np.ndarray, quality: float) -> None:
        D = emb.shape[0]
        emb = emb.astype(np.float32).reshape(1, D)
        if self.embeddings.size == 0:
            self.embeddings = emb.copy()
        else:
            self.embeddings = np.vstack([self.embeddings, emb])
        self.person_ids = np.append(self.person_ids, np.int64(pid))
        self.qualities = np.append(self.qualities, np.float32(quality))
        self.last_seen = np.append(self.last_seen, np.float64(time.time()))

    # ─── stats ───────────────────────────────────────────────────────────────
    @property
    def n_persons(self) -> int:
        return int(len(np.unique(self.person_ids))) if len(self.person_ids) else 0

    @property
    def n_embeddings(self) -> int:
        return int(len(self.person_ids))


class IdentityResolver:
    """
    Maps a tracker's short-lived `trk_id` to a persistent `person_id` via
    embedding-based query into a PersonDB. Buffers the first few high-quality
    embeddings of each new trk_id before committing a decision.
    """

    def __init__(
        self,
        db: PersonDB,
        decision_frames: int = 8,
        quality_min_area: int = 80 * 160,
        quality_min_conf: float = 0.7,
        update_every: int = 5,
        edge_margin: int = 5,
        ratio_threshold: float = 0.85,
        gallery_update_max_dist: float = 0.20,
    ):
        self.db = db
        self.decision_frames = decision_frames
        self.q_area = quality_min_area
        self.q_conf = quality_min_conf
        self.update_every = update_every
        self.edge_margin = edge_margin
        # Lowe-style ratio: require best match to be ratio_threshold * second-best
        # to avoid false-merging when two enrolled persons look similarly close.
        self.ratio_threshold = ratio_threshold
        # Only update an existing person's gallery when the match is *very*
        # confident, so uncertain matches don't drift the centroid.
        self.gallery_update_max_dist = gallery_update_max_dist

        self.trk_to_person: dict[int, int] = {}
        self.pending_buffers: dict[int, list[tuple[np.ndarray, float]]] = {}
        self.last_update_frame: dict[int, int] = {}
        # trk_id → (best_dist_at_decision) for already-resolved tracks; used
        # to gate per-frame gallery updates with a tighter threshold.
        self.last_match_dist: dict[int, float] = {}

        self.n_enrolled = 0
        self.n_matched = 0
        self.n_refused_ambiguous = 0

    # ─── quality gate ────────────────────────────────────────────────────────
    def is_quality_crop(self, bbox, conf, frame_shape) -> bool:
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        if w * h < self.q_area:
            return False
        if conf < self.q_conf:
            return False
        if h <= w * 1.3:  # human aspect ratio sanity
            return False
        H, W = frame_shape[:2]
        m = self.edge_margin
        if x1 < m or y1 < m or x2 > W - m or y2 > H - m:
            return False
        return True

    @staticmethod
    def quality_score(bbox, conf) -> float:
        x1, y1, x2, y2 = bbox
        return float((x2 - x1) * (y2 - y1)) * float(conf)

    # ─── main API ────────────────────────────────────────────────────────────
    def resolve(
        self,
        trk_id,
        emb,
        bbox,
        conf,
        frame_idx,
        frame_shape,
        exclude_pids: set[int] | None = None,
    ):
        """
        Returns persistent person_id for this trk_id (or None while buffering).
        emb: 1-D embedding (will be L2-normalized here).

        ``exclude_pids`` are person_ids that must NOT be assigned to this
        trk_id at decision time — typically the set of persons already mapped
        to other still-active tracks in the current frame, to prevent two
        on-screen tracks from sharing the same Person_XXX label.
        """
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        emb /= np.linalg.norm(emb) + 1e-12

        if trk_id in self.trk_to_person:
            pid = self.trk_to_person[trk_id]
            # Per-frame gallery refinement, tightly gated:
            #   - quality crop only
            #   - throttle to once every `update_every` frames
            #   - only if the *current* embedding stays close to this person
            #     (don't pollute the gallery with drifting features)
            if (
                self.is_quality_crop(bbox, conf, frame_shape)
                and frame_idx - self.last_update_frame.get(trk_id, -10**9) >= self.update_every
            ):
                idxs = np.where(self.db.person_ids == pid)[0]
                if len(idxs):
                    cur_dist = float((1.0 - self.db.embeddings[idxs] @ emb).min())
                    if cur_dist <= self.gallery_update_max_dist:
                        self.db.update(pid, emb, self.quality_score(bbox, conf))
                        self.last_update_frame[trk_id] = frame_idx
            return pid

        if not self.is_quality_crop(bbox, conf, frame_shape):
            return None

        buf = self.pending_buffers.setdefault(trk_id, [])
        buf.append((emb, self.quality_score(bbox, conf)))

        if len(buf) >= self.decision_frames:
            embs = np.stack([e for e, _ in buf])
            centroid = embs.mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-12
            best_q = max(q for _, q in buf)

            pid, dist, second_dist = self.db.query(centroid, exclude_pids=exclude_pids)

            # Lowe-style ratio test: even if best is below tau, if the gap
            # between best and second-best is too small the match is ambiguous.
            ambiguous = (
                pid is not None
                and second_dist != float("inf")
                and (dist / second_dist) > self.ratio_threshold
            )
            if ambiguous:
                pid_old = pid
                pid = None  # force enrollment as a new person
                self.n_refused_ambiguous += 1

            excluded_count = len(exclude_pids) if exclude_pids else 0
            if pid is None:
                pid = self.db.enroll(centroid, best_q)
                self.n_enrolled += 1
                if ambiguous:
                    action = (
                        f"enroll (ambiguous, best→P_{pid_old:03d} d={dist:.3f}, "
                        f"2nd d={second_dist:.3f}, ratio={dist / second_dist:.2f})"
                    )
                else:
                    action = f"enroll (new, nearest d={dist:.3f})"
                if excluded_count:
                    action += f" [excl={excluded_count}]"
            else:
                self.db.update(pid, centroid, best_q)
                self.n_matched += 1
                action = (
                    f"match d={dist:.3f}"
                    if second_dist == float("inf")
                    else f"match d={dist:.3f} (2nd d={second_dist:.3f})"
                )
            print(
                f"  [DB] frame={frame_idx} trk_id={trk_id} → Person_{pid:03d}  "
                f"({action})"
            )

            self.trk_to_person[trk_id] = pid
            self.last_update_frame[trk_id] = frame_idx
            self.last_match_dist[trk_id] = float(dist)
            del self.pending_buffers[trk_id]
            return pid

        return None
