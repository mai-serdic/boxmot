"""
Persistent K-best person gallery for cross-session re-identification.

Storage format (single .npz on disk):
  person_ids:  int64[N]      - which person each embedding row belongs to
  embeddings:  float32[N, D] - L2-normalized body-appearance embeddings
  qualities:   float32[N]    - score used to evict worst slot when at K capacity
  last_seen:   float64[N]    - unix timestamp of last update
  next_id:     int64[1]      - next person_id to hand out
Optional face-anchor arrays (absent in galleries created before face support):
  face_person_ids:  int64[M]
  face_embeddings:  float32[M, 512] - L2-normalized ArcFace, pose-gated only
  face_qualities:   float32[M]
  face_last_seen:   float64[M]
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
        k_faces_per_person: int = 5,
        face_match_threshold: float = 0.50,
    ):
        self.db_path = Path(db_path)
        self.k = k_per_person
        self.tau = match_threshold
        self.k_faces = k_faces_per_person
        # Strict on purpose: a face match is an identity *anchor* that can
        # override body appearance, so it must be merge-safe. ArcFace EER on
        # this footage sits at ~0.70; 0.50 trades recall for near-zero merges
        # (see bench/05_eval_face_anchor.py).
        self.face_tau = face_match_threshold

        self.person_ids = np.empty(0, dtype=np.int64)
        self.embeddings = np.empty((0, 0), dtype=np.float32)
        self.qualities = np.empty(0, dtype=np.float32)
        self.last_seen = np.empty(0, dtype=np.float64)
        self.face_person_ids = np.empty(0, dtype=np.int64)
        self.face_embeddings = np.empty((0, 0), dtype=np.float32)
        self.face_qualities = np.empty(0, dtype=np.float32)
        self.face_last_seen = np.empty(0, dtype=np.float64)
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
        if "face_person_ids" in d.files:  # galleries may predate face support
            self.face_person_ids = d["face_person_ids"].astype(np.int64)
            self.face_embeddings = d["face_embeddings"].astype(np.float32)
            self.face_qualities = d["face_qualities"].astype(np.float32)
            self.face_last_seen = d["face_last_seen"].astype(np.float64)
        n_people = len(np.unique(self.person_ids)) if len(self.person_ids) else 0
        print(
            f"[DB] Loaded {len(self.person_ids)} embeddings "
            f"(+{len(self.face_person_ids)} faces) across "
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
            face_person_ids=self.face_person_ids,
            face_embeddings=self.face_embeddings,
            face_qualities=self.face_qualities,
            face_last_seen=self.face_last_seen,
        )
        tmp.replace(self.db_path)
        n_people = len(np.unique(self.person_ids)) if len(self.person_ids) else 0
        print(
            f"[DB] Saved {len(self.person_ids)} embeddings "
            f"(+{len(self.face_person_ids)} faces) across "
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

    def query_face(
        self,
        emb: np.ndarray,
        exclude_pids: set[int] | None = None,
    ) -> tuple[int | None, float, float]:
        """Top-1 ArcFace cosine match across enrolled faces, mirroring
        :meth:`query` (per-person min distance, second-best for ratio tests).

        Returns (best_pid_if_match_else_None, best_dist, second_best_dist).
        """
        if self.face_embeddings.size == 0:
            return None, float("inf"), float("inf")
        excluded = set(int(p) for p in (exclude_pids or set()))
        dists = 1.0 - self.face_embeddings @ emb.astype(np.float32)
        per_person_min: list[tuple[float, int]] = []
        for pid in np.unique(self.face_person_ids):
            if int(pid) in excluded:
                continue
            mask = self.face_person_ids == pid
            per_person_min.append((float(dists[mask].min()), int(pid)))
        if not per_person_min:
            return None, float("inf"), float("inf")
        per_person_min.sort(key=lambda x: x[0])
        best_dist, best_pid = per_person_min[0]
        second_best = per_person_min[1][0] if len(per_person_min) >= 2 else float("inf")
        if best_dist < self.face_tau:
            return best_pid, best_dist, second_best
        return None, best_dist, second_best

    def add_face(self, person_id: int, emb: np.ndarray, quality: float) -> None:
        """Attach a pose-gated face embedding to a person; K-best eviction."""
        emb = emb.astype(np.float32).reshape(-1)
        idxs = np.where(self.face_person_ids == person_id)[0]
        if len(idxs) >= self.k_faces:
            worst_idx = int(idxs[np.argmin(self.face_qualities[idxs])])
            if quality <= self.face_qualities[worst_idx]:
                return
            self.face_embeddings[worst_idx] = emb
            self.face_qualities[worst_idx] = float(quality)
            self.face_last_seen[worst_idx] = time.time()
            return
        row = emb.reshape(1, -1)
        if self.face_embeddings.size == 0:
            self.face_embeddings = row.copy()
        else:
            self.face_embeddings = np.vstack([self.face_embeddings, row])
        self.face_person_ids = np.append(self.face_person_ids, np.int64(person_id))
        self.face_qualities = np.append(self.face_qualities, np.float32(quality))
        self.face_last_seen = np.append(self.face_last_seen, np.float64(time.time()))

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

    Face anchoring (confirm-only): when pose-gated face observations are
    supplied via ``resolve(..., face_obs=)``, a strict face match can
    (a) decide a pending track's identity ahead of body appearance, and
    (b) re-anchor an already-resolved track whose body match picked the wrong
    person. A face *mismatch* never blocks a body decision — at surveillance
    resolution an unmatched face is "no evidence", not "different person"
    (bench/05: overriding body with face verdicts makes fragmentation worse).
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
        face_ratio_threshold: float = 0.85,
        reanchor_min_frames: int = 15,
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

        # Faces mirror the body ratio test; re-anchoring additionally waits
        # until a track has held its identity reanchor_min_frames frames, so a
        # single early face frame can't thrash the mapping back and forth.
        self.face_ratio_threshold = face_ratio_threshold
        self.reanchor_min_frames = reanchor_min_frames

        self.trk_to_person: dict[int, int] = {}
        self.pending_buffers: dict[int, list[tuple[np.ndarray, float]]] = {}
        self.pending_faces: dict[int, list[tuple[np.ndarray, float]]] = {}
        self.last_update_frame: dict[int, int] = {}
        # trk_id → (best_dist_at_decision) for already-resolved tracks; used
        # to gate per-frame gallery updates with a tighter threshold.
        self.last_match_dist: dict[int, float] = {}
        # trk_id → frame at which the current person mapping was committed.
        self.decision_frame: dict[int, int] = {}

        self.n_enrolled = 0
        self.n_matched = 0
        self.n_refused_ambiguous = 0
        self.n_face_anchored = 0
        self.n_face_reanchored = 0

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
        face_obs=None,
    ):
        """
        Returns persistent person_id for this trk_id (or None while buffering).
        emb: 1-D embedding (will be L2-normalized here).

        ``exclude_pids`` are person_ids that must NOT be assigned to this
        trk_id at decision time — typically the set of persons already mapped
        to other still-active tracks in the current frame, to prevent two
        on-screen tracks from sharing the same Person_XXX label.

        ``face_obs`` is an optional pose-gated ``FaceObservation`` (from
        face_anchor.FaceAnchor.extract) for this track's crop this frame.
        """
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        emb /= np.linalg.norm(emb) + 1e-12

        if trk_id in self.trk_to_person:
            pid = self.trk_to_person[trk_id]
            pid = self._face_maintain(trk_id, pid, face_obs, frame_idx, exclude_pids)
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

        # Buffer gated faces even on frames whose body crop fails the quality
        # gate — a frontal face in a partially-occluded box is still evidence.
        if face_obs is not None:
            self.pending_faces.setdefault(trk_id, []).append(
                (face_obs.embedding, face_obs.quality)
            )

        if not self.is_quality_crop(bbox, conf, frame_shape):
            return None

        buf = self.pending_buffers.setdefault(trk_id, [])
        buf.append((emb, self.quality_score(bbox, conf)))

        if len(buf) >= self.decision_frames:
            embs = np.stack([e for e, _ in buf])
            centroid = embs.mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-12
            best_q = max(q for _, q in buf)

            # ── face-first (confirm-only): a strict, unambiguous face match
            # decides the identity ahead of body appearance. No face match →
            # fall through to the body path unchanged.
            face_pid = self._query_pending_faces(trk_id, exclude_pids)
            if face_pid is not None:
                self.n_face_anchored += 1
                self.db.update(face_pid, centroid, best_q)
                self._commit_faces(trk_id, face_pid)
                print(
                    f"  [DB] frame={frame_idx} trk_id={trk_id} → "
                    f"Person_{face_pid:03d}  (FACE anchor)"
                )
                self.trk_to_person[trk_id] = face_pid
                self.last_update_frame[trk_id] = frame_idx
                self.last_match_dist[trk_id] = 0.0
                self.decision_frame[trk_id] = frame_idx
                del self.pending_buffers[trk_id]
                return face_pid

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

            self._commit_faces(trk_id, pid)
            self.trk_to_person[trk_id] = pid
            self.last_update_frame[trk_id] = frame_idx
            self.last_match_dist[trk_id] = float(dist)
            self.decision_frame[trk_id] = frame_idx
            del self.pending_buffers[trk_id]
            return pid

        return None

    # ─── face internals ──────────────────────────────────────────────────────
    def _query_pending_faces(
        self, trk_id, exclude_pids: set[int] | None
    ) -> int | None:
        """Best strict-and-unambiguous face match over this track's buffered
        gated faces, or None. Queries with the highest-quality observation —
        face views vary too much (yaw sweeps) for a centroid to be meaningful.
        """
        faces = self.pending_faces.get(trk_id)
        if not faces:
            return None
        femb, _q = max(faces, key=lambda fq: fq[1])
        pid, dist, second = self.db.query_face(femb, exclude_pids=exclude_pids)
        if pid is None:
            return None
        if second != float("inf") and (dist / second) > self.face_ratio_threshold:
            return None
        return pid

    def _commit_faces(self, trk_id, pid: int) -> None:
        """Attach this track's buffered gated faces to its decided person."""
        for femb, fq in self.pending_faces.pop(trk_id, []):
            self.db.add_face(pid, femb, fq)

    def _face_maintain(
        self, trk_id, pid: int, face_obs, frame_idx, exclude_pids: set[int] | None
    ) -> int:
        """For an already-resolved track: enroll fresh faces on the mapped
        person, and re-anchor the mapping when an anchor-grade face strictly
        and unambiguously matches a *different* person (body appearance chose
        wrong — the exact failure mode faces exist to correct).
        """
        if face_obs is None:
            return pid
        fpid, dist, second = self.db.query_face(
            face_obs.embedding, exclude_pids=exclude_pids
        )
        unambiguous = second == float("inf") or (
            second > 0 and (dist / second) <= self.face_ratio_threshold
        )
        if fpid is None or fpid == pid:
            # No competing identity: this face refreshes (or seeds) the
            # mapped person's face bank. Skip ambiguous matches so a
            # look-alike's slot never gets polluted.
            if fpid == pid or (fpid is None and face_obs.is_strong):
                if unambiguous:
                    self.db.add_face(pid, face_obs.embedding, face_obs.quality)
            return pid
        held = frame_idx - self.decision_frame.get(trk_id, frame_idx)
        if (
            face_obs.is_strong
            and unambiguous
            and held >= self.reanchor_min_frames
        ):
            self.n_face_reanchored += 1
            print(
                f"  [DB] frame={frame_idx} trk_id={trk_id} → "
                f"Person_{fpid:03d}  (FACE re-anchor, was Person_{pid:03d}, "
                f"d={dist:.3f})"
            )
            self.trk_to_person[trk_id] = fpid
            self.decision_frame[trk_id] = frame_idx
            self.db.add_face(fpid, face_obs.embedding, face_obs.quality)
            return fpid
        return pid
