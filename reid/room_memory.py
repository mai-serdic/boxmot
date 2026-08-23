"""Persistent within-session room memory for fragmented tracklets.

This is an explicit, conservative association layer. It keeps path state
across BoT-SORT id changes and uses the existing metric reachability prior.
The room ledger first protects high-purity continuations, then
``stitch_memory_groups`` performs global revision over those memories. The raw
min-cost-flow result remains available as a rollback baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .trajectory_stitch import Tracklet, link_prob

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None
import numpy as np


class PathStatus(str, Enum):
    VISIBLE = "visible"
    MISSING = "missing"
    UNRESOLVED = "unresolved"
    EXITED = "exited"


@dataclass
class PathState:
    path_id: int
    tracklets: list[int] = field(default_factory=list)
    last: Tracklet | None = None
    reviewed_label: str | None = None
    occupied_intervals: list[tuple[int, int]] = field(default_factory=list)
    last_xy: tuple[float, float] | None = None
    velocity_xy: tuple[float, float] = (0.0, 0.0)
    status: PathStatus = PathStatus.UNRESOLVED


class RoomMemory:
    def __init__(
        self, reach, fps, params, *, min_link_prob=0.18, batch_window_frames=20
    ):
        self.reach, self.fps, self.params = reach, fps, params
        self.min_link_prob = min_link_prob
        self.batch_window_frames = batch_window_frames
        self.paths: dict[int, PathState] = {}
        self.next_id = 1
        self.verdicts: list[dict] = []

    def observe(self, tracklets: list[Tracklet], labels=None) -> dict[int, int]:
        """Assign each tracklet to persistent path memory.

        Candidate paths are evaluated against their latest endpoint. Matching
        is one-to-one within each birth-time batch; overlapping tracklets are
        never attached to the same path. ``labels`` optionally anchors a path
        to a reviewed person label for downstream diagnostics.
        """
        labels = labels or {}
        owner = {}
        ordered = sorted(tracklets, key=lambda x: (x.t0, x.tid))
        batches = []
        for t in ordered:
            if not batches or t.t0 - batches[-1][0].t0 > self.batch_window_frames:
                batches.append([t])
            else:
                batches[-1].append(t)
        for batch in batches:
            frame = batch[0].t0
            for state in self.paths.values():
                if state.last is not None and state.last.t1 < frame:
                    state.status = PathStatus.MISSING
            assignments = self._assign_batch(batch)
            for t in batch:
                chosen, best, second = assignments.get(t.tid, (None, 0.0, 0.0))
                if chosen is None:
                    chosen = self.next_id
                    self.next_id += 1
                    self.paths[chosen] = PathState(chosen)
                    confidence, reason = (
                        "algorithm-low",
                        "no jointly feasible prior path",
                    )
                elif best >= 0.60 and best - second >= 0.15:
                    confidence, reason = "algorithm-high", "unique joint continuation"
                else:
                    confidence, reason = (
                        "algorithm-uncertain",
                        "near-tied joint continuation",
                    )
                state = self.paths[chosen]
                state.tracklets.append(t.tid)
                state.last = t
                state.occupied_intervals.append((t.t0, t.t1))
                xy, vel = t.end_state(self.fps)
                state.last_xy = xy
                state.velocity_xy = vel
                state.status = PathStatus.VISIBLE
                state.reviewed_label = labels.get(t.tid, state.reviewed_label)
                owner[t.tid] = chosen
                self.verdicts.append(
                    {
                        "tid": t.tid,
                        "path_id": chosen,
                        "confidence": confidence,
                        "reason": reason,
                        "best_probability": round(float(best), 6),
                        "runner_up_probability": round(float(second), 6),
                    }
                )
        return owner

    def snapshot(self):
        return {
            pid: {
                "path_id": pid,
                "tracklets": list(s.tracklets),
                "occupied_intervals": list(s.occupied_intervals),
                "last_frame": s.last.t1 if s.last is not None else None,
                "last_xy": s.last_xy,
                "velocity_xy": s.velocity_xy,
                "status": s.status.value,
                "reviewed_label": s.reviewed_label,
            }
            for pid, s in self.paths.items()
        }

    def _assign_batch(self, batch):
        pids = sorted(self.paths)
        if not pids:
            return {}
        P = np.zeros((len(pids), len(batch)), float)
        for i, pid in enumerate(pids):
            state = self.paths[pid]
            for j, t in enumerate(batch):
                if state.last is not None and not self._overlaps(state, t):
                    P[i, j] = link_prob(
                        self.reach, state.last, t, self.fps, self.params
                    )
        out = {}
        if linear_sum_assignment is not None and P.size:
            ri, ci = linear_sum_assignment(-P)
            for i, j in zip(ri, ci):
                if P[i, j] >= self.min_link_prob:
                    out[batch[j].tid] = (
                        pids[i],
                        P[i, j],
                        max((P[k, j] for k in range(len(pids)) if k != i), default=0.0),
                    )
        else:  # deterministic fallback
            used = set()
            for j, t in enumerate(batch):
                cand = sorted(
                    ((P[i, j], pids[i]) for i in range(len(pids))), reverse=True
                )
                for prob, pid in cand:
                    if prob >= self.min_link_prob and pid not in used:
                        out[t.tid] = (pid, prob, cand[1][0] if len(cand) > 1 else 0.0)
                        used.add(pid)
                        break
        return out

    @staticmethod
    def _overlaps(state: PathState, t: Tracklet) -> bool:
        a = state.last
        return a is not None and t.t0 <= a.t1 and a.t0 <= t.t1


def groups_from_memory(owner):
    out = {}
    for tid, pid in owner.items():
        out.setdefault(pid, []).append(tid)
    return out


def stitch_memory_groups(reach, tracklets, memory_owner, fps, params, birth_cost=6.0):
    """Global revision over high-purity path-memory groups."""
    from .trajectory_stitch import Tracklet, compact_path_ids, stitch_global

    grouped = {}
    for t in tracklets:
        grouped.setdefault(memory_owner[t.tid], []).append(t)
    supers = []
    for pid, ts in grouped.items():
        frames = np.concatenate([t.frames for t in ts])
        xy = np.concatenate([t.xy for t in ts])
        order = np.argsort(frames)
        supers.append(Tracklet(pid, frames[order], xy[order]))
    root = compact_path_ids(
        stitch_global(reach, supers, fps, params, birth_cost=birth_cost)
    )
    return {t.tid: root[memory_owner[t.tid]] for t in tracklets}
