"""Find close two-track interactions where identity may swap inside a tracklet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CrossingEvent:
    tid_a: int
    tid_b: int
    start: int
    end: int
    closest: int
    min_distance: float


def crossing_event_id(event: CrossingEvent) -> str:
    """Stable identifier used by review and resolution files."""
    return f"crossing:{event.tid_a}:{event.tid_b}:{event.start}"


def detect_crossing_events(traj, threshold=0.8, min_frames=3, merge_gap=12):
    tracks = {int(t): {int(r[0]): r for r in rows} for t, rows in traj.items()}
    events = []
    tids = sorted(tracks)
    for i, a in enumerate(tids):
        for b in tids[i + 1 :]:
            close = []
            for f in sorted(set(tracks[a]) & set(tracks[b])):
                x, y = tracks[a][f], tracks[b][f]
                ca = np.array(((x[1] + x[3]) / 2, (x[2] + x[4]) / 2), float)
                cb = np.array(((y[1] + y[3]) / 2, (y[2] + y[4]) / 2), float)
                h = max(((x[4] - x[2]) + (y[4] - y[2])) / 2, 1.0)
                d = float(np.linalg.norm(ca - cb) / h)
                if d < threshold:
                    close.append((f, d))
            groups = []
            for item in close:
                if not groups or item[0] > groups[-1][-1][0] + merge_gap + 1:
                    groups.append([item])
                else:
                    groups[-1].append(item)
            for g in groups:
                if len(g) >= min_frames:
                    closest = min(g, key=lambda z: z[1])
                    events.append(
                        CrossingEvent(a, b, g[0][0], g[-1][0], closest[0], closest[1])
                    )
    return sorted(events, key=lambda e: (e.start, e.tid_a, e.tid_b))


def candidate_split_frames(events):
    """Tracklet boundaries around interactions; assignment remains global."""
    out = {}
    for e in events:
        for tid in (e.tid_a, e.tid_b):
            out.setdefault(tid, set()).update((e.start, e.end + 1))
    return {tid: sorted(fs) for tid, fs in out.items()}


def unresolved_crossings(events, threshold=0.4):
    """Events where geometry locates ambiguity but cannot prove ownership."""
    return [
        {
            "event_id": crossing_event_id(e),
            "tid_a": e.tid_a,
            "tid_b": e.tid_b,
            "start": e.start,
            "end": e.end,
            "closest": e.closest,
            "min_distance": e.min_distance,
            "status": "unresolved",
            "hypotheses": [
                {"state": "same", "a": "incoming_a", "b": "incoming_b"},
                {"state": "swapped", "a": "incoming_b", "b": "incoming_a"},
                {"state": "duplicate_a", "a": "incoming_a", "b": "incoming_a"},
                {"state": "duplicate_b", "a": "incoming_b", "b": "incoming_b"},
                {"state": "missing", "a": None, "b": None},
            ],
        }
        for e in events
        if e.min_distance <= threshold
    ]
