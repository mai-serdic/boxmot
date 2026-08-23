"""Split tracker IDs into temporal segments while preserving provenance."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    sid: int
    source_tid: int
    rows: tuple

    @property
    def t0(self):
        return int(self.rows[0][0])

    @property
    def t1(self):
        return int(self.rows[-1][0])


def split_trajectory(traj, boundaries=None):
    """Split before each boundary frame; no observation is dropped or copied."""
    boundaries = {
        int(k): sorted(set(map(int, v))) for k, v in (boundaries or {}).items()
    }
    result = []
    provenance = {}
    sid = 1
    for tid_s, rows in sorted(traj.items(), key=lambda kv: int(kv[0])):
        tid = int(tid_s)
        cuts = boundaries.get(tid, [])
        chunks = []
        current = []
        for row in sorted(rows, key=lambda r: int(r[0])):
            if current and int(row[0]) in cuts:
                chunks.append(current)
                current = []
            current.append(tuple(row))
        if current:
            chunks.append(current)
        for chunk in chunks:
            seg = Segment(sid, tid, tuple(chunk))
            result.append(seg)
            for row in chunk:
                provenance[(tid, int(row[0]))] = sid
            sid += 1
    return result, provenance


def segments_to_traj(segments):
    return {str(s.sid): [list(r) for r in s.rows] for s in segments}
