"""Validated manual path labels and safe interval propagation."""

from __future__ import annotations

from dataclasses import dataclass

VALID = {"1", "2", "unknown"}


@dataclass(frozen=True)
class Label:
    tid: int
    frame: int
    value: str
    source: str = "reviewed"


def normalise_edits(edits) -> list[Label]:
    out = []
    for e in edits or []:
        tid, frame, value = int(e["tid"]), int(e["frame"]), str(e["label"])
        if value not in VALID:
            raise ValueError(f"invalid label {value!r}")
        out.append(Label(tid, frame, value, str(e.get("source", "reviewed"))))
    return sorted(out, key=lambda x: (x.tid, x.frame))


def propagate_tracklet(
    rows, edits, *, tid=None, initial=None, allow_unknown=True
) -> list[dict]:
    """Expand reviewed change points only across the same tracklet.

    A later explicit review wins from its frame onward. No label crosses a
    tracklet boundary; connecting two tracklets requires a separate reviewed
    handover edge.
    """
    points = {}
    for e in normalise_edits(edits):
        points[e.frame] = e
    if not points and initial is None:
        return []
    ordered = sorted(points.items())
    result = []
    for row in rows:
        f = int(row[0])
        prior = [e for pf, e in ordered if pf <= f]
        if prior:
            e = prior[-1]
        elif initial is not None:
            e = Label(int(tid), f, str(initial), "algorithm")
        else:
            e = ordered[0][1]
        if not allow_unknown and e.value == "unknown":
            continue
        result.append(
            {
                "tid": int(tid if tid is not None else e.tid),
                "frame": f,
                "label": e.value,
                "source": "reviewed"
                if f in points
                else ("propagated" if prior else "algorithm"),
            }
        )
    return result


def validate_exclusivity(labels: list[dict]) -> list[tuple[int, str, list[int]]]:
    """Return same-frame duplicate-person conflicts, without guessing fixes."""
    by = {}
    for e in labels:
        if e["label"] in ("1", "2"):
            by.setdefault((int(e["frame"]), e["label"]), set()).add(int(e["tid"]))
    return [(f, lab, sorted(tids)) for (f, lab), tids in by.items() if len(tids) > 1]


def duplicate_groups(labels: list[dict]) -> list[dict]:
    """Same-person boxes in one frame, retained as explicit tracker duplicates."""
    return [
        {"frame": f, "person": lab, "tids": tids}
        for f, lab, tids in validate_exclusivity(labels)
    ]
