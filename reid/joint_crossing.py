"""Joint state space for two paths through a close-contact event."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .crossing_events import crossing_event_id

AUTHORITATIVE_RESOLUTION_SOURCES = {"badge", "uwb", "turnstile", "operator"}


def _track_rows(traj, tid):
    return traj.get(str(tid), traj.get(int(tid), []))


def trajectory_fingerprint(traj):
    """Stable binding for track-ID-specific operator corrections."""
    normalized = {
        str(int(tid)): sorted((list(row) for row in rows), key=lambda row: row[0])
        for tid, rows in traj.items()
    }
    raw = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


class JointState(str, Enum):
    SAME = "same"  # incoming A owns tid A; incoming B owns tid B
    SWAPPED = "swapped"  # incoming A owns tid B; incoming B owns tid A
    DUPLICATE_A = "duplicate_a"  # both boxes may depict incoming A
    DUPLICATE_B = "duplicate_b"  # both boxes may depict incoming B
    MISSING_A = "missing_a"
    MISSING_B = "missing_b"


@dataclass(frozen=True)
class JointHypothesis:
    frame: int
    states: tuple[JointState, ...]


def hypotheses_for_event(event, traj):
    """Enumerate legal states per frame; scoring is supplied by the optimizer."""
    ta = {
        int(r[0])
        for r in _track_rows(traj, event.tid_a)
        if event.start <= int(r[0]) <= event.end
    }
    tb = {
        int(r[0])
        for r in _track_rows(traj, event.tid_b)
        if event.start <= int(r[0]) <= event.end
    }
    out = []
    for f in range(event.start, event.end + 1):
        a, b = f in ta, f in tb
        if a and b:
            states = (
                JointState.SAME,
                JointState.SWAPPED,
                JointState.DUPLICATE_A,
                JointState.DUPLICATE_B,
            )
        elif a:
            states = (JointState.SAME, JointState.SWAPPED, JointState.MISSING_B)
        elif b:
            states = (JointState.SAME, JointState.SWAPPED, JointState.MISSING_A)
        else:
            states = (JointState.MISSING_A, JointState.MISSING_B)
        out.append(JointHypothesis(f, states))
    return out


def transition_cost(previous, current, swap_cost=2.0, duplicate_cost=1.0):
    """Regularise state sequences without forbidding temporary contamination."""
    if previous == current:
        return 0.0
    if current in (
        JointState.DUPLICATE_A,
        JointState.DUPLICATE_B,
        JointState.MISSING_A,
        JointState.MISSING_B,
    ):
        return duplicate_cost
    return swap_cost


def _observations(event, traj):
    tracks = {int(t): {int(r[0]): r for r in rows} for t, rows in traj.items()}
    out = {}
    for f in range(event.start, event.end + 1):
        boxes = {}
        for key, tid in (("a", event.tid_a), ("b", event.tid_b)):
            r = tracks.get(tid, {}).get(f)
            if r is not None:
                boxes[key] = (
                    np.asarray(((r[1] + r[3]) / 2, (r[2] + r[4]) / 2), float),
                    max(float(r[4] - r[2]), 1.0),
                )
        if boxes:
            out[f] = boxes
    return out


def _assignment(state, boxes):
    if state == JointState.SAME:
        return {
            0: [boxes["a"]] if "a" in boxes else [],
            1: [boxes["b"]] if "b" in boxes else [],
        }
    if state == JointState.SWAPPED:
        return {
            0: [boxes["b"]] if "b" in boxes else [],
            1: [boxes["a"]] if "a" in boxes else [],
        }
    if state == JointState.DUPLICATE_A:
        return {0: list(boxes.values()), 1: []}
    if state == JointState.DUPLICATE_B:
        return {0: [], 1: list(boxes.values())}
    return {0: [], 1: []}


def resolve_event_motion(
    event, traj, *, switch_penalty=0.15, duplicate_penalty=0.45, missing_penalty=0.35
):
    """Viterbi resolution from box motion only; appearance contributes zero.

    Returns frame -> joint state. It intentionally permits temporary duplicate
    states and a return to SAME after an occlusion.
    """
    obs = _observations(event, traj)
    if not obs:
        return {}
    frames = sorted(obs)
    states = (
        JointState.SAME,
        JointState.SWAPPED,
        JointState.DUPLICATE_A,
        JointState.DUPLICATE_B,
    )
    cost = {
        JointState.SAME: 0.0,
        JointState.SWAPPED: switch_penalty,
        JointState.DUPLICATE_A: duplicate_penalty,
        JointState.DUPLICATE_B: duplicate_penalty,
    }
    back = []
    for i in range(1, len(frames)):
        f0, f1 = frames[i - 1], frames[i]
        dt = max(f1 - f0, 1)
        nxt = {}
        bp = {}
        for cur in states:
            ac = _assignment(cur, obs[f1])
            best = (float("inf"), None)
            for prev in states:
                ap = _assignment(prev, obs[f0])
                c = cost[prev]
                c += 0.0 if prev == cur else switch_penalty
                if cur in (JointState.DUPLICATE_A, JointState.DUPLICATE_B):
                    c += duplicate_penalty
                for ident in (0, 1):
                    if ap[ident] and ac[ident]:
                        c += min(
                            float(
                                np.linalg.norm(x[0] - y[0])
                                / max((x[1] + y[1]) / 2, 1.0)
                                / dt
                            )
                            for x in ap[ident]
                            for y in ac[ident]
                        )
                    elif ap[ident] or ac[ident]:
                        c += missing_penalty
                if c < best[0]:
                    best = (c, prev)
            nxt[cur] = best[0]
            bp[cur] = best[1]
        cost = nxt
        back.append(bp)
    state = min(cost, key=cost.get)
    seq = [state]
    for bp in reversed(back):
        state = bp[state]
        seq.append(state)
    seq.reverse()
    return dict(zip(frames, seq))


def resolve_from_anchor(incoming_a, incoming_b, observed_a, observed_b):
    """Collapse a crossing after authoritative downstream identity evidence."""
    if observed_a == incoming_a and observed_b == incoming_b:
        return JointState.SAME
    if observed_a == incoming_b and observed_b == incoming_a:
        return JointState.SWAPPED
    if observed_a == observed_b == incoming_a:
        return JointState.DUPLICATE_A
    if observed_a == observed_b == incoming_b:
        return JointState.DUPLICATE_B
    return None


def states_from_reviewed_labels(event, traj, labels):
    """Derive reviewed joint states and tail persistence for one event.

    ``labels`` maps ``(tid, frame)`` to an operator-reviewed person label. The
    incoming identities come from the latest reviewed observations before the
    event. A tracker ID born inside the event may instead use its first reviewed
    in-event observation; that is direct operator evidence, not propagation.
    """
    frames_by_tid = {
        tid: {int(row[0]) for row in _track_rows(traj, tid)}
        for tid in (event.tid_a, event.tid_b)
    }

    def incoming_label(tid):
        candidates = [
            (frame, value)
            for (label_tid, frame), value in labels.items()
            if label_tid == tid and frame < event.start
        ]
        if candidates:
            return max(candidates)[1]
        first = sorted(
            (frame, value)
            for (label_tid, frame), value in labels.items()
            if label_tid == tid and event.start <= frame <= event.end
        )
        return first[0][1] if first else None

    incoming_a = incoming_label(event.tid_a)
    incoming_b = incoming_label(event.tid_b)
    if incoming_a is None or incoming_b is None or incoming_a == incoming_b:
        return {}, False, False, None

    states = {}
    observed_frames = sorted(frames_by_tid[event.tid_a] | frames_by_tid[event.tid_b])
    observed_frames = [f for f in observed_frames if event.start <= f <= event.end]
    for frame in observed_frames:
        has_a = frame in frames_by_tid[event.tid_a]
        has_b = frame in frames_by_tid[event.tid_b]
        label_a = labels.get((event.tid_a, frame)) if has_a else None
        label_b = labels.get((event.tid_b, frame)) if has_b else None
        state = None
        if has_a and has_b and label_a is not None and label_b is not None:
            state = resolve_from_anchor(incoming_a, incoming_b, label_a, label_b)
        elif has_a and label_a is not None:
            if label_a == incoming_a:
                state = JointState.MISSING_B
            elif label_a == incoming_b:
                state = JointState.SWAPPED
        elif has_b and label_b is not None:
            if label_b == incoming_b:
                state = JointState.MISSING_A
            elif label_b == incoming_a:
                state = JointState.SWAPPED
        if state is not None:
            states[frame] = state

    complete = bool(observed_frames) and set(states) == set(observed_frames)
    changes = {}
    previous = None
    for frame, state in sorted(states.items()):
        if state != previous:
            changes[frame] = state
            previous = state

    carry = False
    if states and states[max(states)] == JointState.SWAPPED:
        expected = {event.tid_a: incoming_b, event.tid_b: incoming_a}
        tail = [
            ((tid, frame), value)
            for (tid, frame), value in labels.items()
            if tid in expected and frame > event.end
        ]
        carry = bool(tail) and all(value == expected[tid] for (tid, _), value in tail)
    return changes, complete, carry, (incoming_a, incoming_b)


def crossing_path_overrides(
    event, traj, owner, states, *, carry_tail=False, incoming_paths=None
):
    """Convert a resolved joint-state sequence into frame-level path owners.

    Temporary duplicate/swap states affect only their explicit frames and may
    return to SAME without rewriting the remainder. Tail rewriting is explicit
    because a state observed at the end of a partial review is not proof that
    the swap persisted.
    """
    path_a, path_b = incoming_paths or (
        owner[event.tid_a],
        owner[event.tid_b],
    )
    frames_by_tid = {
        tid: {int(r[0]) for r in _track_rows(traj, tid)}
        for tid in (event.tid_a, event.tid_b)
    }
    overrides = {}
    for frame, state in sorted(states.items()):
        if state == JointState.SAME:
            pair = (path_a, path_b)
        elif state == JointState.SWAPPED:
            pair = (path_b, path_a)
        elif state == JointState.DUPLICATE_A:
            pair = (path_a, path_a)
        elif state == JointState.DUPLICATE_B:
            pair = (path_b, path_b)
        elif state == JointState.MISSING_A:
            pair = (None, path_b)
        elif state == JointState.MISSING_B:
            pair = (path_a, None)
        else:
            continue
        for tid, path in zip((event.tid_a, event.tid_b), pair):
            if path is not None and frame in frames_by_tid[tid]:
                overrides[(tid, frame)] = path
    final_state = states[max(states)] if states else None
    if carry_tail:
        if final_state != JointState.SWAPPED:
            raise ValueError("carry_tail requires the final state to be 'swapped'")
        for tid, path in ((event.tid_a, path_b), (event.tid_b, path_a)):
            for frame in frames_by_tid[tid]:
                if frame > event.end:
                    overrides[(tid, frame)] = path
    return overrides


def _expand_state_changes(event, changes):
    """Expand sparse state change points over an event, leaving its prefix open."""
    parsed = sorted((int(frame), JointState(value)) for frame, value in changes.items())
    for frame, _state in parsed:
        if not event.start <= frame <= event.end:
            raise ValueError(
                f"state frame {frame} is outside event {event.start}-{event.end}"
            )
    expanded = {}
    for frame in range(event.start, event.end + 1):
        prior = [state for at, state in parsed if at <= frame]
        if prior:
            expanded[frame] = prior[-1]
    return expanded


def apply_crossing_resolutions(
    events, traj, owner, document, *, include_bindings=False
):
    """Validate authoritative crossing decisions and create frame overrides.

    ``document`` is either a list or ``{"crossing_resolutions": [...]}``.
    Each entry names ``event_id``, an authoritative ``source``, and sparse
    ``states`` change points. Operator entries must be explicitly reviewed.
    ``complete`` closes the event only when states start at the event boundary;
    ``carry_forward`` explicitly persists a final swap beyond the event.
    """
    entries = (
        document.get("crossing_resolutions", [])
        if isinstance(document, dict)
        else document
    )
    if not isinstance(entries, list):
        raise TypeError("crossing_resolutions must be a list")
    expected_fingerprint = (
        document.get("trajectory_fingerprint") if isinstance(document, dict) else None
    )
    if expected_fingerprint and expected_fingerprint != trajectory_fingerprint(traj):
        raise ValueError("crossing resolutions belong to a different trajectory dump")
    by_id = {crossing_event_id(event): event for event in events}
    entries_by_event = {}
    for entry in entries:
        event_id = str(entry.get("event_id", ""))
        if event_id in entries_by_event:
            raise ValueError(f"duplicate crossing resolution {event_id!r}")
        entries_by_event[event_id] = entry
    unknown = sorted(set(entries_by_event) - set(by_id))
    if unknown:
        raise ValueError(f"unknown crossing events: {unknown}")
    ordered_entries = sorted(
        entries, key=lambda entry: by_id[str(entry["event_id"])].start
    )
    tracks = {
        int(tid): sorted(int(row[0]) for row in rows) for tid, rows in traj.items()
    }
    overrides = {}
    override_event = {}
    label_paths = {}
    applied = []
    resolved = set()
    for entry in ordered_entries:
        event_id = str(entry.get("event_id", ""))
        source = str(entry.get("source", ""))
        if source not in AUTHORITATIVE_RESOLUTION_SOURCES:
            raise ValueError(f"non-authoritative crossing source {source!r}")
        if source == "operator" and entry.get("reviewed") is not True:
            raise ValueError("operator crossing resolutions require reviewed=true")
        changes = entry.get("states")
        if not isinstance(changes, dict) or not changes:
            raise ValueError(f"{event_id}: states must be a non-empty object")
        event = by_id[event_id]
        states = _expand_state_changes(event, changes)
        complete = bool(entry.get("complete", False))
        if complete and event.start not in states:
            raise ValueError(
                f"{event_id}: complete=true requires a state at the event start"
            )
        carry = bool(entry.get("carry_forward", False))

        event_start = event.start

        def incoming_path(tid, start=event_start):
            prior = [frame for frame in tracks.get(tid, []) if frame < start]
            for frame in reversed(prior):
                if (tid, frame) in overrides:
                    return overrides[(tid, frame)]
            return owner[tid]

        incoming_paths = (incoming_path(event.tid_a), incoming_path(event.tid_b))
        incoming_labels = entry.get("incoming_labels")
        if incoming_labels is not None:
            if not isinstance(incoming_labels, dict):
                raise ValueError(f"{event_id}: incoming_labels must be an object")
            labels = (
                str(incoming_labels.get("a", "")),
                str(incoming_labels.get("b", "")),
            )
            if not all(labels) or labels[0] == labels[1]:
                raise ValueError(f"{event_id}: incoming labels must be distinct")
            paths = list(incoming_paths)
            for index, label in enumerate(labels):
                if label in label_paths:
                    paths[index] = label_paths[label]
            for label, path in zip(labels, paths):
                occupied_by = next(
                    (
                        known
                        for known, known_path in label_paths.items()
                        if known_path == path
                    ),
                    None,
                )
                if occupied_by is None or occupied_by == label:
                    label_paths[label] = path
            incoming_paths = tuple(paths)
        current = crossing_path_overrides(
            event,
            traj,
            owner,
            states,
            carry_tail=carry,
            incoming_paths=incoming_paths,
        )
        conflict = {
            key: (overrides[key], value)
            for key, value in current.items()
            if key in overrides
            and overrides[key] != value
            and by_id[override_event[key]].end >= event.start
        }
        if conflict:
            raise ValueError(f"conflicting crossing resolutions: {conflict}")
        overrides.update(current)
        override_event.update({key: event_id for key in current})
        status = "resolved" if complete else "partial"
        if complete:
            resolved.add(event_id)
        applied.append(
            {
                "event_id": event_id,
                "source": source,
                "reviewed": bool(entry.get("reviewed", False)),
                "status": status,
                "carry_forward": carry,
                "state_changes": {
                    str(frame): state.value
                    for frame, state in sorted(
                        (int(frame), JointState(value))
                        for frame, value in changes.items()
                    )
                },
                "overridden_observations": len(current),
                "incoming_labels": incoming_labels,
            }
        )
    anchors = document.get("tracklet_anchors", []) if isinstance(document, dict) else []
    if not isinstance(anchors, list):
        raise TypeError("tracklet_anchors must be a list")
    for anchor in anchors:
        tid = int(anchor["tid"])
        label = str(anchor["label"])
        source = str(anchor.get("source", ""))
        if source not in AUTHORITATIVE_RESOLUTION_SOURCES:
            raise ValueError(f"non-authoritative tracklet anchor source {source!r}")
        if source == "operator" and anchor.get("reviewed") is not True:
            raise ValueError("operator tracklet anchors require reviewed=true")
        if tid not in tracks:
            raise ValueError(f"tracklet anchor references unknown tid {tid}")
        if label not in label_paths:
            raise ValueError(f"tracklet anchor label {label!r} has no path binding")
        path = label_paths[label]
        for frame in tracks[tid]:
            overrides[(tid, frame)] = path
        applied.append(
            {
                "kind": "tracklet_anchor",
                "tid": tid,
                "label": label,
                "source": source,
                "reviewed": bool(anchor.get("reviewed", False)),
                "overridden_observations": len(tracks[tid]),
            }
        )
    label_changes = (
        document.get("tracklet_label_changes", []) if isinstance(document, dict) else []
    )
    if not isinstance(label_changes, list):
        raise TypeError("tracklet_label_changes must be a list")
    for correction in label_changes:
        tid = int(correction["tid"])
        source = str(correction.get("source", ""))
        if source not in AUTHORITATIVE_RESOLUTION_SOURCES:
            raise ValueError(f"non-authoritative label-change source {source!r}")
        if source == "operator" and correction.get("reviewed") is not True:
            raise ValueError("operator label changes require reviewed=true")
        changes = sorted(
            (int(frame), str(label))
            for frame, label in correction.get("changes", {}).items()
        )
        if tid not in tracks or not changes:
            raise ValueError(f"invalid label changes for tid {tid}")
        if changes[0][0] > min(tracks[tid]):
            raise ValueError(f"tid {tid}: label changes do not cover its first frame")
        if any(label not in label_paths for _, label in changes):
            raise ValueError(f"tid {tid}: label change has no path binding")
        count = 0
        for frame in tracks[tid]:
            prior = [label for at, label in changes if at <= frame]
            if prior:
                overrides[(tid, frame)] = label_paths[prior[-1]]
                count += 1
        applied.append(
            {
                "kind": "tracklet_label_changes",
                "tid": tid,
                "source": source,
                "reviewed": bool(correction.get("reviewed", False)),
                "changes": {str(frame): label for frame, label in changes},
                "overridden_observations": count,
            }
        )
    result = (overrides, applied, resolved)
    if include_bindings:
        return (*result, dict(sorted(label_paths.items())))
    return result


def named_path_identities(bindings, document):
    """Promote bindings only when the evidence file declares real person IDs."""
    if (
        not isinstance(document, dict)
        or document.get("labels_are_person_ids") is not True
    ):
        return {}
    source = str(document.get("identity_source", "operator"))
    if source not in AUTHORITATIVE_RESOLUTION_SOURCES:
        raise ValueError(f"non-authoritative named identity source {source!r}")
    return {
        str(path_id): {
            "person_id": person_id,
            "status": "anchored",
            "confidence": 1.0,
            "evidence": [{"source": source, "reviewed": source == "operator"}],
        }
        for person_id, path_id in bindings.items()
    }
