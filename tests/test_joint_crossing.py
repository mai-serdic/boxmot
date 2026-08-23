from types import SimpleNamespace

from reid.joint_crossing import (
    JointState,
    crossing_path_overrides,
    hypotheses_for_event,
    resolve_from_anchor,
    transition_cost,
)


def test_joint_states_include_duplicate_and_allow_return():
    event = SimpleNamespace(tid_a=1, tid_b=2, start=4, end=4)
    traj = {"1": [[4, 0, 0, 1, 1]], "2": [[4, 0, 0, 1, 1]]}
    states = hypotheses_for_event(event, traj)[0].states
    assert JointState.DUPLICATE_A in states and JointState.SWAPPED in states
    assert transition_cost(JointState.SWAPPED, JointState.SAME) > 0
    assert resolve_from_anchor("alice", "bob", "bob", "alice") == JointState.SWAPPED


def test_resolved_swap_rewrites_tails_but_temporary_duplicate_does_not():
    event = SimpleNamespace(tid_a=1, tid_b=2, start=1, end=2)
    traj = {
        "1": [[f, 0, 0, 1, 1] for f in range(4)],
        "2": [[f, 0, 0, 1, 1] for f in range(4)],
    }
    out = crossing_path_overrides(
        event, traj, {1: 10, 2: 20}, {1: JointState.DUPLICATE_A, 2: JointState.SAME}
    )
    assert out[(2, 1)] == 10 and (1, 3) not in out
    out = crossing_path_overrides(
        event,
        traj,
        {1: 10, 2: 20},
        {1: JointState.SWAPPED, 2: JointState.SWAPPED},
        carry_tail=True,
    )
    assert out[(1, 3)] == 20 and out[(2, 3)] == 10


def test_resolution_document_requires_review_and_explicit_tail_carry():
    from reid.joint_crossing import apply_crossing_resolutions

    event = SimpleNamespace(tid_a=1, tid_b=2, start=1, end=2)
    traj = {
        "1": [[f, 0, 0, 1, 1] for f in range(4)],
        "2": [[f, 0, 0, 1, 1] for f in range(4)],
    }
    document = {
        "crossing_resolutions": [
            {
                "event_id": "crossing:1:2:1",
                "source": "operator",
                "reviewed": True,
                "states": {"1": "same", "2": "swapped"},
                "complete": True,
                "carry_forward": True,
            }
        ]
    }
    overrides, applied, resolved = apply_crossing_resolutions(
        [event], traj, {1: 10, 2: 20}, document
    )
    assert overrides[(1, 1)] == 10 and overrides[(1, 3)] == 20
    assert applied[0]["status"] == "resolved"
    assert resolved == {"crossing:1:2:1"}


def test_later_crossing_uses_ownership_after_earlier_confirmed_swap():
    from reid.joint_crossing import apply_crossing_resolutions

    events = [
        SimpleNamespace(tid_a=1, tid_b=2, start=1, end=1),
        SimpleNamespace(tid_a=1, tid_b=2, start=3, end=3),
    ]
    traj = {
        "1": [[f, 0, 0, 1, 1] for f in range(5)],
        "2": [[f, 0, 0, 1, 1] for f in range(5)],
    }
    document = {
        "crossing_resolutions": [
            {
                "event_id": "crossing:1:2:1",
                "source": "badge",
                "states": {"1": "swapped"},
                "complete": True,
                "carry_forward": True,
            },
            {
                "event_id": "crossing:1:2:3",
                "source": "badge",
                "states": {"3": "swapped"},
                "complete": True,
                "carry_forward": True,
            },
        ]
    }
    overrides, _applied, _resolved = apply_crossing_resolutions(
        events, traj, {1: 10, 2: 20}, document
    )
    assert overrides[(1, 2)] == 20
    assert overrides[(1, 4)] == 10


def test_reviewed_labels_become_sparse_state_changes_and_proven_tail():
    from reid.joint_crossing import states_from_reviewed_labels

    event = SimpleNamespace(tid_a=1, tid_b=2, start=1, end=2)
    traj = {
        "1": [[f, 0, 0, 1, 1] for f in range(4)],
        "2": [[f, 0, 0, 1, 1] for f in range(4)],
    }
    labels = {
        (1, 0): "alice",
        (2, 0): "bob",
        (1, 1): "alice",
        (2, 1): "bob",
        (1, 2): "bob",
        (2, 2): "alice",
        (1, 3): "bob",
        (2, 3): "alice",
    }
    changes, complete, carry, incoming = states_from_reviewed_labels(
        event, traj, labels
    )
    assert changes == {1: JointState.SAME, 2: JointState.SWAPPED}
    assert complete and carry and incoming == ("alice", "bob")


def test_reviewed_identity_ledger_can_separate_wrongly_grouped_later_fragments():
    from reid.joint_crossing import apply_crossing_resolutions

    events = [
        SimpleNamespace(tid_a=1, tid_b=2, start=1, end=1),
        SimpleNamespace(tid_a=3, tid_b=4, start=3, end=3),
    ]
    traj = {
        str(tid): [[frame, 0, 0, 1, 1]] for tid, frame in zip(range(1, 5), [1, 1, 3, 3])
    }
    document = {
        "crossing_resolutions": [
            {
                "event_id": "crossing:1:2:1",
                "source": "operator",
                "reviewed": True,
                "states": {"1": "same"},
                "complete": True,
                "incoming_labels": {"a": "alice", "b": "bob"},
            },
            {
                "event_id": "crossing:3:4:3",
                "source": "operator",
                "reviewed": True,
                "states": {"3": "same"},
                "complete": True,
                "incoming_labels": {"a": "alice", "b": "bob"},
            },
        ]
    }
    overrides, _applied, _resolved = apply_crossing_resolutions(
        events, traj, {1: 10, 2: 20, 3: 10, 4: 10}, document
    )
    assert overrides[(3, 3)] == 10
    assert overrides[(4, 3)] == 20


def test_resolution_fingerprint_rejects_another_tracker_run():
    import pytest

    from reid.joint_crossing import apply_crossing_resolutions

    event = SimpleNamespace(tid_a=1, tid_b=2, start=1, end=1)
    traj = {"1": [[1, 0, 0, 1, 1]], "2": [[1, 2, 0, 3, 1]]}
    document = {
        "trajectory_fingerprint": "not-this-trajectory",
        "crossing_resolutions": [],
    }
    with pytest.raises(ValueError, match="different trajectory"):
        apply_crossing_resolutions([event], traj, {1: 1, 2: 2}, document)


def test_authoritative_pure_tracklet_anchor_rewrites_its_whole_fragment():
    from reid.joint_crossing import apply_crossing_resolutions

    event = SimpleNamespace(tid_a=1, tid_b=2, start=1, end=1)
    traj = {
        "1": [[1, 0, 0, 1, 1]],
        "2": [[1, 2, 0, 3, 1]],
        "3": [[0, 4, 0, 5, 1], [2, 4, 0, 5, 1]],
    }
    document = {
        "crossing_resolutions": [
            {
                "event_id": "crossing:1:2:1",
                "source": "operator",
                "reviewed": True,
                "states": {"1": "same"},
                "complete": True,
                "incoming_labels": {"a": "alice", "b": "bob"},
            }
        ],
        "tracklet_anchors": [
            {"tid": 3, "label": "bob", "source": "operator", "reviewed": True}
        ],
    }
    overrides, _applied, _resolved = apply_crossing_resolutions(
        [event], traj, {1: 10, 2: 20, 3: 10}, document
    )
    assert overrides[(3, 0)] == 20 and overrides[(3, 2)] == 20


def test_reviewed_tracklet_change_points_rewrite_only_at_the_cut():
    from reid.joint_crossing import apply_crossing_resolutions

    event = SimpleNamespace(tid_a=1, tid_b=2, start=0, end=0)
    traj = {
        "1": [[f, 0, 0, 1, 1] for f in range(4)],
        "2": [[0, 2, 0, 3, 1]],
    }
    document = {
        "crossing_resolutions": [
            {
                "event_id": "crossing:1:2:0",
                "source": "operator",
                "reviewed": True,
                "states": {"0": "same"},
                "complete": True,
                "incoming_labels": {"a": "alice", "b": "bob"},
            }
        ],
        "tracklet_label_changes": [
            {
                "tid": 1,
                "changes": {"0": "alice", "2": "bob"},
                "source": "operator",
                "reviewed": True,
            }
        ],
    }
    overrides, _applied, _resolved = apply_crossing_resolutions(
        [event], traj, {1: 10, 2: 20}, document
    )
    assert [overrides[(1, frame)] for frame in range(4)] == [10, 10, 20, 20]


def test_named_identity_requires_explicit_real_person_id_declaration():
    from reid.joint_crossing import named_path_identities

    assert named_path_identities({"alice": 1}, {}) == {}
    out = named_path_identities(
        {"alice": 1},
        {"labels_are_person_ids": True, "identity_source": "badge"},
    )
    assert out["1"]["person_id"] == "alice" and out["1"]["status"] == "anchored"
