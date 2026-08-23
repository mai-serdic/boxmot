from reid.crossing_events import (
    candidate_split_frames,
    detect_crossing_events,
    unresolved_crossings,
)


def test_detects_close_interaction_and_returns_boundaries():
    traj = {
        "1": [[f, 0, 0, 10, 20] for f in range(5)],
        "2": [[f, 5, 0, 15, 20] for f in range(1, 4)],
    }
    ev = detect_crossing_events(traj, threshold=1.0, min_frames=3)
    assert len(ev) == 1 and (ev[0].start, ev[0].end) == (1, 3)
    assert candidate_split_frames(ev) == {1: [1, 4], 2: [1, 4]}
    assert unresolved_crossings(ev, threshold=1.0)[0]["status"] == "unresolved"
