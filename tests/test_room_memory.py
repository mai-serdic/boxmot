from reid.room_memory import RoomMemory
from reid.trajectory_stitch import Tracklet


def test_memory_keeps_nonoverlapping_fragment_on_same_path():
    class R:
        pass

    # Avoid geometry dependency: subclass the method used by link_prob.
    mem = RoomMemory(R(), 10, None, min_link_prob=0.0, batch_window_frames=0)
    mem._link = None
    # Empty reachability cannot produce a real link, so test the invariant
    # directly through a patched link function at module scope.
    import reid.room_memory as rm

    old = rm.link_prob
    rm.link_prob = lambda *args: 0.9
    try:
        a = Tracklet(
            1, __import__("numpy").array([0, 1]), __import__("numpy").zeros((2, 2))
        )
        b = Tracklet(
            2, __import__("numpy").array([5, 6]), __import__("numpy").zeros((2, 2))
        )
        assert mem.observe([a, b]) == {1: 1, 2: 1}
        snap = mem.snapshot()[1]
        assert (
            snap["occupied_intervals"] == [(0, 1), (5, 6)]
            and snap["status"] == "visible"
        )
    finally:
        rm.link_prob = old


def test_batch_assignment_is_one_to_one():
    class R:
        pass

    import numpy as np

    import reid.room_memory as rm

    mem = RoomMemory(R(), 10, None, min_link_prob=0.1, batch_window_frames=20)
    old = rm.link_prob
    rm.link_prob = lambda *args: 0.9
    try:
        first = [
            Tracklet(1, np.array([0]), np.zeros((1, 2))),
            Tracklet(2, np.array([0]), np.ones((1, 2))),
        ]
        later = [
            Tracklet(3, np.array([30]), np.zeros((1, 2))),
            Tracklet(4, np.array([30]), np.ones((1, 2))),
        ]
        owner = mem.observe(first + later)
        assert owner[3] != owner[4] and {owner[3], owner[4]} == {owner[1], owner[2]}
    finally:
        rm.link_prob = old
