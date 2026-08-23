from reid.occupancy_partition import bipartite_room_partition


def test_bipartite_overlap_collapses_fragments_to_two_paths():
    traj = {
        "1": [[0, 0, 0, 1, 1], [1, 0, 0, 1, 1]],
        "2": [[0, 2, 0, 3, 1], [1, 2, 0, 3, 1]],
        "3": [[3, 0, 0, 1, 1]],
    }
    owner, diagnostic = bipartite_room_partition(traj, {1: 10, 2: 20, 3: 10})
    assert diagnostic["status"] == "applied"
    assert owner[1] != owner[2] and owner[1] == owner[3]


def test_triangle_refuses_unsafe_two_path_collapse():
    traj = {
        str(tid): [[0, tid, 0, tid + 1, 1], [1, tid, 0, tid + 1, 1]]
        for tid in (1, 2, 3)
    }
    baseline = {1: 1, 2: 2, 3: 3}
    owner, diagnostic = bipartite_room_partition(traj, baseline)
    assert owner == baseline
    assert diagnostic["reason"] == "co-presence graph is not bipartite"
