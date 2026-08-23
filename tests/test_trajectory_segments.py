from reid.trajectory_segments import split_trajectory


def test_split_preserves_every_observation_and_provenance():
    traj = {"7": [[f, 0, 0, 1, 1] for f in range(6)]}
    segs, prov = split_trajectory(traj, {7: [2, 5]})
    assert [(s.t0, s.t1, s.source_tid) for s in segs] == [
        (0, 1, 7),
        (2, 4, 7),
        (5, 5, 7),
    ]
    assert len(prov) == 6 and prov[(7, 5)] == 3
