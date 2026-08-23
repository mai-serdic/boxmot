from reid.trajectory_stitch import number_unassigned_tracklets, path_frames_from_traj


def test_filtered_tracklets_are_explicitly_unassigned():
    traj = {
        "1": [[0, 0, 0, 1, 1]],
        "9": [[0, 2, 2, 3, 3]],
    }
    frames = path_frames_from_traj(traj, {1: 1}, unassigned_id=0)
    assert frames[0] == [[1, 0, 0, 1, 1], [0, 2, 2, 3, 3]]


def test_frame_owner_can_revise_part_of_a_tracker_id():
    traj = {"1": [[0, 0, 0, 1, 1], [1, 0, 0, 1, 1]]}
    frames = path_frames_from_traj(traj, {1: 1, 2: 2}, frame_owner={(1, 1): 2})
    assert frames[0][0][0] == 1 and frames[1][0][0] == 2


def test_frame_owner_is_compacted_in_the_same_namespace_as_owner():
    traj = {"1": [[0, 0, 0, 1, 1]], "2": [[0, 2, 2, 3, 3]]}
    frames = path_frames_from_traj(traj, {1: 10, 2: 20}, frame_owner={(1, 0): 20})
    assert frames[0] == [[2, 0, 0, 1, 1], [2, 2, 2, 3, 3]]


def test_short_fragments_get_new_paths_without_changing_core_owners():
    traj = {"1": [[5, 0, 0, 1, 1]], "9": [[1, 2, 2, 3, 3]]}
    owner, short = number_unassigned_tracklets({1: 4}, traj)
    assert owner == {1: 4, 9: 5} and short == [9]
