from reid.path_labels import propagate_tracklet, validate_exclusivity


def test_propagation_does_not_cross_tracklet_and_respects_cut():
    rows = [[f, 0, 0, 1, 1] for f in range(5)]
    out = propagate_tracklet(
        rows,
        [
            {"tid": 7, "frame": 1, "label": "1"},
            {"tid": 7, "frame": 3, "label": "2"},
        ],
    )
    assert [x["label"] for x in out] == ["1", "1", "1", "2", "2"]
    assert all(x["tid"] == 7 for x in out)


def test_exclusivity_reports_conflict_without_fixing_it():
    labels = [
        {"tid": 1, "frame": 4, "label": "1"},
        {"tid": 2, "frame": 4, "label": "1"},
    ]
    assert validate_exclusivity(labels) == [(4, "1", [1, 2])]
