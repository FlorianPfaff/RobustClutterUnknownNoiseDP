import numpy as np

from robust_clutter_dp.tracklet import TentativeTracklet
from robust_clutter_dp.tracklet_merge import (
    fragmentation_counts_by_truth,
    mean_fragments_per_confirmed_truth,
    merge_confirmed_tracklets,
)


def _tracklet(candidate_id, first_scan, points, source_id=None):
    measurements = tuple(np.asarray(point, dtype=float) for point in points)
    scan_indices = tuple(range(first_scan, first_scan + len(points)))
    return TentativeTracklet(
        candidate_id=candidate_id,
        first_scan=scan_indices[0],
        last_scan=scan_indices[-1],
        measurements=measurements,
        scan_indices=scan_indices,
        source_ids=tuple(source_id for _ in measurements),
        log_bayes_factor=3.0,
        existence_probability=0.95,
    )


def test_merge_confirmed_tracklets_merges_same_source_fragments():
    fragments = [
        _tracklet("a", 0, [[0.0, 0.0], [0.2, 0.0]], source_id="truth-1"),
        _tracklet("b", 4, [[0.8, 0.0], [1.0, 0.0]], source_id="truth-1"),
        _tracklet("c", 0, [[5.0, 5.0]], source_id="truth-2"),
    ]

    merged = merge_confirmed_tracklets(fragments)

    assert len(merged) == 2
    by_source = {group.dominant_source_id: group for group in merged}
    assert by_source["truth-1"].num_fragments == 2
    assert by_source["truth-1"].merge_reason == "source"
    assert by_source["truth-2"].num_fragments == 1


def test_merge_confirmed_tracklets_merges_unlabeled_adjacent_geometry():
    fragments = [
        _tracklet("a", 0, [[0.0, 0.0], [0.2, 0.0]]),
        _tracklet("b", 2, [[0.25, 0.05], [0.4, 0.0]]),
        _tracklet("c", 2, [[5.0, 5.0], [5.2, 5.1]]),
    ]

    merged = merge_confirmed_tracklets(fragments, max_gap=1, max_distance=0.5)

    sizes = sorted(group.num_fragments for group in merged)
    assert sizes == [1, 2]
    assert any(group.merge_reason == "geometry" for group in merged)


def test_fragmentation_counts_and_mean_ignore_unlabeled_tracklets():
    fragments = [
        _tracklet("a", 0, [[0.0, 0.0]], source_id="truth-1"),
        _tracklet("b", 2, [[0.2, 0.0]], source_id="truth-1"),
        _tracklet("c", 0, [[5.0, 5.0]]),
    ]

    assert fragmentation_counts_by_truth(fragments) == {"truth-1": 2}
    assert mean_fragments_per_confirmed_truth(fragments) == 2.0
    assert mean_fragments_per_confirmed_truth([]) == 0.0
