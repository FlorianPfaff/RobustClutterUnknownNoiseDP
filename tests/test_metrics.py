import numpy as np

from robust_clutter_dp import (
    ConfirmationRecord,
    PointObject,
    confirmation_metrics,
    gospa_decomposition,
)


def test_gospa_decomposition_exposes_false_and_missed_components():
    estimates = [
        PointObject("e1", np.array([0.1, 0.0])),
        PointObject("false-track", np.array([10.0, 10.0])),
    ]
    truths = [
        PointObject("t1", np.array([0.0, 0.0])),
        PointObject("missed-target", np.array([5.0, 5.0])),
    ]

    result = gospa_decomposition(estimates, truths, cutoff=2.0, p=2.0, alpha=2.0)

    assert result.num_matches == 1
    assert result.num_false == 1
    assert result.num_missed == 1
    assert result.matches[0].estimate_id == "e1"
    assert result.matches[0].truth_id == "t1"
    assert result.false_estimate_ids == ("false-track",)
    assert result.missed_truth_ids == ("missed-target",)
    assert np.isclose(result.localization_cost, 0.01)
    assert np.isclose(result.false_cost, 2.0)
    assert np.isclose(result.missed_cost, 2.0)


def test_gospa_decomposition_prefers_unmatched_over_far_match():
    estimates = [PointObject("e", np.array([100.0, 100.0]))]
    truths = [PointObject("t", np.array([0.0, 0.0]))]

    result = gospa_decomposition(estimates, truths, cutoff=1.0, p=2.0, alpha=2.0)

    assert result.num_matches == 0
    assert result.false_estimate_ids == ("e",)
    assert result.missed_truth_ids == ("t",)
    assert np.isclose(result.total_cost, 1.0)
    assert np.isclose(result.distance, 1.0)


def test_confirmation_metrics_report_observed_and_posterior_fdr():
    records = [
        ConfirmationRecord("a", existence_probability=0.99, is_target=True, accepted=True),
        ConfirmationRecord("b", existence_probability=0.81, is_target=False, accepted=True),
        ConfirmationRecord("c", existence_probability=0.30, is_target=False, accepted=False),
    ]

    metrics = confirmation_metrics(records)

    assert metrics.accepted_count == 2
    assert metrics.true_accepted_count == 1
    assert metrics.false_accepted_count == 1
    assert np.isclose(metrics.false_discovery_proportion, 0.5)
    assert np.isclose(metrics.posterior_expected_fdr, 0.10)
    assert metrics.brier_score >= 0.0
