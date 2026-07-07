import numpy as np

from robust_clutter_dp import (
    AssociationResult,
    ExperimentConfig,
    SimulationConfig,
    TentativeBirthManager,
    TrackletManagerConfig,
    run_structured_clutter_comparison,
)


def _association_result(birth_log_weight: float, clutter_log_weight: float) -> AssociationResult:
    weights = {"birth": birth_log_weight, "clutter": clutter_log_weight}
    max_weight = max(weights.values())
    normalizer = sum(np.exp(value - max_weight) for value in weights.values())
    probabilities = {
        key: float(np.exp(value - max_weight) / normalizer)
        for key, value in weights.items()
    }
    return AssociationResult(log_weights=weights, probabilities=probabilities)


def test_tentative_birth_manager_confirms_consistent_sequence():
    manager = TentativeBirthManager(
        TrackletManagerConfig(
            prior_existence=0.10,
            birth_probability_threshold=0.20,
            association_distance=0.75,
            dynamic_sigma=0.45,
            min_updates_for_confirmation=3,
            min_motion_span_for_confirmation=0.0,
            min_confirmation_probability=0.0,
            fdr_q=0.50,
        )
    )

    for scan_index, measurement in enumerate(
        [np.array([0.0, 0.0]), np.array([0.3, 0.0]), np.array([0.6, 0.0])]
    ):
        manager.process_measurements(
            scan_index=scan_index,
            measurements=[measurement],
            association_results=[_association_result(0.0, -4.0)],
            source_ids=["target-1"],
        )

    assert len(manager.confirmed_tracklets) == 1
    confirmed = manager.confirmed_tracklets[0]
    assert confirmed.num_measurements == 3
    assert confirmed.dominant_source_id == "target-1"
    assert confirmed.existence_probability > 0.9


def test_tentative_birth_manager_ignores_clutter_dominated_measurements():
    manager = TentativeBirthManager(
        TrackletManagerConfig(
            birth_probability_threshold=0.50,
            min_updates_for_confirmation=2,
            fdr_q=0.50,
        )
    )

    manager.process_measurements(
        scan_index=0,
        measurements=[np.array([2.0, 2.0])],
        association_results=[_association_result(-5.0, 0.0)],
        source_ids=[None],
    )

    assert manager.active_tracklets == ()
    assert manager.confirmed_tracklets == ()


def test_structured_clutter_comparison_runs_all_requested_methods():
    simulation_config = SimulationConfig(num_scans=8, calibration_scans=4)
    experiment_config = ExperimentConfig(methods=("uniform", "grid", "dp", "oracle"))

    results = run_structured_clutter_comparison(
        seeds=(1,),
        simulation_config=simulation_config,
        experiment_config=experiment_config,
    )

    assert {result.method for result in results} == {"uniform", "grid", "dp", "oracle"}
    assert all(np.isfinite(result.gospa_distance) for result in results)
    assert all(result.confirmed_tracks >= result.false_tracks for result in results)
    assert all(result.gospa_false_cost >= 0.0 for result in results)
    assert all(result.gospa_missed_cost >= 0.0 for result in results)
