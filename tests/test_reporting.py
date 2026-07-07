from robust_clutter_dp import (
    MethodResult,
    aggregate_method_results,
    format_method_aggregates_csv,
    format_method_results_csv,
)


def _result(method: str, seed: int, false_tracks: int, gospa_false_cost: float) -> MethodResult:
    return MethodResult(
        method=method,
        seed=seed,
        confirmed_tracks=2,
        true_confirmed_tracks=2 - false_tracks,
        false_tracks=false_tracks,
        false_track_duration=false_tracks * 3,
        mean_time_to_confirm=3.0,
        mean_true_time_to_confirm=2.5,
        missed_targets=1,
        gospa_distance=1.0,
        gospa_total_cost=2.0 + gospa_false_cost,
        gospa_localization_cost=0.5,
        gospa_missed_cost=1.0,
        gospa_false_cost=gospa_false_cost,
        posterior_expected_fdr=0.1,
        observed_false_discovery_proportion=0.25,
        existence_brier_score=0.05,
    )


def test_aggregate_method_results_groups_by_method_and_averages_metrics():
    aggregates = aggregate_method_results(
        [
            _result("dp", seed=0, false_tracks=0, gospa_false_cost=0.0),
            _result("dp", seed=1, false_tracks=2, gospa_false_cost=4.0),
            _result("uniform", seed=0, false_tracks=1, gospa_false_cost=2.0),
        ]
    )

    assert [aggregate.method for aggregate in aggregates] == ["dp", "uniform"]
    dp = aggregates[0]
    assert dp.num_runs == 2
    assert dp.mean_false_tracks == 1.0
    assert dp.mean_false_track_duration == 3.0
    assert dp.mean_gospa_false_cost == 2.0
    assert dp.mean_time_to_confirm == 3.0
    assert dp.mean_true_time_to_confirm == 2.5


def test_format_method_csv_helpers_include_headers_and_rows():
    results = [_result("dp", seed=0, false_tracks=0, gospa_false_cost=0.0)]
    aggregates = aggregate_method_results(results)

    raw_csv = format_method_results_csv(results)
    aggregate_csv = format_method_aggregates_csv(aggregates)

    assert raw_csv.startswith("method,seed,confirmed_tracks")
    assert "dp,0," in raw_csv
    assert aggregate_csv.startswith("method,num_runs,mean_confirmed_tracks")
    assert "mean_true_time_to_confirm" in aggregate_csv
