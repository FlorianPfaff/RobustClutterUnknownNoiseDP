import pytest

from robust_clutter_dp import (
    ExperimentConfig,
    SCENARIO_NAMES,
    make_scenario,
    run_named_scenarios_comparison,
)


def test_named_scenarios_are_constructible():
    for name in SCENARIO_NAMES:
        scenario = make_scenario(name)
        assert scenario.num_scans > 0
        assert scenario.dimension == 2
        assert scenario.total_clutter_rate >= 0.0


def test_unknown_scenario_raises_clear_error():
    with pytest.raises(ValueError, match="unknown scenario"):
        make_scenario("does_not_exist")


def test_run_named_scenarios_comparison_tags_result_rows():
    results = run_named_scenarios_comparison(
        scenario_names=("hotspot", "no_hotspot_control"),
        seeds=(0,),
        experiment_config=ExperimentConfig(methods=("uniform", "oracle")),
    )

    assert {result.scenario for result in results} == {"hotspot", "no_hotspot_control"}
    assert {result.method for result in results} == {"uniform", "oracle"}
    assert len(results) == 4
