import pytest

from robust_clutter_dp.cli import (
    build_parser,
    main,
    make_experiment_config,
    parse_method_spec,
    parse_scenario_spec,
    parse_seed_spec,
    run_cli,
)


def test_parse_seed_spec_accepts_ranges_and_lists():
    assert parse_seed_spec("0:5") == (0, 1, 2, 3, 4)
    assert parse_seed_spec("1:6:2") == (1, 3, 5)
    assert parse_seed_spec("0,3,7") == (0, 3, 7)


@pytest.mark.parametrize("spec", ["", "0:5:0", "0:1:2:3"])
def test_parse_seed_spec_rejects_invalid_specs(spec):
    with pytest.raises(ValueError):
        parse_seed_spec(spec)


def test_parse_scenario_and_method_specs_accept_all_and_validate_unknowns():
    assert "hotspot" in parse_scenario_spec("all")
    assert parse_scenario_spec("hotspot,no_hotspot_control") == ("hotspot", "no_hotspot_control")
    assert parse_method_spec("uniform,dp") == ("uniform", "dp")

    with pytest.raises(ValueError, match="unknown scenario"):
        parse_scenario_spec("missing")
    with pytest.raises(ValueError, match="unknown method"):
        parse_method_spec("missing")


def test_make_experiment_config_applies_cli_overrides():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--methods",
            "uniform,oracle",
            "--birth-rate",
            "0.4",
            "--birth-threshold",
            "0.3",
            "--min-confirmation-probability",
            "0.8",
            "--fdr-q",
            "0.05",
        ]
    )

    config = make_experiment_config(args)

    assert config.methods == ("uniform", "oracle")
    assert config.birth_rate == 0.4
    assert config.tracklet_config.birth_probability_threshold == 0.3
    assert config.tracklet_config.min_confirmation_probability == 0.8
    assert config.tracklet_config.fdr_q == 0.05


def test_run_cli_outputs_requested_summary_table():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--scenarios",
            "hotspot",
            "--methods",
            "uniform,oracle",
            "--seeds",
            "0:1",
            "--output",
            "summary",
        ]
    )

    output = run_cli(args)

    assert "# cross-seed scenario/method summary" in output
    assert "scenario,method,num_runs" in output
    assert "hotspot,uniform" in output
    assert "hotspot,oracle" in output


def test_run_cli_outputs_paired_comparison_table():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--scenarios",
            "hotspot",
            "--methods",
            "uniform,oracle",
            "--seeds",
            "0:2",
            "--output",
            "paired-comparison",
        ]
    )

    output = run_cli(args)

    assert "# paired deltas versus reference clutter model" in output
    assert "scenario,method,reference_method,num_pairs" in output
    assert "se_delta_false_tracks" in output


def test_main_returns_zero_for_valid_small_run(capsys):
    exit_code = main(
        [
            "--scenarios",
            "hotspot",
            "--methods",
            "uniform,oracle",
            "--seeds",
            "0:1",
            "--output",
            "comparison",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert "# deltas versus reference clutter model" in captured.out
