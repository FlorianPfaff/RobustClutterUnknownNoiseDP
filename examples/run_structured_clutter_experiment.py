"""Run the minimal structured-clutter comparison experiment.

This prints raw per-seed rows, cross-seed scenario/method summaries, and deltas
against the oracle clutter model. It is small enough to serve as a smoke test
and as the seed for the first paper-facing benchmark.
"""

from __future__ import annotations

from robust_clutter_dp import (
    ExperimentConfig,
    aggregate_method_results,
    compare_to_reference,
    format_method_aggregates_csv,
    format_method_comparisons_csv,
    format_method_results_csv,
    run_named_scenarios_comparison,
)


def main() -> None:
    results = run_named_scenarios_comparison(
        scenario_names=("hotspot", "no_hotspot_control", "near_hotspot_crossing"),
        seeds=range(5),
        experiment_config=ExperimentConfig(methods=("uniform", "grid", "dp", "oracle")),
    )
    aggregates = aggregate_method_results(results)
    comparisons = compare_to_reference(aggregates, reference_method="oracle")

    print("# raw per-seed results")
    print(format_method_results_csv(results))
    print()
    print("# cross-seed scenario/method summary")
    print(format_method_aggregates_csv(aggregates))
    print()
    print("# deltas versus oracle clutter model")
    print(format_method_comparisons_csv(comparisons))


if __name__ == "__main__":
    main()
