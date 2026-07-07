"""Run the minimal structured-clutter comparison experiment.

This prints raw per-seed rows followed by a cross-seed summary table. It is
small enough to serve as a smoke test and as the seed for the first paper-facing
benchmark.
"""

from __future__ import annotations

from robust_clutter_dp import (
    ExperimentConfig,
    aggregate_method_results,
    format_method_aggregates_csv,
    format_method_results_csv,
    run_structured_clutter_comparison,
)


def main() -> None:
    results = run_structured_clutter_comparison(
        seeds=range(5),
        experiment_config=ExperimentConfig(methods=("uniform", "grid", "dp", "oracle")),
    )
    aggregates = aggregate_method_results(results)

    print("# raw per-seed results")
    print(format_method_results_csv(results))
    print()
    print("# cross-seed method summary")
    print(format_method_aggregates_csv(aggregates))


if __name__ == "__main__":
    main()
