"""Run the minimal structured-clutter comparison experiment.

This produces one CSV-like table on stdout. It is deliberately small enough to
serve as a smoke test and as the seed for the first paper-facing benchmark.
"""

from __future__ import annotations

from robust_clutter_dp import ExperimentConfig, run_structured_clutter_comparison


def main() -> None:
    results = run_structured_clutter_comparison(
        seeds=range(5),
        experiment_config=ExperimentConfig(methods=("uniform", "grid", "dp", "oracle")),
    )
    rows = [result.to_dict() for result in results]
    columns = list(rows[0])

    print(",".join(columns))
    for row in rows:
        print(",".join(str(row[column]) for column in columns))


if __name__ == "__main__":
    main()
