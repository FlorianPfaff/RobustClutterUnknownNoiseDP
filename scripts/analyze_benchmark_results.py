"""Analyze structured-clutter benchmark CSV artifacts.

Usage:
    python scripts/analyze_benchmark_results.py benchmark-results/

The script intentionally avoids plotting dependencies. It produces deterministic
Markdown tables that can be copied into experiment notes or the paper repository.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


REQUIRED_FILES = (
    "summary_by_scenario_method.csv",
    "paired_deltas_vs_reference.csv",
)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def format_table(rows: list[dict[str, str]], columns: list[str]) -> str:
    if not rows:
        return "(no rows)"

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(row.get(column, "") for column in columns) + " |")
    return "\n".join([header, separator, *body])


def summarize(summary_rows: list[dict[str, str]]) -> str:
    columns = [
        "scenario",
        "method",
        "mean_false_tracks",
        "mean_gospa_false_cost",
        "mean_merged_gospa_false_cost",
        "mean_missed_targets",
        "mean_gospa_distance",
        "mean_merged_gospa_distance",
        "mean_fragments_per_confirmed_truth",
    ]
    return format_table(summary_rows, columns)


def summarize_paired(rows: list[dict[str, str]]) -> str:
    columns = [
        "scenario",
        "method",
        "reference_method",
        "num_pairs",
        "mean_delta_false_tracks",
        "mean_delta_gospa_false_cost",
        "mean_delta_missed_targets",
        "se_delta_false_tracks",
    ]
    return format_table(rows, columns)


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze benchmark CSV artifacts.")
    parser.add_argument("artifact_dir", type=Path)
    args = parser.parse_args()

    missing = [name for name in REQUIRED_FILES if not (args.artifact_dir / name).exists()]
    if missing:
        parser.error(f"missing benchmark files: {', '.join(missing)}")

    summary_rows = load_csv(args.artifact_dir / "summary_by_scenario_method.csv")
    paired_rows = load_csv(args.artifact_dir / "paired_deltas_vs_reference.csv")

    print("# Scenario/method summary")
    print()
    print(summarize(summary_rows))
    print()
    print("# Paired deltas versus reference")
    print()
    print(summarize_paired(paired_rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
