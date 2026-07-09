"""Reporting helpers for structured-clutter experiments."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean, stdev
from typing import Iterable, Sequence

from .experiment import MethodResult


@dataclass(frozen=True)
class MethodAggregate:
    """Cross-seed aggregate for one scenario/method pair."""

    scenario: str
    method: str
    num_runs: int
    mean_confirmed_tracks: float
    mean_true_confirmed_tracks: float
    mean_false_tracks: float
    mean_false_track_duration: float
    mean_track_fragment_count: float
    mean_fragments_per_confirmed_truth: float
    mean_merged_estimates: float
    mean_time_to_confirm: float
    mean_true_time_to_confirm: float
    mean_missed_targets: float
    mean_gospa_distance: float
    mean_gospa_localization_cost: float
    mean_gospa_missed_cost: float
    mean_gospa_false_cost: float
    mean_merged_gospa_distance: float
    mean_merged_gospa_localization_cost: float
    mean_merged_gospa_missed_cost: float
    mean_merged_gospa_false_cost: float
    mean_posterior_expected_fdr: float
    mean_observed_false_discovery_proportion: float
    mean_existence_brier_score: float

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a CSV/dataframe-friendly representation."""

        return {
            "scenario": self.scenario,
            "method": self.method,
            "num_runs": self.num_runs,
            "mean_confirmed_tracks": self.mean_confirmed_tracks,
            "mean_true_confirmed_tracks": self.mean_true_confirmed_tracks,
            "mean_false_tracks": self.mean_false_tracks,
            "mean_false_track_duration": self.mean_false_track_duration,
            "mean_track_fragment_count": self.mean_track_fragment_count,
            "mean_fragments_per_confirmed_truth": self.mean_fragments_per_confirmed_truth,
            "mean_merged_estimates": self.mean_merged_estimates,
            "mean_time_to_confirm": self.mean_time_to_confirm,
            "mean_true_time_to_confirm": self.mean_true_time_to_confirm,
            "mean_missed_targets": self.mean_missed_targets,
            "mean_gospa_distance": self.mean_gospa_distance,
            "mean_gospa_localization_cost": self.mean_gospa_localization_cost,
            "mean_gospa_missed_cost": self.mean_gospa_missed_cost,
            "mean_gospa_false_cost": self.mean_gospa_false_cost,
            "mean_merged_gospa_distance": self.mean_merged_gospa_distance,
            "mean_merged_gospa_localization_cost": self.mean_merged_gospa_localization_cost,
            "mean_merged_gospa_missed_cost": self.mean_merged_gospa_missed_cost,
            "mean_merged_gospa_false_cost": self.mean_merged_gospa_false_cost,
            "mean_posterior_expected_fdr": self.mean_posterior_expected_fdr,
            "mean_observed_false_discovery_proportion": self.mean_observed_false_discovery_proportion,
            "mean_existence_brier_score": self.mean_existence_brier_score,
        }


@dataclass(frozen=True)
class MethodComparison:
    """Difference between one method aggregate and a reference aggregate."""

    scenario: str
    method: str
    reference_method: str
    delta_mean_false_tracks: float
    delta_mean_false_track_duration: float
    delta_mean_missed_targets: float
    delta_mean_gospa_distance: float
    delta_mean_gospa_false_cost: float
    delta_mean_gospa_missed_cost: float
    delta_mean_merged_gospa_distance: float
    delta_mean_merged_gospa_false_cost: float
    delta_mean_merged_gospa_missed_cost: float

    def to_dict(self) -> dict[str, float | str]:
        return {
            "scenario": self.scenario,
            "method": self.method,
            "reference_method": self.reference_method,
            "delta_mean_false_tracks": self.delta_mean_false_tracks,
            "delta_mean_false_track_duration": self.delta_mean_false_track_duration,
            "delta_mean_missed_targets": self.delta_mean_missed_targets,
            "delta_mean_gospa_distance": self.delta_mean_gospa_distance,
            "delta_mean_gospa_false_cost": self.delta_mean_gospa_false_cost,
            "delta_mean_gospa_missed_cost": self.delta_mean_gospa_missed_cost,
            "delta_mean_merged_gospa_distance": self.delta_mean_merged_gospa_distance,
            "delta_mean_merged_gospa_false_cost": self.delta_mean_merged_gospa_false_cost,
            "delta_mean_merged_gospa_missed_cost": self.delta_mean_merged_gospa_missed_cost,
        }


@dataclass(frozen=True)
class PairedMethodComparison:
    """Paired seed-wise deltas between one method and a reference method."""

    scenario: str
    method: str
    reference_method: str
    num_pairs: int
    mean_delta_false_tracks: float
    se_delta_false_tracks: float
    mean_delta_false_track_duration: float
    se_delta_false_track_duration: float
    mean_delta_missed_targets: float
    se_delta_missed_targets: float
    mean_delta_gospa_distance: float
    se_delta_gospa_distance: float
    mean_delta_gospa_false_cost: float
    se_delta_gospa_false_cost: float
    mean_delta_gospa_missed_cost: float
    se_delta_gospa_missed_cost: float
    mean_delta_merged_gospa_distance: float
    se_delta_merged_gospa_distance: float
    mean_delta_merged_gospa_false_cost: float
    se_delta_merged_gospa_false_cost: float
    mean_delta_merged_gospa_missed_cost: float
    se_delta_merged_gospa_missed_cost: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "scenario": self.scenario,
            "method": self.method,
            "reference_method": self.reference_method,
            "num_pairs": self.num_pairs,
            "mean_delta_false_tracks": self.mean_delta_false_tracks,
            "se_delta_false_tracks": self.se_delta_false_tracks,
            "mean_delta_false_track_duration": self.mean_delta_false_track_duration,
            "se_delta_false_track_duration": self.se_delta_false_track_duration,
            "mean_delta_missed_targets": self.mean_delta_missed_targets,
            "se_delta_missed_targets": self.se_delta_missed_targets,
            "mean_delta_gospa_distance": self.mean_delta_gospa_distance,
            "se_delta_gospa_distance": self.se_delta_gospa_distance,
            "mean_delta_gospa_false_cost": self.mean_delta_gospa_false_cost,
            "se_delta_gospa_false_cost": self.se_delta_gospa_false_cost,
            "mean_delta_gospa_missed_cost": self.mean_delta_gospa_missed_cost,
            "se_delta_gospa_missed_cost": self.se_delta_gospa_missed_cost,
            "mean_delta_merged_gospa_distance": self.mean_delta_merged_gospa_distance,
            "se_delta_merged_gospa_distance": self.se_delta_merged_gospa_distance,
            "mean_delta_merged_gospa_false_cost": self.mean_delta_merged_gospa_false_cost,
            "se_delta_merged_gospa_false_cost": self.se_delta_merged_gospa_false_cost,
            "mean_delta_merged_gospa_missed_cost": self.mean_delta_merged_gospa_missed_cost,
            "se_delta_merged_gospa_missed_cost": self.se_delta_merged_gospa_missed_cost,
        }


def aggregate_method_results(results: Sequence[MethodResult]) -> tuple[MethodAggregate, ...]:
    """Aggregate method results across seeds by scenario and method."""

    by_group: dict[tuple[str, str], list[MethodResult]] = {}
    for result in results:
        by_group.setdefault((result.scenario, result.method), []).append(result)

    aggregates: list[MethodAggregate] = []
    for scenario, method in sorted(by_group):
        method_results = by_group[(scenario, method)]
        aggregates.append(
            MethodAggregate(
                scenario=scenario,
                method=method,
                num_runs=len(method_results),
                mean_confirmed_tracks=_mean_attr(method_results, "confirmed_tracks"),
                mean_true_confirmed_tracks=_mean_attr(method_results, "true_confirmed_tracks"),
                mean_false_tracks=_mean_attr(method_results, "false_tracks"),
                mean_false_track_duration=_mean_attr(method_results, "false_track_duration"),
                mean_track_fragment_count=_mean_attr(method_results, "track_fragment_count"),
                mean_fragments_per_confirmed_truth=_mean_attr(method_results, "mean_fragments_per_confirmed_truth"),
                mean_merged_estimates=_mean_attr(method_results, "merged_estimates"),
                mean_time_to_confirm=_mean_attr(method_results, "mean_time_to_confirm"),
                mean_true_time_to_confirm=_mean_attr(method_results, "mean_true_time_to_confirm"),
                mean_missed_targets=_mean_attr(method_results, "missed_targets"),
                mean_gospa_distance=_mean_attr(method_results, "gospa_distance"),
                mean_gospa_localization_cost=_mean_attr(method_results, "gospa_localization_cost"),
                mean_gospa_missed_cost=_mean_attr(method_results, "gospa_missed_cost"),
                mean_gospa_false_cost=_mean_attr(method_results, "gospa_false_cost"),
                mean_merged_gospa_distance=_mean_attr(method_results, "merged_gospa_distance"),
                mean_merged_gospa_localization_cost=_mean_attr(method_results, "merged_gospa_localization_cost"),
                mean_merged_gospa_missed_cost=_mean_attr(method_results, "merged_gospa_missed_cost"),
                mean_merged_gospa_false_cost=_mean_attr(method_results, "merged_gospa_false_cost"),
                mean_posterior_expected_fdr=_mean_attr(method_results, "posterior_expected_fdr"),
                mean_observed_false_discovery_proportion=_mean_attr(
                    method_results,
                    "observed_false_discovery_proportion",
                ),
                mean_existence_brier_score=_mean_attr(method_results, "existence_brier_score"),
            )
        )
    return tuple(aggregates)


def compare_to_reference(
    aggregates: Sequence[MethodAggregate],
    reference_method: str = "oracle",
) -> tuple[MethodComparison, ...]:
    """Compare method aggregates against a reference method within each scenario."""

    by_scenario_method = {(aggregate.scenario, aggregate.method): aggregate for aggregate in aggregates}
    comparisons: list[MethodComparison] = []
    for aggregate in aggregates:
        if aggregate.method == reference_method:
            continue
        reference = by_scenario_method.get((aggregate.scenario, reference_method))
        if reference is None:
            continue
        comparisons.append(
            MethodComparison(
                scenario=aggregate.scenario,
                method=aggregate.method,
                reference_method=reference_method,
                delta_mean_false_tracks=aggregate.mean_false_tracks - reference.mean_false_tracks,
                delta_mean_false_track_duration=(
                    aggregate.mean_false_track_duration - reference.mean_false_track_duration
                ),
                delta_mean_missed_targets=aggregate.mean_missed_targets - reference.mean_missed_targets,
                delta_mean_gospa_distance=aggregate.mean_gospa_distance - reference.mean_gospa_distance,
                delta_mean_gospa_false_cost=aggregate.mean_gospa_false_cost - reference.mean_gospa_false_cost,
                delta_mean_gospa_missed_cost=aggregate.mean_gospa_missed_cost - reference.mean_gospa_missed_cost,
                delta_mean_merged_gospa_distance=(
                    aggregate.mean_merged_gospa_distance - reference.mean_merged_gospa_distance
                ),
                delta_mean_merged_gospa_false_cost=(
                    aggregate.mean_merged_gospa_false_cost - reference.mean_merged_gospa_false_cost
                ),
                delta_mean_merged_gospa_missed_cost=(
                    aggregate.mean_merged_gospa_missed_cost - reference.mean_merged_gospa_missed_cost
                ),
            )
        )
    return tuple(comparisons)


def compare_to_reference_paired(
    results: Sequence[MethodResult],
    reference_method: str = "oracle",
) -> tuple[PairedMethodComparison, ...]:
    """Compute paired seed-wise method deltas against a reference method."""

    by_key = {(result.scenario, result.method, result.seed): result for result in results}
    groups = sorted({(result.scenario, result.method) for result in results if result.method != reference_method})

    comparisons: list[PairedMethodComparison] = []
    for scenario, method in groups:
        method_rows = sorted(
            (result for result in results if result.scenario == scenario and result.method == method),
            key=lambda result: result.seed,
        )
        pairs = [
            (row, by_key[(scenario, reference_method, row.seed)])
            for row in method_rows
            if (scenario, reference_method, row.seed) in by_key
        ]
        if not pairs:
            continue

        false_tracks = [row.false_tracks - reference.false_tracks for row, reference in pairs]
        false_duration = [row.false_track_duration - reference.false_track_duration for row, reference in pairs]
        missed_targets = [row.missed_targets - reference.missed_targets for row, reference in pairs]
        gospa_distance = [row.gospa_distance - reference.gospa_distance for row, reference in pairs]
        gospa_false = [row.gospa_false_cost - reference.gospa_false_cost for row, reference in pairs]
        gospa_missed = [row.gospa_missed_cost - reference.gospa_missed_cost for row, reference in pairs]
        merged_gospa_distance = [row.merged_gospa_distance - reference.merged_gospa_distance for row, reference in pairs]
        merged_gospa_false = [row.merged_gospa_false_cost - reference.merged_gospa_false_cost for row, reference in pairs]
        merged_gospa_missed = [row.merged_gospa_missed_cost - reference.merged_gospa_missed_cost for row, reference in pairs]

        comparisons.append(
            PairedMethodComparison(
                scenario=scenario,
                method=method,
                reference_method=reference_method,
                num_pairs=len(pairs),
                mean_delta_false_tracks=_mean(false_tracks),
                se_delta_false_tracks=_standard_error(false_tracks),
                mean_delta_false_track_duration=_mean(false_duration),
                se_delta_false_track_duration=_standard_error(false_duration),
                mean_delta_missed_targets=_mean(missed_targets),
                se_delta_missed_targets=_standard_error(missed_targets),
                mean_delta_gospa_distance=_mean(gospa_distance),
                se_delta_gospa_distance=_standard_error(gospa_distance),
                mean_delta_gospa_false_cost=_mean(gospa_false),
                se_delta_gospa_false_cost=_standard_error(gospa_false),
                mean_delta_gospa_missed_cost=_mean(gospa_missed),
                se_delta_gospa_missed_cost=_standard_error(gospa_missed),
                mean_delta_merged_gospa_distance=_mean(merged_gospa_distance),
                se_delta_merged_gospa_distance=_standard_error(merged_gospa_distance),
                mean_delta_merged_gospa_false_cost=_mean(merged_gospa_false),
                se_delta_merged_gospa_false_cost=_standard_error(merged_gospa_false),
                mean_delta_merged_gospa_missed_cost=_mean(merged_gospa_missed),
                se_delta_merged_gospa_missed_cost=_standard_error(merged_gospa_missed),
            )
        )
    return tuple(comparisons)


def format_csv(rows: Iterable[dict[str, object]]) -> str:
    """Format dictionaries as a small CSV string without external dependencies."""

    materialized = list(rows)
    if not materialized:
        return ""

    columns = list(materialized[0])
    lines = [",".join(columns)]
    for row in materialized:
        lines.append(",".join(_format_csv_cell(row[column]) for column in columns))
    return "\n".join(lines)


def format_method_results_csv(results: Sequence[MethodResult]) -> str:
    """Format per-seed method results as CSV."""

    return format_csv(result.to_dict() for result in results)


def format_method_aggregates_csv(aggregates: Sequence[MethodAggregate]) -> str:
    """Format cross-seed aggregates as CSV."""

    return format_csv(aggregate.to_dict() for aggregate in aggregates)


def format_method_comparisons_csv(comparisons: Sequence[MethodComparison]) -> str:
    """Format reference-comparison rows as CSV."""

    return format_csv(comparison.to_dict() for comparison in comparisons)


def format_paired_method_comparisons_csv(comparisons: Sequence[PairedMethodComparison]) -> str:
    """Format paired reference-comparison rows as CSV."""

    return format_csv(comparison.to_dict() for comparison in comparisons)


def _mean_attr(results: Sequence[MethodResult], attr: str) -> float:
    if not results:
        raise ValueError("cannot aggregate an empty result list")
    return _mean(float(getattr(result, attr)) for result in results)


def _mean(values: Iterable[float]) -> float:
    materialized = tuple(float(value) for value in values)
    if not materialized:
        raise ValueError("cannot summarize an empty value list")
    return float(mean(materialized))


def _standard_error(values: Sequence[float]) -> float:
    materialized = tuple(float(value) for value in values)
    if len(materialized) < 2:
        return 0.0
    return float(stdev(materialized) / sqrt(len(materialized)))


def _format_csv_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.8g}"
    text = str(value)
    if any(char in text for char in ',"\n'):
        return '"' + text.replace('"', '""') + '"'
    return text
