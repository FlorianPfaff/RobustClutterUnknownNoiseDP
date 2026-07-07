"""Reporting helpers for structured-clutter experiments."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
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
    mean_time_to_confirm: float
    mean_true_time_to_confirm: float
    mean_missed_targets: float
    mean_gospa_distance: float
    mean_gospa_localization_cost: float
    mean_gospa_missed_cost: float
    mean_gospa_false_cost: float
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
            "mean_time_to_confirm": self.mean_time_to_confirm,
            "mean_true_time_to_confirm": self.mean_true_time_to_confirm,
            "mean_missed_targets": self.mean_missed_targets,
            "mean_gospa_distance": self.mean_gospa_distance,
            "mean_gospa_localization_cost": self.mean_gospa_localization_cost,
            "mean_gospa_missed_cost": self.mean_gospa_missed_cost,
            "mean_gospa_false_cost": self.mean_gospa_false_cost,
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
        }


def aggregate_method_results(results: Sequence[MethodResult]) -> tuple[MethodAggregate, ...]:
    """Aggregate method results across seeds by scenario and method.

    The output is sorted by scenario and method for deterministic reporting.
    """

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
                mean_time_to_confirm=_mean_attr(method_results, "mean_time_to_confirm"),
                mean_true_time_to_confirm=_mean_attr(method_results, "mean_true_time_to_confirm"),
                mean_missed_targets=_mean_attr(method_results, "missed_targets"),
                mean_gospa_distance=_mean_attr(method_results, "gospa_distance"),
                mean_gospa_localization_cost=_mean_attr(method_results, "gospa_localization_cost"),
                mean_gospa_missed_cost=_mean_attr(method_results, "gospa_missed_cost"),
                mean_gospa_false_cost=_mean_attr(method_results, "gospa_false_cost"),
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
    """Compare method aggregates against a reference method within each scenario.

    Positive deltas mean the method is worse than the reference on that metric;
    negative deltas mean lower error than the reference. The reference row itself
    is omitted from the returned comparisons.
    """

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


def _mean_attr(results: Sequence[MethodResult], attr: str) -> float:
    if not results:
        raise ValueError("cannot aggregate an empty result list")
    return float(mean(float(getattr(result, attr)) for result in results))


def _format_csv_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.8g}"
    text = str(value)
    if any(char in text for char in ',"\n'):
        return '"' + text.replace('"', '""') + '"'
    return text
