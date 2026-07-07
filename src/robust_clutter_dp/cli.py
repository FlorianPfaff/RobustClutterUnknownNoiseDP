"""Command-line interface for the structured-clutter benchmark."""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Sequence

from .experiment import ExperimentConfig, SUPPORTED_METHODS, run_named_scenarios_comparison
from .reporting import (
    aggregate_method_results,
    compare_to_reference,
    format_method_aggregates_csv,
    format_method_comparisons_csv,
    format_method_results_csv,
)
from .scenarios import SCENARIO_NAMES


OUTPUT_MODES = ("all", "raw", "summary", "comparison")


def build_parser() -> argparse.ArgumentParser:
    """Build the experiment CLI parser."""

    parser = argparse.ArgumentParser(
        prog="robust-clutter-dp-experiment",
        description="Run the toy structured-clutter benchmark.",
    )
    parser.add_argument(
        "--scenarios",
        default="hotspot,no_hotspot_control,near_hotspot_crossing",
        help=f"Comma-separated scenario names or 'all'. Available: {', '.join(SCENARIO_NAMES)}.",
    )
    parser.add_argument(
        "--methods",
        default="uniform,grid,dp,oracle",
        help=f"Comma-separated methods. Available: {', '.join(SUPPORTED_METHODS)}.",
    )
    parser.add_argument(
        "--seeds",
        default="0:5",
        help="Comma-separated integers or Python-like range start:stop[:step], e.g. '0:20' or '0,3,7'.",
    )
    parser.add_argument(
        "--output",
        choices=OUTPUT_MODES,
        default="all",
        help="Which CSV table(s) to print.",
    )
    parser.add_argument(
        "--reference-method",
        default="oracle",
        help="Reference method for comparison deltas.",
    )
    parser.add_argument(
        "--birth-rate",
        type=float,
        default=0.20,
        help="Birth intensity used by the toy benchmark.",
    )
    parser.add_argument(
        "--birth-threshold",
        type=float,
        default=0.20,
        help="Minimum posterior birth probability needed to create/update a tentative tracklet.",
    )
    parser.add_argument(
        "--min-confirmation-probability",
        type=float,
        default=0.75,
        help="Minimum posterior existence probability for FDR-based confirmation.",
    )
    parser.add_argument(
        "--fdr-q",
        type=float,
        default=0.15,
        help="Posterior expected FDR level for confirming tentative births.",
    )
    return parser


def parse_seed_spec(spec: str) -> tuple[int, ...]:
    """Parse a seed specification.

    Supported forms are comma-separated integers (``"0,3,7"``) and
    ``start:stop[:step]`` ranges (``"0:10"`` or ``"0:10:2"``).
    """

    text = spec.strip()
    if not text:
        raise ValueError("seed specification must not be empty")

    if ":" in text:
        parts = text.split(":")
        if len(parts) not in (2, 3):
            raise ValueError("range seed specification must be start:stop[:step]")
        start = int(parts[0]) if parts[0] else 0
        stop = int(parts[1])
        step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
        if step == 0:
            raise ValueError("seed range step must not be zero")
        seeds = tuple(range(start, stop, step))
    else:
        seeds = tuple(int(part.strip()) for part in text.split(",") if part.strip())

    if not seeds:
        raise ValueError("seed specification produced no seeds")
    return seeds


def parse_scenario_spec(spec: str) -> tuple[str, ...]:
    """Parse scenario names from a comma-separated string or ``all``."""

    names = SCENARIO_NAMES if spec.strip() == "all" else _parse_name_list(spec)
    unknown = tuple(name for name in names if name not in SCENARIO_NAMES)
    if unknown:
        raise ValueError(f"unknown scenario(s): {', '.join(unknown)}")
    return tuple(names)


def parse_method_spec(spec: str) -> tuple[str, ...]:
    """Parse method names from a comma-separated string or ``all``."""

    names = SUPPORTED_METHODS if spec.strip() == "all" else _parse_name_list(spec)
    unknown = tuple(name for name in names if name not in SUPPORTED_METHODS)
    if unknown:
        raise ValueError(f"unknown method(s): {', '.join(unknown)}")
    return tuple(names)


def make_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
    """Create an ``ExperimentConfig`` from parsed CLI arguments."""

    base = ExperimentConfig(methods=parse_method_spec(args.methods), birth_rate=args.birth_rate)
    tracklet_config = replace(
        base.tracklet_config,
        birth_probability_threshold=args.birth_threshold,
        min_confirmation_probability=args.min_confirmation_probability,
        fdr_q=args.fdr_q,
    )
    return replace(base, tracklet_config=tracklet_config)


def run_cli(args: argparse.Namespace) -> str:
    """Run the benchmark and return the requested CSV output string."""

    scenarios = parse_scenario_spec(args.scenarios)
    seeds = parse_seed_spec(args.seeds)
    experiment_config = make_experiment_config(args)

    results = run_named_scenarios_comparison(
        scenario_names=scenarios,
        seeds=seeds,
        experiment_config=experiment_config,
    )
    aggregates = aggregate_method_results(results)
    comparisons = compare_to_reference(aggregates, reference_method=args.reference_method)

    sections: list[str] = []
    if args.output in ("all", "raw"):
        sections.extend(["# raw per-seed results", format_method_results_csv(results)])
    if args.output in ("all", "summary"):
        sections.extend(["# cross-seed scenario/method summary", format_method_aggregates_csv(aggregates)])
    if args.output in ("all", "comparison"):
        sections.extend(["# deltas versus reference clutter model", format_method_comparisons_csv(comparisons)])
    return "\n\n".join(section for section in sections if section)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = run_cli(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2
    if output:
        print(output)
    return 0


def _parse_name_list(spec: str) -> tuple[str, ...]:
    names = tuple(part.strip() for part in spec.split(",") if part.strip())
    if not names:
        raise ValueError("name list must not be empty")
    return names


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
