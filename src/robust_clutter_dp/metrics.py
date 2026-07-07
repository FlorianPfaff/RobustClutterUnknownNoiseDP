"""False-track-focused evaluation metrics.

The first paper should not be evaluated with localization RMSE alone. This
module provides small dependency-free primitives for the quantities that matter
for the proposed method: decomposed point-set error and posterior calibration of
track-confirmation decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import inf
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class PointObject:
    """A point target or point estimate used by set-distance metrics."""

    object_id: str
    position: ArrayLike


@dataclass(frozen=True)
class GospaMatch:
    """One matched estimate/truth pair in a GOSPA decomposition."""

    estimate_id: str
    truth_id: str
    distance: float


@dataclass(frozen=True)
class GospaDecomposition:
    """GOSPA-style error decomposition for point objects.

    Costs are reported before taking the final ``1 / p`` root. The ``distance``
    field is the rooted GOSPA value.
    """

    p: float
    cutoff: float
    alpha: float
    distance: float
    total_cost: float
    localization_cost: float
    missed_cost: float
    false_cost: float
    matches: tuple[GospaMatch, ...]
    missed_truth_ids: tuple[str, ...]
    false_estimate_ids: tuple[str, ...]

    @property
    def num_matches(self) -> int:
        return len(self.matches)

    @property
    def num_missed(self) -> int:
        return len(self.missed_truth_ids)

    @property
    def num_false(self) -> int:
        return len(self.false_estimate_ids)


@dataclass(frozen=True)
class ConfirmationRecord:
    """Observed outcome for one tentative-birth confirmation decision."""

    candidate_id: str
    existence_probability: float
    is_target: bool
    accepted: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.existence_probability <= 1.0:
            raise ValueError("existence_probability must be in [0, 1]")


@dataclass(frozen=True)
class ConfirmationMetrics:
    """Summary statistics for posterior-existence confirmation decisions."""

    accepted_count: int
    false_accepted_count: int
    true_accepted_count: int
    false_discovery_proportion: float
    posterior_expected_fdr: float
    brier_score: float


def gospa_decomposition(
    estimates: Sequence[PointObject],
    truths: Sequence[PointObject],
    cutoff: float,
    p: float = 2.0,
    alpha: float = 2.0,
) -> GospaDecomposition:
    """Compute a GOSPA-style decomposition for point estimates.

    This implementation is exact for the supplied point sets and uses a bitmask
    dynamic program. It is intended for small-to-moderate experiment outputs and
    test scenarios, not for very large dense assignment problems.
    """

    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    if p < 1:
        raise ValueError("p must be at least 1")
    if not 0.0 < alpha <= 2.0:
        raise ValueError("alpha must be in (0, 2]")
    if len(truths) > 22:
        raise ValueError("at most 22 truth objects are supported by the exact bitmask solver")

    estimate_positions = [_as_position(estimate.position, "estimate position") for estimate in estimates]
    truth_positions = [_as_position(truth.position, "truth position") for truth in truths]
    _validate_dimensions(estimate_positions, truth_positions)

    n_estimates = len(estimates)
    n_truths = len(truths)
    cardinality_penalty = cutoff**p / alpha

    if n_estimates == 0:
        missed_cost = cardinality_penalty * n_truths
        return GospaDecomposition(
            p=float(p),
            cutoff=float(cutoff),
            alpha=float(alpha),
            distance=float(missed_cost ** (1.0 / p)),
            total_cost=float(missed_cost),
            localization_cost=0.0,
            missed_cost=float(missed_cost),
            false_cost=0.0,
            matches=(),
            missed_truth_ids=tuple(truth.object_id for truth in truths),
            false_estimate_ids=(),
        )

    if n_truths == 0:
        false_cost = cardinality_penalty * n_estimates
        return GospaDecomposition(
            p=float(p),
            cutoff=float(cutoff),
            alpha=float(alpha),
            distance=float(false_cost ** (1.0 / p)),
            total_cost=float(false_cost),
            localization_cost=0.0,
            missed_cost=0.0,
            false_cost=float(false_cost),
            matches=(),
            missed_truth_ids=(),
            false_estimate_ids=tuple(estimate.object_id for estimate in estimates),
        )

    distances = np.empty((n_estimates, n_truths), dtype=float)
    for estimate_index, estimate_position in enumerate(estimate_positions):
        for truth_index, truth_position in enumerate(truth_positions):
            distances[estimate_index, truth_index] = float(np.linalg.norm(estimate_position - truth_position))

    @lru_cache(maxsize=None)
    def solve(estimate_index: int, used_truth_mask: int) -> tuple[float, tuple[tuple[int, int], ...]]:
        if estimate_index == n_estimates:
            missed = n_truths - used_truth_mask.bit_count()
            return cardinality_penalty * missed, ()

        skipped_cost, skipped_pairs = solve(estimate_index + 1, used_truth_mask)
        best_cost = cardinality_penalty + skipped_cost
        best_pairs = skipped_pairs

        for truth_index in range(n_truths):
            if used_truth_mask & (1 << truth_index):
                continue
            distance = distances[estimate_index, truth_index]
            if distance >= cutoff:
                continue
            tail_cost, tail_pairs = solve(estimate_index + 1, used_truth_mask | (1 << truth_index))
            candidate_cost = distance**p + tail_cost
            if candidate_cost < best_cost - 1e-12:
                best_cost = candidate_cost
                best_pairs = ((estimate_index, truth_index),) + tail_pairs

        return best_cost, best_pairs

    total_cost, match_indices = solve(0, 0)
    matched_estimates = {estimate_index for estimate_index, _ in match_indices}
    matched_truths = {truth_index for _, truth_index in match_indices}

    matches = tuple(
        GospaMatch(
            estimate_id=estimates[estimate_index].object_id,
            truth_id=truths[truth_index].object_id,
            distance=float(distances[estimate_index, truth_index]),
        )
        for estimate_index, truth_index in match_indices
    )
    localization_cost = float(sum(match.distance**p for match in matches))
    false_estimate_ids = tuple(
        estimate.object_id for index, estimate in enumerate(estimates) if index not in matched_estimates
    )
    missed_truth_ids = tuple(truth.object_id for index, truth in enumerate(truths) if index not in matched_truths)
    false_cost = float(cardinality_penalty * len(false_estimate_ids))
    missed_cost = float(cardinality_penalty * len(missed_truth_ids))
    total_cost = float(localization_cost + false_cost + missed_cost)

    return GospaDecomposition(
        p=float(p),
        cutoff=float(cutoff),
        alpha=float(alpha),
        distance=float(total_cost ** (1.0 / p)),
        total_cost=total_cost,
        localization_cost=localization_cost,
        missed_cost=missed_cost,
        false_cost=false_cost,
        matches=matches,
        missed_truth_ids=missed_truth_ids,
        false_estimate_ids=false_estimate_ids,
    )


def confirmation_metrics(records: Sequence[ConfirmationRecord]) -> ConfirmationMetrics:
    """Compute observed and posterior-expected FDR diagnostics.

    ``posterior_expected_fdr`` is the empirical counterpart of the Bayesian FDR
    rule used during confirmation: the average posterior false probability among
    accepted candidates. ``false_discovery_proportion`` uses ground-truth labels
    and is available only in simulation or annotated data.
    """

    accepted = [record for record in records if record.accepted]
    accepted_count = len(accepted)
    false_accepted_count = sum(not record.is_target for record in accepted)
    true_accepted_count = accepted_count - false_accepted_count

    if accepted_count == 0:
        false_discovery_proportion = 0.0
        posterior_expected_fdr = 0.0
    else:
        false_discovery_proportion = false_accepted_count / accepted_count
        posterior_expected_fdr = sum(1.0 - record.existence_probability for record in accepted) / accepted_count

    if records:
        brier_score = sum(
            (record.existence_probability - float(record.is_target)) ** 2 for record in records
        ) / len(records)
    else:
        brier_score = 0.0

    return ConfirmationMetrics(
        accepted_count=accepted_count,
        false_accepted_count=false_accepted_count,
        true_accepted_count=true_accepted_count,
        false_discovery_proportion=float(false_discovery_proportion),
        posterior_expected_fdr=float(posterior_expected_fdr),
        brier_score=float(brier_score),
    )


def _as_position(value: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_dimensions(
    estimate_positions: Sequence[NDArray[np.float64]],
    truth_positions: Sequence[NDArray[np.float64]],
) -> None:
    dimension = None
    for position in (*estimate_positions, *truth_positions):
        if dimension is None:
            dimension = position.size
        elif position.size != dimension:
            raise ValueError("all positions must have the same dimension")
