"""Posterior-existence and false-discovery control utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, inf, log
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class CandidateBirth:
    """Tentative birth hypothesis with posterior existence probability."""

    candidate_id: str
    existence_probability: float
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.existence_probability <= 1.0:
            raise ValueError("existence_probability must be in [0, 1]")


@dataclass(frozen=True)
class ConfirmationDecision:
    """Result of Bayesian FDR-based track confirmation."""

    accepted_ids: tuple[str, ...]
    threshold: float
    estimated_fdr: float


def posterior_existence_from_log_bayes_factor(
    log_bayes_factor: float,
    prior_existence: float,
) -> float:
    """Update a Bernoulli existence probability from a log Bayes factor.

    ``log_bayes_factor`` is log p(data | target) - log p(data | clutter).
    """

    if not 0.0 <= prior_existence <= 1.0:
        raise ValueError("prior_existence must be in [0, 1]")
    if prior_existence == 0.0:
        return 0.0
    if prior_existence == 1.0:
        return 1.0

    log_prior_odds = log(prior_existence) - log1p_negative(prior_existence)
    log_posterior_odds = log_prior_odds + log_bayes_factor
    return logistic_from_log_odds(log_posterior_odds)


def estimate_bayesian_fdr(candidates: Iterable[CandidateBirth]) -> float:
    """Estimate posterior false-discovery rate for an accepted candidate set."""

    accepted = tuple(candidates)
    if not accepted:
        return 0.0
    expected_false = sum(1.0 - c.existence_probability for c in accepted)
    return float(expected_false / len(accepted))


def select_by_bayesian_fdr(
    candidates: Sequence[CandidateBirth],
    q: float,
    min_probability: float = 0.0,
) -> ConfirmationDecision:
    """Accept as many tentative births as possible subject to posterior FDR.

    Candidates are sorted by decreasing posterior existence probability. The
    largest prefix whose posterior expected false-discovery proportion is at
    most ``q`` is accepted.
    """

    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    if not 0.0 <= min_probability <= 1.0:
        raise ValueError("min_probability must be in [0, 1]")

    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: candidate.existence_probability,
        reverse=True,
    )

    accepted: list[CandidateBirth] = []
    cumulative_expected_false = 0.0
    best_count = 0
    best_fdr = 0.0

    for index, candidate in enumerate(sorted_candidates, start=1):
        if candidate.existence_probability < min_probability:
            break
        accepted.append(candidate)
        cumulative_expected_false += 1.0 - candidate.existence_probability
        current_fdr = cumulative_expected_false / index
        if current_fdr <= q:
            best_count = index
            best_fdr = current_fdr

    if best_count == 0:
        return ConfirmationDecision(accepted_ids=(), threshold=inf, estimated_fdr=0.0)

    final = tuple(accepted[:best_count])
    threshold = final[-1].existence_probability
    return ConfirmationDecision(
        accepted_ids=tuple(candidate.candidate_id for candidate in final),
        threshold=float(threshold),
        estimated_fdr=float(best_fdr),
    )


def logistic_from_log_odds(log_odds: float) -> float:
    """Stable logistic transform from log odds."""

    if log_odds >= 0:
        return float(1.0 / (1.0 + exp(-log_odds)))
    odds = exp(log_odds)
    return float(odds / (1.0 + odds))


def log1p_negative(value: float) -> float:
    """Return log(1 - value) with clear validation."""

    if not 0.0 <= value <= 1.0:
        raise ValueError("value must be in [0, 1]")
    if value == 1.0:
        return -inf
    # math.log1p gives better precision when value is close to zero.
    from math import log1p

    return log1p(-value)
