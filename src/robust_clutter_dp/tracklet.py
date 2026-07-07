"""Tentative-birth tracklet management and FDR confirmation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .confirmation import CandidateBirth, posterior_existence_from_log_bayes_factor, select_by_bayesian_fdr
from .scoring import AssociationResult, gaussian_logpdf


@dataclass(frozen=True)
class TentativeTracklet:
    """A sequential candidate birth accumulated from measurement evidence."""

    candidate_id: str
    first_scan: int
    last_scan: int
    measurements: tuple[NDArray[np.float64], ...]
    source_ids: tuple[str | None, ...]
    log_bayes_factor: float
    existence_probability: float

    def __post_init__(self) -> None:
        if not self.measurements:
            raise ValueError("tracklet must contain at least one measurement")
        if len(self.measurements) != len(self.source_ids):
            raise ValueError("measurements and source_ids must have the same length")
        if self.first_scan > self.last_scan:
            raise ValueError("first_scan must not exceed last_scan")
        if not 0.0 <= self.existence_probability <= 1.0:
            raise ValueError("existence_probability must be in [0, 1]")

    @property
    def num_measurements(self) -> int:
        return len(self.measurements)

    @property
    def last_position(self) -> NDArray[np.float64]:
        return np.asarray(self.measurements[-1], dtype=float)

    @property
    def first_position(self) -> NDArray[np.float64]:
        return np.asarray(self.measurements[0], dtype=float)

    @property
    def duration(self) -> int:
        return self.last_scan - self.first_scan + 1

    @property
    def motion_span(self) -> float:
        return float(np.linalg.norm(self.last_position - self.first_position))

    @property
    def dominant_source_id(self) -> str | None:
        counts: dict[str, int] = {}
        for source_id in self.source_ids:
            if source_id is None:
                continue
            counts[source_id] = counts.get(source_id, 0) + 1
        if not counts:
            return None
        return max(counts, key=counts.__getitem__)

    @property
    def dominant_source_fraction(self) -> float:
        dominant = self.dominant_source_id
        if dominant is None:
            return 0.0
        return sum(source_id == dominant for source_id in self.source_ids) / len(self.source_ids)

    def predicted_position(self) -> NDArray[np.float64]:
        """Constant-velocity one-step prediction for tracklet association."""

        if len(self.measurements) < 2:
            return self.last_position
        return self.last_position + (self.last_position - np.asarray(self.measurements[-2], dtype=float))


@dataclass(frozen=True)
class TrackletManagerConfig:
    """Configuration for online tentative-birth management."""

    prior_existence: float = 0.05
    birth_probability_threshold: float = 0.25
    association_distance: float = 0.75
    dynamic_sigma: float = 0.35
    max_missed_scans: int = 2
    min_updates_for_confirmation: int = 3
    min_motion_span_for_confirmation: float = 0.0
    min_confirmation_probability: float = 0.0
    fdr_q: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 <= self.prior_existence <= 1.0:
            raise ValueError("prior_existence must be in [0, 1]")
        if not 0.0 <= self.birth_probability_threshold <= 1.0:
            raise ValueError("birth_probability_threshold must be in [0, 1]")
        if self.association_distance <= 0:
            raise ValueError("association_distance must be positive")
        if self.dynamic_sigma <= 0:
            raise ValueError("dynamic_sigma must be positive")
        if self.max_missed_scans < 0:
            raise ValueError("max_missed_scans must be non-negative")
        if self.min_updates_for_confirmation <= 0:
            raise ValueError("min_updates_for_confirmation must be positive")
        if self.min_motion_span_for_confirmation < 0:
            raise ValueError("min_motion_span_for_confirmation must be non-negative")
        if not 0.0 <= self.min_confirmation_probability <= 1.0:
            raise ValueError("min_confirmation_probability must be in [0, 1]")
        if not 0.0 <= self.fdr_q <= 1.0:
            raise ValueError("fdr_q must be in [0, 1]")


@dataclass(frozen=True)
class TrackletStepSummary:
    """Summary of one manager update."""

    scan_index: int
    created: int
    updated: int
    confirmed_ids: tuple[str, ...]
    pruned_ids: tuple[str, ...]


class TentativeBirthManager:
    """Online manager for candidate births.

    The manager is intentionally simple. It is not a full multi-object filter; it
    is the missing glue between per-measurement birth-vs-clutter scores and the
    posterior-FDR confirmation rule.
    """

    def __init__(self, config: TrackletManagerConfig | None = None) -> None:
        self.config = TrackletManagerConfig() if config is None else config
        self._active: dict[str, TentativeTracklet] = {}
        self._confirmed: dict[str, TentativeTracklet] = {}
        self._next_id = 1

    @property
    def active_tracklets(self) -> tuple[TentativeTracklet, ...]:
        return tuple(self._active.values())

    @property
    def confirmed_tracklets(self) -> tuple[TentativeTracklet, ...]:
        return tuple(self._confirmed.values())

    def process_measurements(
        self,
        scan_index: int,
        measurements: Sequence[ArrayLike],
        association_results: Sequence[AssociationResult],
        source_ids: Sequence[str | None] | None = None,
    ) -> TrackletStepSummary:
        """Update tentative births from one scan of scored measurements."""

        if scan_index < 0:
            raise ValueError("scan_index must be non-negative")
        if len(measurements) != len(association_results):
            raise ValueError("measurements and association_results must have the same length")
        if source_ids is None:
            source_ids = [None] * len(measurements)
        if len(source_ids) != len(measurements):
            raise ValueError("source_ids length must match measurements")

        created = 0
        updated = 0

        for measurement, result, source_id in zip(measurements, association_results, source_ids, strict=True):
            z = _as_vector(measurement, "measurement")
            if result.birth_probability < self.config.birth_probability_threshold:
                continue

            candidate_id = self._nearest_active_candidate(z, scan_index)
            if candidate_id is None:
                self._active[self._new_candidate_id()] = self._create_tracklet(scan_index, z, result, source_id)
                created += 1
            else:
                self._active[candidate_id] = self._update_tracklet(
                    self._active[candidate_id],
                    scan_index,
                    z,
                    result,
                    source_id,
                )
                updated += 1

        confirmed_ids = self.confirm_eligible()
        pruned_ids = self.prune_stale(scan_index)
        return TrackletStepSummary(
            scan_index=scan_index,
            created=created,
            updated=updated,
            confirmed_ids=confirmed_ids,
            pruned_ids=pruned_ids,
        )

    def confirm_eligible(self) -> tuple[str, ...]:
        """Confirm eligible tentative births under posterior-FDR control."""

        eligible = [tracklet for tracklet in self._active.values() if self._is_eligible(tracklet)]
        candidates = [
            CandidateBirth(tracklet.candidate_id, tracklet.existence_probability)
            for tracklet in eligible
        ]
        decision = select_by_bayesian_fdr(
            candidates,
            q=self.config.fdr_q,
            min_probability=self.config.min_confirmation_probability,
        )
        accepted = set(decision.accepted_ids)
        for candidate_id in accepted:
            self._confirmed[candidate_id] = self._active.pop(candidate_id)
        return tuple(sorted(accepted))

    def prune_stale(self, scan_index: int) -> tuple[str, ...]:
        """Drop inactive tentative births that have not been updated recently."""

        stale_ids = tuple(
            candidate_id
            for candidate_id, tracklet in self._active.items()
            if scan_index - tracklet.last_scan > self.config.max_missed_scans
        )
        for candidate_id in stale_ids:
            self._active.pop(candidate_id)
        return stale_ids

    def _is_eligible(self, tracklet: TentativeTracklet) -> bool:
        return (
            tracklet.num_measurements >= self.config.min_updates_for_confirmation
            and tracklet.motion_span >= self.config.min_motion_span_for_confirmation
        )

    def _nearest_active_candidate(self, measurement: NDArray[np.float64], scan_index: int) -> str | None:
        best_id: str | None = None
        best_distance = self.config.association_distance
        for candidate_id, tracklet in self._active.items():
            if scan_index - tracklet.last_scan > self.config.max_missed_scans + 1:
                continue
            distance = float(np.linalg.norm(measurement - tracklet.predicted_position()))
            if distance <= best_distance:
                best_distance = distance
                best_id = candidate_id
        return best_id

    def _create_tracklet(
        self,
        scan_index: int,
        measurement: NDArray[np.float64],
        result: AssociationResult,
        source_id: str | None,
    ) -> TentativeTracklet:
        log_bayes_factor = birth_vs_clutter_log_bayes_factor(result)
        return TentativeTracklet(
            candidate_id=f"birth-{self._next_id - 1}",
            first_scan=scan_index,
            last_scan=scan_index,
            measurements=(measurement,),
            source_ids=(source_id,),
            log_bayes_factor=log_bayes_factor,
            existence_probability=posterior_existence_from_log_bayes_factor(
                log_bayes_factor,
                self.config.prior_existence,
            ),
        )

    def _update_tracklet(
        self,
        tracklet: TentativeTracklet,
        scan_index: int,
        measurement: NDArray[np.float64],
        result: AssociationResult,
        source_id: str | None,
    ) -> TentativeTracklet:
        dynamic_covariance = np.eye(measurement.size) * self.config.dynamic_sigma**2
        log_target = gaussian_logpdf(measurement, tracklet.predicted_position(), dynamic_covariance)
        log_clutter = result.log_weights["clutter"]
        log_bayes_factor = tracklet.log_bayes_factor + log_target - log_clutter
        return TentativeTracklet(
            candidate_id=tracklet.candidate_id,
            first_scan=tracklet.first_scan,
            last_scan=scan_index,
            measurements=tracklet.measurements + (measurement,),
            source_ids=tracklet.source_ids + (source_id,),
            log_bayes_factor=log_bayes_factor,
            existence_probability=posterior_existence_from_log_bayes_factor(
                log_bayes_factor,
                self.config.prior_existence,
            ),
        )

    def _new_candidate_id(self) -> str:
        candidate_id = f"birth-{self._next_id}"
        self._next_id += 1
        return candidate_id


def birth_vs_clutter_log_bayes_factor(result: AssociationResult) -> float:
    """Return log p(z | birth) - log p(z | clutter) from an association result."""

    return float(result.log_weights["birth"] - result.log_weights["clutter"])


def tracklet_truth_labels(tracklets: Sequence[TentativeTracklet]) -> Mapping[str, bool]:
    """Simulation helper: classify confirmed tracklets by source annotations."""

    return {
        tracklet.candidate_id: tracklet.dominant_source_id is not None and tracklet.dominant_source_fraction >= 0.5
        for tracklet in tracklets
    }


def _as_vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr
