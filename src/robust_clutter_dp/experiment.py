"""Minimal end-to-end structured-clutter comparison experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .clutter import GridClutterIntensity, UniformClutterIntensity
from .confirmation import CandidateBirth
from .dp_clutter import OnlineDPGaussianClutterIntensity
from .metrics import ConfirmationRecord, PointObject, confirmation_metrics, gospa_decomposition
from .scoring import BirthModel, MeasurementModel, compete_measurement
from .simulation import SimulationConfig, SimulationRun, StructuredClutterIntensity, simulate_structured_clutter_scene
from .tracklet import TentativeBirthManager, TentativeTracklet, TrackletManagerConfig, tracklet_truth_labels


SUPPORTED_METHODS = ("uniform", "grid", "dp", "oracle")


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the minimal structured-clutter comparison."""

    methods: tuple[str, ...] = SUPPORTED_METHODS
    birth_rate: float = 0.20
    grid_bins: tuple[int, ...] = (8, 8)
    grid_smoothing: float = 0.5
    dp_concentration: float = 0.8
    dp_covariance_floor: float = 0.03
    gospa_cutoff: float = 1.0
    tracklet_config: TrackletManagerConfig = field(
        default_factory=lambda: TrackletManagerConfig(
            prior_existence=0.05,
            birth_probability_threshold=0.20,
            association_distance=0.90,
            dynamic_sigma=0.45,
            max_missed_scans=2,
            min_updates_for_confirmation=3,
            min_motion_span_for_confirmation=0.45,
            min_confirmation_probability=0.75,
            fdr_q=0.15,
        )
    )

    def __post_init__(self) -> None:
        if self.birth_rate < 0:
            raise ValueError("birth_rate must be non-negative")
        if self.gospa_cutoff <= 0:
            raise ValueError("gospa_cutoff must be positive")
        for method in self.methods:
            if method not in SUPPORTED_METHODS:
                raise ValueError(f"unsupported method {method!r}; expected one of {SUPPORTED_METHODS}")
        if any(bin_count <= 0 for bin_count in self.grid_bins):
            raise ValueError("grid_bins must be positive")
        if self.grid_smoothing < 0:
            raise ValueError("grid_smoothing must be non-negative")
        if self.dp_concentration <= 0:
            raise ValueError("dp_concentration must be positive")
        if self.dp_covariance_floor <= 0:
            raise ValueError("dp_covariance_floor must be positive")


@dataclass(frozen=True)
class MethodResult:
    """One method/seed result row."""

    method: str
    seed: int
    confirmed_tracks: int
    true_confirmed_tracks: int
    false_tracks: int
    false_track_duration: int
    missed_targets: int
    gospa_distance: float
    gospa_total_cost: float
    gospa_localization_cost: float
    gospa_missed_cost: float
    gospa_false_cost: float
    posterior_expected_fdr: float
    observed_false_discovery_proportion: float
    existence_brier_score: float

    def to_dict(self) -> dict[str, float | int | str]:
        """Return a CSV/dataframe-friendly representation."""

        return {
            "method": self.method,
            "seed": self.seed,
            "confirmed_tracks": self.confirmed_tracks,
            "true_confirmed_tracks": self.true_confirmed_tracks,
            "false_tracks": self.false_tracks,
            "false_track_duration": self.false_track_duration,
            "missed_targets": self.missed_targets,
            "gospa_distance": self.gospa_distance,
            "gospa_total_cost": self.gospa_total_cost,
            "gospa_localization_cost": self.gospa_localization_cost,
            "gospa_missed_cost": self.gospa_missed_cost,
            "gospa_false_cost": self.gospa_false_cost,
            "posterior_expected_fdr": self.posterior_expected_fdr,
            "observed_false_discovery_proportion": self.observed_false_discovery_proportion,
            "existence_brier_score": self.existence_brier_score,
        }


def run_structured_clutter_comparison(
    seeds: Sequence[int] = (0,),
    simulation_config: SimulationConfig | None = None,
    experiment_config: ExperimentConfig | None = None,
) -> tuple[MethodResult, ...]:
    """Run uniform/grid/DP/oracle clutter comparisons across seeds."""

    sim_config = SimulationConfig() if simulation_config is None else simulation_config
    exp_config = ExperimentConfig() if experiment_config is None else experiment_config

    results: list[MethodResult] = []
    for seed in seeds:
        run = simulate_structured_clutter_scene(sim_config, seed=seed)
        for method in exp_config.methods:
            results.append(run_method(run, method, exp_config))
    return tuple(results)


def run_method(
    run: SimulationRun,
    method: str,
    experiment_config: ExperimentConfig | None = None,
) -> MethodResult:
    """Run one clutter model through the tentative-birth pipeline."""

    exp_config = ExperimentConfig() if experiment_config is None else experiment_config
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported method {method!r}; expected one of {SUPPORTED_METHODS}")

    clutter_model = build_clutter_model(run, method, exp_config)
    measurement_model = MeasurementModel(
        covariance=np.eye(run.config.dimension) * run.config.measurement_std**2,
    )
    birth_model = BirthModel(
        rate=exp_config.birth_rate,
        mean=run.config.center,
        covariance=run.config.broad_birth_covariance,
        detection_probability=run.config.detection_probability,
    )
    manager = TentativeBirthManager(exp_config.tracklet_config)

    for frame in run.frames:
        measurements = [measurement.position for measurement in frame.measurements]
        source_ids = [measurement.source_id for measurement in frame.measurements]
        association_results = [
            compete_measurement(
                measurement=measurement,
                tracks=[],
                birth_model=birth_model,
                measurement_model=measurement_model,
                clutter_model=clutter_model,
            )
            for measurement in measurements
        ]
        manager.process_measurements(
            scan_index=frame.scan_index,
            measurements=measurements,
            association_results=association_results,
            source_ids=source_ids,
        )

    manager.confirm_eligible()
    return summarize_method_result(
        method=method,
        seed=run.seed,
        confirmed=manager.confirmed_tracklets,
        final_truths=tuple(run.frames[-1].truths),
        gospa_cutoff=exp_config.gospa_cutoff,
    )


def build_clutter_model(run: SimulationRun, method: str, config: ExperimentConfig):
    """Construct a clutter intensity model for a method name."""

    if method == "uniform":
        return UniformClutterIntensity(
            rate=run.config.total_clutter_rate,
            volume=run.config.volume,
        )

    if method == "grid":
        calibration = _nonempty_calibration(run)
        return GridClutterIntensity.from_samples(
            samples=calibration,
            bounds=run.config.bounds,
            bins=config.grid_bins,
            total_rate=run.config.total_clutter_rate,
            smoothing=config.grid_smoothing,
        )

    if method == "dp":
        calibration = _nonempty_calibration(run)
        return OnlineDPGaussianClutterIntensity.fit_sugs(
            samples=calibration,
            concentration=config.dp_concentration,
            total_rate=run.config.total_clutter_rate,
            covariance_floor=np.eye(run.config.dimension) * config.dp_covariance_floor,
            max_components=20,
        )

    if method == "oracle":
        return StructuredClutterIntensity(
            bounds=run.config.bounds,
            uniform_rate=run.config.uniform_clutter_rate,
            hotspots=run.config.clutter_hotspots,
        )

    raise ValueError(f"unsupported method {method!r}; expected one of {SUPPORTED_METHODS}")


def summarize_method_result(
    method: str,
    seed: int,
    confirmed: Sequence[TentativeTracklet],
    final_truths,
    gospa_cutoff: float,
) -> MethodResult:
    """Convert confirmed tracklets into paper-facing metrics."""

    truth_labels = tracklet_truth_labels(confirmed)
    records = [
        ConfirmationRecord(
            candidate_id=tracklet.candidate_id,
            existence_probability=tracklet.existence_probability,
            is_target=truth_labels[tracklet.candidate_id],
            accepted=True,
        )
        for tracklet in confirmed
    ]
    confirmation = confirmation_metrics(records)

    estimates = [
        PointObject(tracklet.candidate_id, tracklet.last_position)
        for tracklet in confirmed
    ]
    truths = [
        PointObject(truth.target_id, truth.position)
        for truth in final_truths
    ]
    gospa = gospa_decomposition(estimates, truths, cutoff=gospa_cutoff)

    false_tracklets = [tracklet for tracklet in confirmed if not truth_labels[tracklet.candidate_id]]
    return MethodResult(
        method=method,
        seed=seed,
        confirmed_tracks=len(confirmed),
        true_confirmed_tracks=sum(truth_labels.values()),
        false_tracks=len(false_tracklets),
        false_track_duration=sum(tracklet.duration for tracklet in false_tracklets),
        missed_targets=gospa.num_missed,
        gospa_distance=gospa.distance,
        gospa_total_cost=gospa.total_cost,
        gospa_localization_cost=gospa.localization_cost,
        gospa_missed_cost=gospa.missed_cost,
        gospa_false_cost=gospa.false_cost,
        posterior_expected_fdr=confirmation.posterior_expected_fdr,
        observed_false_discovery_proportion=confirmation.false_discovery_proportion,
        existence_brier_score=confirmation.brier_score,
    )


def _nonempty_calibration(run: SimulationRun) -> np.ndarray:
    if run.calibration_measurements.size == 0:
        return np.array([run.config.center], dtype=float)
    return run.calibration_measurements
