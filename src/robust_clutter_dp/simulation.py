"""Small synthetic scenes for structured-clutter experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .scoring import gaussian_logpdf, logsumexp


Bounds = tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class LinearTarget:
    """Constant-velocity point target used by the toy simulator."""

    target_id: str
    initial_position: ArrayLike
    velocity: ArrayLike

    def __post_init__(self) -> None:
        initial_position = _as_vector(self.initial_position, "initial_position")
        velocity = _as_vector(self.velocity, "velocity")
        if initial_position.size != velocity.size:
            raise ValueError("initial_position and velocity must have the same dimension")

    def position_at(self, scan_index: int) -> NDArray[np.float64]:
        if scan_index < 0:
            raise ValueError("scan_index must be non-negative")
        return _as_vector(self.initial_position, "initial_position") + scan_index * _as_vector(
            self.velocity, "velocity"
        )


@dataclass(frozen=True)
class ClutterHotspot:
    """Gaussian clutter source with a Poisson count per scan."""

    mean: ArrayLike
    covariance: ArrayLike
    rate: float

    def __post_init__(self) -> None:
        mean = _as_vector(self.mean, "hotspot mean")
        covariance = _as_square_matrix(self.covariance, "hotspot covariance")
        if covariance.shape[0] != mean.size:
            raise ValueError("hotspot mean and covariance dimensions disagree")
        if self.rate < 0:
            raise ValueError("hotspot rate must be non-negative")
        sign, _ = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ValueError("hotspot covariance must be positive definite")


@dataclass(frozen=True)
class SimulationConfig:
    """Configuration for a 2D structured-clutter toy scene."""

    num_scans: int = 20
    bounds: Bounds = ((0.0, 10.0), (0.0, 10.0))
    targets: tuple[LinearTarget, ...] = field(
        default_factory=lambda: (
            LinearTarget("target-1", np.array([1.0, 8.0]), np.array([0.35, -0.18])),
        )
    )
    detection_probability: float = 0.95
    measurement_std: float = 0.12
    uniform_clutter_rate: float = 1.0
    clutter_hotspots: tuple[ClutterHotspot, ...] = field(
        default_factory=lambda: (
            ClutterHotspot(np.array([2.0, 2.0]), np.eye(2) * 0.06, rate=5.0),
        )
    )
    calibration_scans: int = 12

    def __post_init__(self) -> None:
        if self.num_scans <= 0:
            raise ValueError("num_scans must be positive")
        _validate_bounds(self.bounds)
        dim = len(self.bounds)
        if not 0.0 <= self.detection_probability <= 1.0:
            raise ValueError("detection_probability must be in [0, 1]")
        if self.measurement_std <= 0:
            raise ValueError("measurement_std must be positive")
        if self.uniform_clutter_rate < 0:
            raise ValueError("uniform_clutter_rate must be non-negative")
        if self.calibration_scans < 0:
            raise ValueError("calibration_scans must be non-negative")
        for target in self.targets:
            if target.position_at(0).size != dim:
                raise ValueError("all targets must match bounds dimension")
        for hotspot in self.clutter_hotspots:
            if _as_vector(hotspot.mean, "hotspot mean").size != dim:
                raise ValueError("all hotspots must match bounds dimension")

    @property
    def dimension(self) -> int:
        return len(self.bounds)

    @property
    def volume(self) -> float:
        volume = 1.0
        for low, high in self.bounds:
            volume *= high - low
        return float(volume)

    @property
    def total_clutter_rate(self) -> float:
        return float(self.uniform_clutter_rate + sum(hotspot.rate for hotspot in self.clutter_hotspots))

    @property
    def center(self) -> NDArray[np.float64]:
        return np.array([(low + high) / 2.0 for low, high in self.bounds], dtype=float)

    @property
    def broad_birth_covariance(self) -> NDArray[np.float64]:
        ranges = np.array([high - low for low, high in self.bounds], dtype=float)
        return np.diag((ranges / 2.0) ** 2)


@dataclass(frozen=True)
class SimulatedTruth:
    """Ground-truth point target at one scan."""

    target_id: str
    position: NDArray[np.float64]


@dataclass(frozen=True)
class SimulatedMeasurement:
    """Measurement with simulation-only source annotation."""

    position: NDArray[np.float64]
    source_id: str | None

    @property
    def is_target_generated(self) -> bool:
        return self.source_id is not None


@dataclass(frozen=True)
class SimulationFrame:
    """One scan of truths and measurements."""

    scan_index: int
    truths: tuple[SimulatedTruth, ...]
    measurements: tuple[SimulatedMeasurement, ...]


@dataclass(frozen=True)
class SimulationRun:
    """Complete synthetic run including clutter-only calibration samples."""

    config: SimulationConfig
    seed: int
    frames: tuple[SimulationFrame, ...]
    calibration_measurements: NDArray[np.float64]


@dataclass(frozen=True)
class StructuredClutterIntensity:
    """Oracle clutter intensity for the simulator's known clutter process."""

    bounds: Bounds
    uniform_rate: float
    hotspots: tuple[ClutterHotspot, ...]
    min_intensity: float = 1e-12

    def __post_init__(self) -> None:
        _validate_bounds(self.bounds)
        if self.uniform_rate < 0:
            raise ValueError("uniform_rate must be non-negative")
        if self.min_intensity < 0:
            raise ValueError("min_intensity must be non-negative")

    @property
    def volume(self) -> float:
        volume = 1.0
        for low, high in self.bounds:
            volume *= high - low
        return float(volume)

    def log_intensity(self, measurement: ArrayLike) -> float:
        z = _as_vector(measurement, "measurement")
        if z.size != len(self.bounds):
            raise ValueError("measurement dimension must match bounds")

        terms: list[float] = []
        if self.uniform_rate > 0:
            terms.append(log(self.uniform_rate / self.volume))
        for hotspot in self.hotspots:
            if hotspot.rate > 0:
                terms.append(
                    log(hotspot.rate)
                    + gaussian_logpdf(z, hotspot.mean, hotspot.covariance)
                )
        if self.min_intensity > 0:
            terms.append(log(self.min_intensity))
        if not terms:
            return -np.inf
        return logsumexp(np.array(terms, dtype=float))


def simulate_structured_clutter_scene(
    config: SimulationConfig | None = None,
    seed: int = 0,
) -> SimulationRun:
    """Simulate moving targets plus spatially structured clutter."""

    cfg = SimulationConfig() if config is None else config
    rng = np.random.default_rng(seed)
    calibration = _sample_clutter_measurements(rng, cfg, cfg.calibration_scans)

    frames: list[SimulationFrame] = []
    for scan_index in range(cfg.num_scans):
        truths = tuple(
            SimulatedTruth(target.target_id, target.position_at(scan_index))
            for target in cfg.targets
        )
        measurements: list[SimulatedMeasurement] = []

        for truth in truths:
            if rng.random() <= cfg.detection_probability:
                noisy_position = truth.position + rng.normal(0.0, cfg.measurement_std, size=cfg.dimension)
                measurements.append(SimulatedMeasurement(noisy_position, truth.target_id))

        for clutter_position in _sample_clutter_scan(rng, cfg):
            measurements.append(SimulatedMeasurement(clutter_position, None))

        rng.shuffle(measurements)
        frames.append(
            SimulationFrame(
                scan_index=scan_index,
                truths=truths,
                measurements=tuple(measurements),
            )
        )

    return SimulationRun(
        config=cfg,
        seed=seed,
        frames=tuple(frames),
        calibration_measurements=calibration,
    )


def _sample_clutter_measurements(
    rng: np.random.Generator,
    config: SimulationConfig,
    num_scans: int,
) -> NDArray[np.float64]:
    samples: list[NDArray[np.float64]] = []
    for _ in range(num_scans):
        samples.extend(_sample_clutter_scan(rng, config))
    if samples:
        return np.vstack(samples)
    return np.empty((0, config.dimension), dtype=float)


def _sample_clutter_scan(
    rng: np.random.Generator,
    config: SimulationConfig,
) -> list[NDArray[np.float64]]:
    samples: list[NDArray[np.float64]] = []

    for _ in range(rng.poisson(config.uniform_clutter_rate)):
        samples.append(_sample_uniform_in_bounds(rng, config.bounds))

    for hotspot in config.clutter_hotspots:
        count = rng.poisson(hotspot.rate)
        if count == 0:
            continue
        hotspot_samples = rng.multivariate_normal(
            mean=np.asarray(hotspot.mean, dtype=float),
            cov=np.asarray(hotspot.covariance, dtype=float),
            size=count,
        )
        samples.extend(np.asarray(sample, dtype=float) for sample in hotspot_samples)

    return samples


def _sample_uniform_in_bounds(rng: np.random.Generator, bounds: Bounds) -> NDArray[np.float64]:
    return np.array([rng.uniform(low, high) for low, high in bounds], dtype=float)


def _validate_bounds(bounds: Bounds) -> None:
    if not bounds:
        raise ValueError("bounds must not be empty")
    for low, high in bounds:
        if not low < high:
            raise ValueError("each bound must satisfy low < high")


def _as_vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_square_matrix(value: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return 0.5 * (arr + arr.T)
