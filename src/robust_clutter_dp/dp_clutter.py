"""Online DP-style Gaussian-mixture clutter intensity estimation.

This module puts the Bayesian nonparametric layer on the *clutter density* side
of the tracker. The approximation is intentionally lightweight: a deterministic
SUGS/MAP assignment pass creates Gaussian clutter components from samples or
high-clutter-responsibility measurements. The resulting object implements the
same ``log_intensity(z)`` protocol used by the association scorer.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .scoring import gaussian_logpdf, logsumexp, normalize_log_weights


@dataclass(frozen=True)
class DPGaussianComponent:
    """A fitted Gaussian component in the clutter intensity mixture."""

    count: float
    mean: NDArray[np.float64]
    covariance: NDArray[np.float64]

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("component count must be positive")
        mean = _as_vector(self.mean, "component mean")
        covariance = _as_square_matrix(self.covariance, "component covariance")
        if covariance.shape[0] != mean.size:
            raise ValueError("component mean and covariance dimensions disagree")
        sign, _ = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise ValueError("component covariance must be positive definite")


@dataclass(frozen=True)
class OnlineDPGaussianClutterIntensity:
    """DP-style Gaussian-mixture estimate of spatial clutter intensity.

    The mixture density integrates to ``total_rate``. It can therefore be used as
    ``kappa_t(z)`` in birth-vs-clutter association weights.
    """

    components: tuple[DPGaussianComponent, ...]
    total_rate: float
    min_intensity: float = 1e-12

    def __post_init__(self) -> None:
        if self.total_rate < 0:
            raise ValueError("total_rate must be non-negative")
        if self.min_intensity < 0:
            raise ValueError("min_intensity must be non-negative")
        if not self.components and self.total_rate > 0:
            raise ValueError("positive total_rate requires at least one component")
        if self.components:
            dim = self.components[0].mean.size
            for component in self.components:
                if component.mean.size != dim or component.covariance.shape != (dim, dim):
                    raise ValueError("all components must have the same dimension")

    @classmethod
    def fit_sugs(
        cls,
        samples: ArrayLike,
        concentration: float = 1.0,
        total_rate: float | None = None,
        base_mean: ArrayLike | None = None,
        base_covariance: ArrayLike | None = None,
        covariance_floor: ArrayLike | None = None,
        min_intensity: float = 1e-12,
        max_components: int | None = None,
    ) -> "OnlineDPGaussianClutterIntensity":
        """Fit a deterministic online DP-mixture approximation.

        Parameters
        ----------
        samples:
            Clutter-like measurements, typically selected or weighted by a
            previous association pass.
        concentration:
            CRP concentration controlling the tendency to create new clutter
            components. Larger values create more components.
        total_rate:
            Expected clutter count per scan represented by the intensity. If
            omitted, the number of samples is used.
        base_mean, base_covariance:
            Base predictive distribution for a new clutter component.
        covariance_floor:
            Positive-definite jitter added to component covariance estimates.
        max_components:
            Optional truncation for real-time bounded-memory use.
        """

        sample_arr = np.asarray(samples, dtype=float)
        if sample_arr.ndim != 2:
            raise ValueError("samples must be a two-dimensional array")
        if sample_arr.shape[0] == 0:
            raise ValueError("at least one sample is required")
        if concentration <= 0:
            raise ValueError("concentration must be positive")
        if max_components is not None and max_components <= 0:
            raise ValueError("max_components must be positive or None")

        n_samples, dim = sample_arr.shape
        if total_rate is None:
            total_rate = float(n_samples)
        if total_rate < 0:
            raise ValueError("total_rate must be non-negative")

        floor = _default_covariance_floor(dim) if covariance_floor is None else _as_square_matrix(
            covariance_floor, "covariance_floor"
        )
        if floor.shape != (dim, dim):
            raise ValueError("covariance_floor dimension must match samples")

        base_mu = np.mean(sample_arr, axis=0) if base_mean is None else _as_vector(base_mean, "base_mean")
        if base_mu.size != dim:
            raise ValueError("base_mean dimension must match samples")

        if base_covariance is None:
            base_cov = _empirical_covariance(sample_arr) + floor
        else:
            base_cov = _as_square_matrix(base_covariance, "base_covariance") + floor
        if base_cov.shape != (dim, dim):
            raise ValueError("base_covariance dimension must match samples")

        clusters: list[_RunningGaussian] = []
        for sample in sample_arr:
            if not clusters:
                clusters.append(_RunningGaussian.create(sample, initial_covariance=base_cov, covariance_floor=floor))
                continue

            log_weights: dict[str, float] = {
                str(index): log(cluster.count) + gaussian_logpdf(sample, cluster.mean, cluster.covariance)
                for index, cluster in enumerate(clusters)
            }
            if max_components is None or len(clusters) < max_components:
                log_weights["new"] = log(concentration) + gaussian_logpdf(sample, base_mu, base_cov)

            assignment_probabilities = normalize_log_weights(log_weights)
            assignment = max(assignment_probabilities, key=assignment_probabilities.__getitem__)
            if assignment == "new":
                clusters.append(_RunningGaussian.create(sample, initial_covariance=base_cov, covariance_floor=floor))
            else:
                clusters[int(assignment)].update(sample)

        components = tuple(
            DPGaussianComponent(
                count=float(cluster.count),
                mean=np.array(cluster.mean, dtype=float),
                covariance=np.array(cluster.covariance, dtype=float),
            )
            for cluster in clusters
        )
        return cls(components=components, total_rate=float(total_rate), min_intensity=min_intensity)

    @property
    def total_component_count(self) -> float:
        """Total pseudo-count across mixture components."""

        return float(sum(component.count for component in self.components))

    def log_intensity(self, measurement: ArrayLike) -> float:
        """Return log kappa_t(z) for the fitted clutter mixture."""

        if self.total_rate == 0 or not self.components:
            return -np.inf if self.min_intensity == 0 else float(log(self.min_intensity))

        z = _as_vector(measurement, "measurement")
        total_count = self.total_component_count
        terms = []
        for component in self.components:
            if z.size != component.mean.size:
                raise ValueError("measurement dimension must match clutter components")
            terms.append(
                log(self.total_rate)
                + log(component.count / total_count)
                + gaussian_logpdf(z, component.mean, component.covariance)
            )
        value = logsumexp(np.array(terms, dtype=float))
        if self.min_intensity > 0:
            return float(max(value, log(self.min_intensity)))
        return float(value)

    def component_responsibilities(self, measurement: ArrayLike) -> dict[int, float]:
        """Posterior component responsibilities within the clutter mixture."""

        if not self.components:
            return {}
        z = _as_vector(measurement, "measurement")
        total_count = self.total_component_count
        log_weights = {
            index: log(component.count / total_count) + gaussian_logpdf(z, component.mean, component.covariance)
            for index, component in enumerate(self.components)
        }
        total = logsumexp(np.array(list(log_weights.values()), dtype=float))
        return {index: float(np.exp(weight - total)) for index, weight in log_weights.items()}


@dataclass
class _RunningGaussian:
    count: int
    mean: NDArray[np.float64]
    scatter: NDArray[np.float64]
    initial_covariance: NDArray[np.float64]
    covariance_floor: NDArray[np.float64]

    @classmethod
    def create(
        cls,
        sample: NDArray[np.float64],
        initial_covariance: NDArray[np.float64],
        covariance_floor: NDArray[np.float64],
    ) -> "_RunningGaussian":
        dim = sample.size
        return cls(
            count=1,
            mean=np.array(sample, dtype=float),
            scatter=np.zeros((dim, dim), dtype=float),
            initial_covariance=np.array(initial_covariance, dtype=float),
            covariance_floor=np.array(covariance_floor, dtype=float),
        )

    @property
    def covariance(self) -> NDArray[np.float64]:
        if self.count < 2:
            return np.array(self.initial_covariance, dtype=float)
        return _symmetrize(self.scatter / float(self.count - 1) + self.covariance_floor)

    def update(self, sample: NDArray[np.float64]) -> None:
        self.count += 1
        delta = sample - self.mean
        self.mean = self.mean + delta / self.count
        delta2 = sample - self.mean
        self.scatter = self.scatter + np.outer(delta, delta2)


def _empirical_covariance(samples: NDArray[np.float64]) -> NDArray[np.float64]:
    dim = samples.shape[1]
    if samples.shape[0] < 2:
        return np.eye(dim)
    cov = np.cov(samples, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    return _symmetrize(np.asarray(cov, dtype=float))


def _default_covariance_floor(dim: int) -> NDArray[np.float64]:
    return np.eye(dim, dtype=float) * 1e-3


def _as_vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    return arr


def _as_square_matrix(value: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    return _symmetrize(arr)


def _symmetrize(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * (matrix + matrix.T)
