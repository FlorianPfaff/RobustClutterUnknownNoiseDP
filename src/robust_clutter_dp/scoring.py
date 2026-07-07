"""Association scoring for robust target-vs-clutter competition.

The functions in this module deliberately separate *component creation* from
*target confirmation*. A measurement can support an existing target, a birth
hypothesis, or clutter. The normalized posterior weights are the primitive that
higher-level DP/RFS/MHT code can use before confirming new tracks.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, lgamma, log, pi
from typing import Mapping, Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


_LOG_ZERO = -np.inf


class ClutterIntensity(Protocol):
    """Protocol for spatial clutter intensities.

    The returned value is an intensity/density-like quantity in measurement
    space. It competes directly with target-generated predictive likelihoods.
    """

    def log_intensity(self, measurement: ArrayLike) -> float:
        """Return log kappa_t(z)."""


@dataclass(frozen=True)
class TrackPrediction:
    """Predicted single-target state used for association scoring."""

    track_id: str
    mean: ArrayLike
    covariance: ArrayLike
    existence_probability: float = 1.0
    detection_probability: float = 1.0

    def __post_init__(self) -> None:
        _validate_probability(self.existence_probability, "existence_probability")
        _validate_probability(self.detection_probability, "detection_probability")


@dataclass(frozen=True)
class BirthModel:
    """Gaussian prior intensity for new-target births."""

    rate: float
    mean: ArrayLike
    covariance: ArrayLike
    detection_probability: float = 1.0

    def __post_init__(self) -> None:
        if self.rate < 0:
            raise ValueError("birth rate must be non-negative")
        _validate_probability(self.detection_probability, "detection_probability")


@dataclass(frozen=True)
class MeasurementModel:
    """Linear-Gaussian or linear-Student-t predictive measurement model.

    If ``degrees_of_freedom`` is ``None``, predictive likelihoods are Gaussian.
    Otherwise they are multivariate Student-t. The Student-t option is a robust
    likelihood for unknown/heavy-tailed measurement noise.
    """

    covariance: ArrayLike
    matrix: ArrayLike | None = None
    degrees_of_freedom: float | None = None

    def __post_init__(self) -> None:
        covariance = _as_square_matrix(self.covariance, "measurement covariance")
        if self.degrees_of_freedom is not None and self.degrees_of_freedom <= 0:
            raise ValueError("degrees_of_freedom must be positive or None")
        if self.matrix is not None:
            matrix = np.asarray(self.matrix, dtype=float)
            if matrix.ndim != 2:
                raise ValueError("measurement matrix must be two-dimensional")
            if matrix.shape[0] != covariance.shape[0]:
                raise ValueError(
                    "measurement matrix row count must match measurement covariance dimension"
                )

    def projected(self, mean: ArrayLike, covariance: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return predictive measurement mean and covariance."""

        mean_arr = _as_vector(mean, "state mean")
        cov_arr = _as_square_matrix(covariance, "state covariance")
        meas_cov = _as_square_matrix(self.covariance, "measurement covariance")

        if self.matrix is None:
            if cov_arr.shape != meas_cov.shape:
                raise ValueError(
                    "state covariance must match measurement covariance when matrix is None"
                )
            return mean_arr, cov_arr + meas_cov

        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.shape[1] != mean_arr.size or cov_arr.shape[0] != matrix.shape[1]:
            raise ValueError("measurement matrix is incompatible with state dimension")
        pred_mean = matrix @ mean_arr
        pred_cov = matrix @ cov_arr @ matrix.T + meas_cov
        return pred_mean, _symmetrize(pred_cov)

    def log_likelihood(self, measurement: ArrayLike, mean: ArrayLike, covariance: ArrayLike) -> float:
        """Evaluate the predictive log likelihood log g(z | x)."""

        pred_mean, pred_cov = self.projected(mean, covariance)
        z = _as_vector(measurement, "measurement")
        if z.size != pred_mean.size:
            raise ValueError("measurement dimension is incompatible with predictive mean")
        if self.degrees_of_freedom is None:
            return gaussian_logpdf(z, pred_mean, pred_cov)
        return student_t_logpdf(z, pred_mean, pred_cov, self.degrees_of_freedom)


@dataclass(frozen=True)
class AssociationResult:
    """Normalized source probabilities for one measurement."""

    log_weights: Mapping[str, float]
    probabilities: Mapping[str, float]

    @property
    def best_source(self) -> str:
        """Source label with the largest posterior probability."""

        return max(self.probabilities, key=self.probabilities.__getitem__)

    @property
    def birth_probability(self) -> float:
        """Posterior probability assigned to the birth hypothesis."""

        return self.probabilities.get("birth", 0.0)

    @property
    def clutter_probability(self) -> float:
        """Posterior probability assigned to clutter."""

        return self.probabilities.get("clutter", 0.0)


def compete_measurement(
    measurement: ArrayLike,
    tracks: Sequence[TrackPrediction],
    birth_model: BirthModel,
    measurement_model: MeasurementModel,
    clutter_model: ClutterIntensity,
) -> AssociationResult:
    """Score a measurement against existing targets, birth, and clutter.

    The returned labels are ``track:<track_id>``, ``birth``, and ``clutter``.
    New DP components can be initialized from high birth probability, but should
    remain tentative until confirmed by a sequential rule.
    """

    log_weights: dict[str, float] = {}

    for track in tracks:
        log_weights[f"track:{track.track_id}"] = _safe_log(track.existence_probability) + _safe_log(
            track.detection_probability
        ) + measurement_model.log_likelihood(measurement, track.mean, track.covariance)

    log_weights["birth"] = _safe_log(birth_model.rate) + _safe_log(
        birth_model.detection_probability
    ) + measurement_model.log_likelihood(
        measurement,
        birth_model.mean,
        birth_model.covariance,
    )

    log_weights["clutter"] = float(clutter_model.log_intensity(measurement))

    probabilities = normalize_log_weights(log_weights)
    return AssociationResult(log_weights=log_weights, probabilities=probabilities)


def normalize_log_weights(log_weights: Mapping[str, float]) -> dict[str, float]:
    """Normalize log weights into probabilities using a stable log-sum-exp."""

    if not log_weights:
        raise ValueError("log_weights must not be empty")
    values = np.array(list(log_weights.values()), dtype=float)
    total = logsumexp(values)
    if not isfinite(total):
        raise ValueError("at least one log weight must be finite")
    return {key: float(np.exp(value - total)) for key, value in log_weights.items()}


def gaussian_logpdf(x: ArrayLike, mean: ArrayLike, covariance: ArrayLike) -> float:
    """Multivariate Gaussian log density."""

    x_arr = _as_vector(x, "x")
    mean_arr = _as_vector(mean, "mean")
    cov_arr = _as_square_matrix(covariance, "covariance")
    if x_arr.size != mean_arr.size or cov_arr.shape[0] != x_arr.size:
        raise ValueError("incompatible Gaussian dimensions")

    sign, logdet = np.linalg.slogdet(cov_arr)
    if sign <= 0:
        raise ValueError("covariance must be positive definite")
    residual = x_arr - mean_arr
    maha = float(residual.T @ np.linalg.solve(cov_arr, residual))
    dim = x_arr.size
    return float(-0.5 * (dim * log(2.0 * pi) + logdet + maha))


def student_t_logpdf(
    x: ArrayLike,
    mean: ArrayLike,
    scale: ArrayLike,
    degrees_of_freedom: float,
) -> float:
    """Multivariate Student-t log density.

    ``scale`` plays the usual Student-t scale-matrix role. For large residuals,
    this density has heavier tails than a Gaussian with the same scale matrix.
    """

    if degrees_of_freedom <= 0:
        raise ValueError("degrees_of_freedom must be positive")
    x_arr = _as_vector(x, "x")
    mean_arr = _as_vector(mean, "mean")
    scale_arr = _as_square_matrix(scale, "scale")
    if x_arr.size != mean_arr.size or scale_arr.shape[0] != x_arr.size:
        raise ValueError("incompatible Student-t dimensions")

    sign, logdet = np.linalg.slogdet(scale_arr)
    if sign <= 0:
        raise ValueError("scale must be positive definite")
    residual = x_arr - mean_arr
    maha = float(residual.T @ np.linalg.solve(scale_arr, residual))
    dim = x_arr.size
    nu = float(degrees_of_freedom)
    return float(
        lgamma((nu + dim) / 2.0)
        - lgamma(nu / 2.0)
        - 0.5 * (dim * log(nu * pi) + logdet)
        - 0.5 * (nu + dim) * np.log1p(maha / nu)
    )


def logsumexp(values: ArrayLike) -> float:
    """Stable log-sum-exp for a one-dimensional array."""

    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if arr.size == 0:
        raise ValueError("values must not be empty")
    max_value = float(np.max(arr))
    if not isfinite(max_value):
        return _LOG_ZERO
    return float(max_value + np.log(np.sum(np.exp(arr - max_value))))


def _safe_log(value: float) -> float:
    if value < 0:
        raise ValueError("weights and probabilities must be non-negative")
    if value == 0:
        return _LOG_ZERO
    return log(value)


def _validate_probability(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


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
