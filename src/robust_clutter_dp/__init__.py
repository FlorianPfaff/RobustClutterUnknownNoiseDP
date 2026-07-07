"""Robust clutter / unknown-noise primitives for DP-based MTT."""

from .clutter import GridClutterIntensity, UniformClutterIntensity
from .confirmation import (
    CandidateBirth,
    ConfirmationDecision,
    estimate_bayesian_fdr,
    posterior_existence_from_log_bayes_factor,
    select_by_bayesian_fdr,
)
from .dp_clutter import DPGaussianComponent, OnlineDPGaussianClutterIntensity
from .metrics import (
    ConfirmationMetrics,
    ConfirmationRecord,
    GospaDecomposition,
    GospaMatch,
    PointObject,
    confirmation_metrics,
    gospa_decomposition,
)
from .scoring import (
    AssociationResult,
    BirthModel,
    MeasurementModel,
    TrackPrediction,
    compete_measurement,
    gaussian_logpdf,
    student_t_logpdf,
)

__all__ = [
    "AssociationResult",
    "BirthModel",
    "CandidateBirth",
    "ConfirmationDecision",
    "ConfirmationMetrics",
    "ConfirmationRecord",
    "DPGaussianComponent",
    "GospaDecomposition",
    "GospaMatch",
    "GridClutterIntensity",
    "MeasurementModel",
    "OnlineDPGaussianClutterIntensity",
    "PointObject",
    "TrackPrediction",
    "UniformClutterIntensity",
    "compete_measurement",
    "confirmation_metrics",
    "estimate_bayesian_fdr",
    "gaussian_logpdf",
    "gospa_decomposition",
    "posterior_existence_from_log_bayes_factor",
    "select_by_bayesian_fdr",
    "student_t_logpdf",
]
