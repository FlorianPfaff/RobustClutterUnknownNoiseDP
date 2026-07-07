"""Robust clutter / unknown-noise primitives for DP-based MTT."""

from .clutter import GridClutterIntensity, UniformClutterIntensity
from .confirmation import (
    CandidateBirth,
    ConfirmationDecision,
    estimate_bayesian_fdr,
    posterior_existence_from_log_bayes_factor,
    select_by_bayesian_fdr,
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
    "GridClutterIntensity",
    "MeasurementModel",
    "TrackPrediction",
    "UniformClutterIntensity",
    "compete_measurement",
    "estimate_bayesian_fdr",
    "gaussian_logpdf",
    "posterior_existence_from_log_bayes_factor",
    "select_by_bayesian_fdr",
    "student_t_logpdf",
]
