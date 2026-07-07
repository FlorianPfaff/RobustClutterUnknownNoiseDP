import numpy as np

from robust_clutter_dp import (
    BirthModel,
    CandidateBirth,
    GridClutterIntensity,
    MeasurementModel,
    TrackPrediction,
    UniformClutterIntensity,
    compete_measurement,
    gaussian_logpdf,
    posterior_existence_from_log_bayes_factor,
    select_by_bayesian_fdr,
    student_t_logpdf,
)


def test_existing_track_wins_near_prediction():
    measurement_model = MeasurementModel(covariance=np.eye(2) * 0.1)
    tracks = [
        TrackPrediction(
            track_id="A",
            mean=np.array([0.0, 0.0]),
            covariance=np.eye(2) * 0.1,
            existence_probability=0.95,
            detection_probability=0.9,
        )
    ]
    birth = BirthModel(rate=0.01, mean=np.array([5.0, 5.0]), covariance=np.eye(2) * 4.0)
    clutter = UniformClutterIntensity(rate=1.0, volume=400.0)

    result = compete_measurement(
        measurement=np.array([0.05, -0.05]),
        tracks=tracks,
        birth_model=birth,
        measurement_model=measurement_model,
        clutter_model=clutter,
    )

    assert result.best_source == "track:A"
    assert result.probabilities["track:A"] > 0.95


def test_high_clutter_intensity_can_suppress_birth():
    measurement_model = MeasurementModel(covariance=np.eye(2) * 0.5)
    birth = BirthModel(
        rate=0.05,
        mean=np.array([0.0, 0.0]),
        covariance=np.eye(2),
        detection_probability=0.9,
    )
    clutter = UniformClutterIntensity(rate=100.0, volume=10.0)

    result = compete_measurement(
        measurement=np.array([0.1, 0.1]),
        tracks=[],
        birth_model=birth,
        measurement_model=measurement_model,
        clutter_model=clutter,
    )

    assert result.best_source == "clutter"
    assert result.clutter_probability > result.birth_probability


def test_grid_clutter_hotspot_beats_birth_locally():
    samples = np.array(
        [
            [0.1, 0.1],
            [0.2, 0.1],
            [0.1, 0.2],
            [9.0, 9.0],
        ]
    )
    grid = GridClutterIntensity.from_samples(
        samples=samples,
        bounds=((0.0, 10.0), (0.0, 10.0)),
        bins=(2, 2),
        total_rate=20.0,
        smoothing=0.1,
    )

    near_hotspot = grid.log_intensity(np.array([0.25, 0.25]))
    away = grid.log_intensity(np.array([7.5, 2.5]))
    assert near_hotspot > away


def test_student_t_is_less_punitive_for_outlier_than_gaussian():
    x = np.array([8.0])
    mean = np.array([0.0])
    scale = np.eye(1)

    assert student_t_logpdf(x, mean, scale, degrees_of_freedom=3.0) > gaussian_logpdf(x, mean, scale)


def test_posterior_existence_from_log_bayes_factor_increases_with_evidence():
    posterior = posterior_existence_from_log_bayes_factor(
        log_bayes_factor=np.log(9.0),
        prior_existence=0.1,
    )
    assert np.isclose(posterior, 0.5)


def test_select_by_bayesian_fdr_accepts_largest_valid_prefix():
    candidates = [
        CandidateBirth("a", 0.99),
        CandidateBirth("b", 0.95),
        CandidateBirth("c", 0.60),
        CandidateBirth("d", 0.20),
    ]

    decision = select_by_bayesian_fdr(candidates, q=0.10)

    assert decision.accepted_ids == ("a", "b")
    assert decision.estimated_fdr <= 0.10
    assert decision.threshold == 0.95
