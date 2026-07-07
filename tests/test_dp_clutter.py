import numpy as np

from robust_clutter_dp import (
    BirthModel,
    MeasurementModel,
    OnlineDPGaussianClutterIntensity,
    compete_measurement,
)


def test_online_dp_clutter_learns_separated_hotspots():
    rng = np.random.default_rng(3)
    samples = np.vstack(
        [
            rng.normal(loc=[0.0, 0.0], scale=0.15, size=(25, 2)),
            rng.normal(loc=[5.0, 5.0], scale=0.15, size=(25, 2)),
        ]
    )

    clutter = OnlineDPGaussianClutterIntensity.fit_sugs(
        samples,
        concentration=2.0,
        total_rate=30.0,
        covariance_floor=np.eye(2) * 0.02,
    )

    assert len(clutter.components) >= 2
    assert clutter.log_intensity(np.array([0.0, 0.0])) > clutter.log_intensity(np.array([2.5, 2.5]))
    assert clutter.log_intensity(np.array([5.0, 5.0])) > clutter.log_intensity(np.array([2.5, 2.5]))


def test_dp_clutter_intensity_competes_against_birth_in_hotspot():
    rng = np.random.default_rng(4)
    clutter_samples = rng.normal(loc=[0.0, 0.0], scale=0.2, size=(40, 2))
    clutter = OnlineDPGaussianClutterIntensity.fit_sugs(
        clutter_samples,
        concentration=0.5,
        total_rate=40.0,
        covariance_floor=np.eye(2) * 0.03,
    )

    result = compete_measurement(
        measurement=np.array([0.05, -0.05]),
        tracks=[],
        birth_model=BirthModel(
            rate=0.02,
            mean=np.array([0.0, 0.0]),
            covariance=np.eye(2),
            detection_probability=0.9,
        ),
        measurement_model=MeasurementModel(covariance=np.eye(2) * 0.1),
        clutter_model=clutter,
    )

    assert result.best_source == "clutter"
    assert result.clutter_probability > 0.99
