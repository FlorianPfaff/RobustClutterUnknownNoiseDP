"""Toy structured-clutter birth-suppression example.

This is a diagnostic experiment, not a full multitarget tracker. It isolates the
paper-1 mechanism: a spatial clutter map should suppress false births in a
persistent clutter hotspot more strongly than a uniform clutter model with the
same broad scene volume.
"""

from __future__ import annotations

import numpy as np

from robust_clutter_dp import (
    BirthModel,
    MeasurementModel,
    OnlineDPGaussianClutterIntensity,
    UniformClutterIntensity,
    compete_measurement,
)


def average_birth_probability(clutter_model, measurements: np.ndarray) -> float:
    measurement_model = MeasurementModel(covariance=np.eye(2) * 0.05)
    birth_model = BirthModel(
        rate=0.2,
        mean=np.array([2.0, 2.0]),
        covariance=np.eye(2) * 1.5,
        detection_probability=0.9,
    )

    probabilities = []
    for measurement in measurements:
        result = compete_measurement(
            measurement=measurement,
            tracks=[],
            birth_model=birth_model,
            measurement_model=measurement_model,
            clutter_model=clutter_model,
        )
        probabilities.append(result.birth_probability)
    return float(np.mean(probabilities))


def main() -> None:
    rng = np.random.default_rng(7)
    hotspot_samples = rng.normal(loc=[2.0, 2.0], scale=0.15, size=(80, 2))
    heldout_hotspot_measurements = rng.normal(loc=[2.0, 2.0], scale=0.15, size=(25, 2))

    uniform_clutter = UniformClutterIntensity(rate=5.0, volume=400.0)
    learned_clutter = OnlineDPGaussianClutterIntensity.fit_sugs(
        hotspot_samples,
        concentration=0.5,
        total_rate=80.0,
        covariance_floor=np.eye(2) * 0.02,
    )

    uniform_birth_probability = average_birth_probability(
        uniform_clutter,
        heldout_hotspot_measurements,
    )
    learned_birth_probability = average_birth_probability(
        learned_clutter,
        heldout_hotspot_measurements,
    )

    print("Average birth probability inside clutter hotspot")
    print(f"  uniform clutter model: {uniform_birth_probability:.4f}")
    print(f"  learned DP clutter map: {learned_birth_probability:.4f}")
    print(f"  learned clutter components: {len(learned_clutter.components)}")


if __name__ == "__main__":
    main()
