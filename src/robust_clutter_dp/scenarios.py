"""Named scenario presets for structured-clutter experiments."""

from __future__ import annotations

import numpy as np

from .simulation import ClutterHotspot, LinearTarget, SimulationConfig


SCENARIO_NAMES = (
    "hotspot",
    "no_hotspot_control",
    "two_hotspots",
    "near_hotspot_crossing",
)


def make_scenario(name: str) -> SimulationConfig:
    """Return a named simulation scenario.

    The presets are intentionally small and deterministic. They are designed for
    fast benchmark iteration before replacing the toy generator with a stronger
    tracking benchmark.
    """

    if name == "hotspot":
        return hotspot_scenario()
    if name == "no_hotspot_control":
        return no_hotspot_control_scenario()
    if name == "two_hotspots":
        return two_hotspots_scenario()
    if name == "near_hotspot_crossing":
        return near_hotspot_crossing_scenario()
    raise ValueError(f"unknown scenario {name!r}; expected one of {SCENARIO_NAMES}")


def hotspot_scenario() -> SimulationConfig:
    """Default one-target scene with one persistent clutter hotspot."""

    return SimulationConfig(
        num_scans=20,
        targets=(
            LinearTarget("target-1", np.array([1.0, 8.0]), np.array([0.35, -0.18])),
        ),
        uniform_clutter_rate=1.0,
        clutter_hotspots=(
            ClutterHotspot(np.array([2.0, 2.0]), np.eye(2) * 0.06, rate=5.0),
        ),
        calibration_scans=12,
    )


def no_hotspot_control_scenario() -> SimulationConfig:
    """Control scene with only spatially uniform clutter."""

    return SimulationConfig(
        num_scans=20,
        targets=(
            LinearTarget("target-1", np.array([1.0, 8.0]), np.array([0.35, -0.18])),
        ),
        uniform_clutter_rate=6.0,
        clutter_hotspots=(),
        calibration_scans=12,
    )


def two_hotspots_scenario() -> SimulationConfig:
    """Structured-clutter scene with two separated hotspot regions."""

    return SimulationConfig(
        num_scans=22,
        targets=(
            LinearTarget("target-1", np.array([1.0, 8.0]), np.array([0.32, -0.16])),
            LinearTarget("target-2", np.array([8.5, 1.5]), np.array([-0.22, 0.24])),
        ),
        uniform_clutter_rate=1.0,
        clutter_hotspots=(
            ClutterHotspot(np.array([2.0, 2.0]), np.eye(2) * 0.06, rate=4.0),
            ClutterHotspot(np.array([7.5, 6.5]), np.array([[0.10, 0.03], [0.03, 0.05]]), rate=3.0),
        ),
        calibration_scans=14,
    )


def near_hotspot_crossing_scenario() -> SimulationConfig:
    """Identifiability stress case where a true target passes near a hotspot."""

    return SimulationConfig(
        num_scans=24,
        targets=(
            LinearTarget("target-1", np.array([0.8, 2.8]), np.array([0.25, -0.04])),
        ),
        uniform_clutter_rate=1.0,
        clutter_hotspots=(
            ClutterHotspot(np.array([3.0, 2.4]), np.array([[0.08, 0.02], [0.02, 0.05]]), rate=5.0),
        ),
        calibration_scans=14,
    )
