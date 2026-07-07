# RobustClutterUnknownNoiseDP

Prototype implementation for robust target-vs-clutter competition in Bayesian nonparametric multitarget tracking.

The first milestone is deliberately narrow:

> use a DP-style mixture where it is strong — spatial clutter-density estimation — and avoid using occupied DP components as physical target-cardinality estimates.

Each measurement or tentative tracklet is scored against three competing explanations:

1. existing target;
2. new target / birth;
3. clutter.

Candidate births should remain tentative until they beat the learned clutter explanation and pass posterior-existence / Bayesian-FDR confirmation.

The package currently provides:

- explicit birth-vs-clutter association weights;
- uniform, grid-based, and online DP-style Gaussian-mixture clutter intensities;
- Gaussian predictive likelihoods as the default model;
- Student-t predictive likelihoods as an optional robustness ablation;
- posterior existence updates from target-vs-clutter Bayes factors;
- Bayesian false-discovery-rate control for confirming tentative births;
- false-track-oriented evaluation helpers, including GOSPA-style decomposition and confirmation calibration/FDR diagnostics.

## Install for development

```bash
python -m pip install -e .[test]
pytest
```

## Minimal example

```python
import numpy as np
from robust_clutter_dp import (
    BirthModel,
    MeasurementModel,
    OnlineDPGaussianClutterIntensity,
    TrackPrediction,
    compete_measurement,
)

measurement_model = MeasurementModel(covariance=np.eye(2) * 0.25)

tracks = [
    TrackPrediction(
        track_id="T1",
        mean=np.array([0.0, 0.0]),
        covariance=np.eye(2) * 0.2,
        existence_probability=0.95,
        detection_probability=0.9,
    )
]

birth = BirthModel(
    rate=0.05,
    mean=np.array([5.0, 5.0]),
    covariance=np.eye(2) * 4.0,
    detection_probability=0.8,
)

clutter_samples = np.array(
    [
        [2.0, 2.0],
        [2.1, 1.9],
        [1.9, 2.2],
        [8.0, 1.0],
    ]
)
clutter = OnlineDPGaussianClutterIntensity.fit_sugs(
    clutter_samples,
    concentration=1.0,
    total_rate=20.0,
    covariance_floor=np.eye(2) * 0.05,
)

result = compete_measurement(
    measurement=np.array([0.1, -0.2]),
    tracks=tracks,
    birth_model=birth,
    measurement_model=measurement_model,
    clutter_model=clutter,
)

print(result.probabilities)
print(result.best_source)
```

## Evaluation helper example

```python
from robust_clutter_dp import PointObject, gospa_decomposition

metrics = gospa_decomposition(
    estimates=[PointObject("est-1", [0.1, 0.0]), PointObject("false", [10.0, 10.0])],
    truths=[PointObject("truth-1", [0.0, 0.0])],
    cutoff=2.0,
)

print(metrics.num_false)
print(metrics.false_cost)
```

## Research scope

The intended first-paper framing is:

```text
BNP clutter density + target-vs-clutter Bayes factor + FDR-controlled confirmation
```

The initial implementation keeps `p_D` fixed and uses Gaussian likelihoods by default. Learned detection-probability fields, repulsive priors, and full dependent-DP temporal dynamics are treated as later extensions rather than first-paper contributions. See `docs/research_scope.md` for the current scope boundary and evaluation plan.
