# RobustClutterUnknownNoiseDP

Prototype implementation for robust target-vs-clutter competition in Bayesian nonparametric multitarget tracking.

The first milestone is intentionally small: a reusable scoring layer that prevents a newly occupied DP component from being interpreted directly as a physical target. Each measurement or tentative tracklet is scored against three competing explanations:

1. existing target;
2. new target / birth;
3. clutter.

The package currently provides:

- Gaussian and Student-t predictive likelihoods;
- explicit birth-vs-clutter association weights;
- uniform and grid-based spatial clutter intensities;
- posterior existence updates from target-vs-clutter Bayes factors;
- Bayesian false-discovery-rate control for confirming tentative births.

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
    TrackPrediction,
    UniformClutterIntensity,
    compete_measurement,
)

measurement_model = MeasurementModel(
    covariance=np.eye(2) * 0.25,
    degrees_of_freedom=5.0,  # Student-t; use None for Gaussian
)

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

clutter = UniformClutterIntensity(rate=20.0, volume=100.0)

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

## Design note

The DP mechanism should be used as a hypothesis generator, not as a direct physical-cardinality estimator. Candidate DP components should be confirmed only after Bayesian competition with an adaptive clutter model and a calibrated posterior-existence rule.
