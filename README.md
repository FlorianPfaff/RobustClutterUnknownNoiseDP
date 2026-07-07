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
- uniform, grid-based, online DP-style Gaussian-mixture, and oracle structured-clutter intensities;
- named structured-clutter scenario presets, including control and identifiability-stress cases;
- a small structured-clutter simulator with clutter-only calibration samples;
- a tentative-birth tracklet manager with sequential Bayes-factor updates and FDR confirmation;
- Gaussian predictive likelihoods as the default model;
- Student-t predictive likelihoods as an optional robustness ablation;
- posterior existence updates from target-vs-clutter Bayes factors;
- Bayesian false-discovery-rate control for confirming tentative births;
- false-track-oriented evaluation helpers, including GOSPA-style decomposition and confirmation calibration/FDR diagnostics;
- a minimal uniform/grid/DP/oracle structured-clutter comparison experiment;
- reporting helpers for raw per-seed CSV rows, cross-seed scenario/method summaries, aggregate oracle deltas, and paired seed-wise oracle deltas;
- a console entry point for reproducible benchmark runs.

## Install for development

```bash
python -m pip install -e .[test]
pytest
```

The install exposes a benchmark command:

```bash
robust-clutter-dp-experiment --help
```

## Minimal scoring example

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

## Structured-clutter comparison example

Use the installed command:

```bash
robust-clutter-dp-experiment \
  --scenarios hotspot,no_hotspot_control,near_hotspot_crossing \
  --methods uniform,grid,dp,oracle \
  --seeds 0:5 \
  --output all
```

Use `--output paired-comparison` to print only paired seed-wise deltas against the reference method.

Or run the convenience wrapper:

```bash
python examples/run_structured_clutter_experiment.py
```

Or use the Python API:

```python
from robust_clutter_dp import (
    ExperimentConfig,
    aggregate_method_results,
    compare_to_reference,
    compare_to_reference_paired,
    format_method_aggregates_csv,
    format_method_comparisons_csv,
    format_paired_method_comparisons_csv,
    run_named_scenarios_comparison,
)

results = run_named_scenarios_comparison(
    scenario_names=("hotspot", "no_hotspot_control", "near_hotspot_crossing"),
    seeds=range(5),
    experiment_config=ExperimentConfig(methods=("uniform", "grid", "dp", "oracle")),
)
aggregates = aggregate_method_results(results)
comparisons = compare_to_reference(aggregates, reference_method="oracle")
paired_comparisons = compare_to_reference_paired(results, reference_method="oracle")

print(format_method_aggregates_csv(aggregates))
print(format_method_comparisons_csv(comparisons))
print(format_paired_method_comparisons_csv(paired_comparisons))
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
