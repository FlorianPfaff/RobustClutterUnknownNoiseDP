# Research scope

This repository should stay focused on one defensible first paper:

> Bayesian nonparametrics for spatially structured clutter density, plus calibrated birth discipline through target-vs-clutter Bayes factors and Bayesian FDR-controlled track confirmation.

## Core thesis

Use the DP-style mechanism where it is useful: adaptive density estimation for structured clutter. Do not treat the number of occupied DP components as a physical target-cardinality estimate. Candidate births should be accepted only after they beat the learned clutter explanation and pass a posterior-existence/FDR confirmation rule.

In shorthand:

```text
DP: learn kappa_t(z), the clutter intensity
Bayes factor: compare birth against clutter
FDR rule: decide which tentative births become confirmed tracks
```

## Paper-1 scope

Include:

- DP-mixture clutter intensity learning;
- explicit birth-vs-clutter source competition;
- posterior existence probabilities for tentative births;
- Bayesian posterior-expected-FDR confirmation;
- fixed detection probability for the first experiments;
- Gaussian measurement likelihood as the default model;
- Student-t likelihood only as a robustness ablation, not as the main novelty.

Exclude from the first paper:

- learned spatial `p_D(x,t)` fields;
- repulsive priors;
- a full dependent-DP temporal model;
- claims that the DP estimates target cardinality;
- broad claims about unknown-noise filtering beyond the birth-vs-clutter decision.

## Current benchmark scaffold

The current codebase now has a deliberately small end-to-end benchmark path:

```text
structured-clutter simulator
    -> clutter-only calibration samples
    -> uniform/grid/DP/oracle clutter intensity
    -> birth-vs-clutter measurement scoring
    -> tentative-birth tracklet manager
    -> posterior-FDR confirmation
    -> GOSPA/FDR/false-track metrics
```

The simulator is not intended to be a final benchmark. Its purpose is to make the first paper claim executable early: does a learned spatial clutter intensity reduce false births in a persistent clutter hotspot compared with a uniform clutter model?

## Reviewer questions to answer experimentally

### Why not robust GLMB/PMBM?

The answer must be empirical and clutter-specific: a learned spatial clutter map should reduce false births in regimes where scalar-rate or coarse-profile clutter models fail. Good stress cases include persistent multipath ghost regions, clutter ridges, localized detector false positives, and bursty nonuniform clutter.

### What resolves identifiability?

A persistent clutter hotspot and a stationary or slow target can be difficult to distinguish from position-only data. The experiments and discussion should make the discriminator explicit: target dynamics, track-level temporal consistency, amplitude/features if available, and the fact that the learned clutter intensity competes against birth rather than automatically suppressing all stationary objects.

The current tentative-birth manager includes a minimal motion-span gate before confirmation. That gate is not a general solution to identifiability; it is a first diagnostic mechanism to avoid confirming several same-location clutter returns as a moving target in the toy benchmark.

## Metrics

Report metrics that expose false-track behavior:

- GOSPA/OSPA decomposition;
- false-track count;
- false-track duration;
- time to confirm;
- missed-target count;
- track fragmentation;
- posterior-existence calibration.

Localization RMSE alone is insufficient because a method can localize real targets well while producing too many false tracks.

## Baseline ladder

A useful baseline ladder is:

1. vanilla DP/DDP tracker that maps occupied components to targets;
2. fixed-uniform-clutter birth-vs-clutter tracker;
3. grid or scalar clutter-rate robust tracker;
4. robust GLMB/PMBM or CPHD-style baseline;
5. oracle-clutter baseline as an upper bound.

The first milestone in this repository targets items 2 and 3, then adds the DP-mixture clutter model needed for item 4/5 comparisons.
