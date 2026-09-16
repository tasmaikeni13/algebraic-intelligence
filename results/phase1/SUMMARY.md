# Phase 1 CPU verification summary

All uncertainty intervals below are 95% Student-t intervals over independent vectors or trials. Deep experiments use float32; scalar identities and Monte Carlo statistics use float64. Hardware throughput is pending.

## Deep residual study: 10,000 trials per depth, width 128

| Depth | Gradient norm ratio, mean ± SEM | Observed range | ALU/GELU gradient variance delta | Activation variance ratio, mean ± SEM |
| --- | --- | --- | --- | --- |
| 8 | 1.018750 ± 0.001354 | [0.623268, 1.869823] | 1.666% | 1.014775 ± 0.001309 |
| 16 | 1.019851 ± 0.001380 | [0.628729, 1.707925] | 1.759% | 1.015582 ± 0.001301 |
| 24 | 1.021469 ± 0.001369 | [0.598508, 1.663790] | 1.814% | 1.016806 ± 0.001287 |
| 32 | 1.020007 ± 0.001369 | [0.628099, 1.657933] | 1.850% | 1.014376 ± 0.001291 |

Every observed gradient ratio is in [0.2,5]. Every observed depth-32 activation variance ratio is in [0.5,2]. All four gradient-variance deltas are below 5%. Full confidence intervals, all baseline arms, and per-trial observations are in `metrics.json` and `deep-trials.npz`.

## AVN: one million scalar samples at each input scale

| Scale | Epsilon | Centered variance, mean ± SEM | 95% CI |
| --- | --- | --- | --- |
| 0.1 | 1e-05 | 0.998972866 ± 0.000006698 | [0.998958847, 0.998986885] |
| 0.1 | 0.0 | 0.999970904 ± 0.000006734 | [0.999956810, 0.999984999] |
| 0.2 | 1e-05 | 0.999733365 ± 0.000007550 | [0.999717563, 0.999749167] |
| 0.2 | 0.0 | 0.999983684 ± 0.000007568 | [0.999967844, 0.999999523] |
| 0.5 | 1e-05 | 0.999947411 ± 0.000004089 | [0.999938852, 0.999955970] |
| 0.5 | 0.0 | 0.999987436 ± 0.000004087 | [0.999978883, 0.999995989] |
| 1.0 | 1e-05 | 0.999967403 ± 0.000006287 | [0.999954244, 0.999980562] |
| 1.0 | 0.0 | 0.999977400 ± 0.000006292 | [0.999964232, 0.999990569] |
| 2.0 | 1e-05 | 0.999972400 ± 0.000006396 | [0.999959012, 0.999985788] |
| 2.0 | 0.0 | 0.999974905 ± 0.000006396 | [0.999961517, 0.999988292] |
| 5.0 | 1e-05 | 0.999988967 ± 0.000002987 | [0.999982716, 0.999995218] |
| 5.0 | 0.0 | 0.999989367 ± 0.000002987 | [0.999983116, 0.999995618] |
| 10.0 | 1e-05 | 0.999988942 ± 0.000001995 | [0.999984766, 0.999993119] |
| 10.0 | 0.0 | 0.999989042 ± 0.000001995 | [0.999984866, 0.999993219] |

The default epsilon does not imply unit centered variance. Version 2 checks the exact regularized identities and preserves the original [0.9999,1.0001] interval for the ideal zero-epsilon experiment. The original failing gate and noncentered counterexample are retained.

## Scalar and graph checks

| Check | Observed maximum | Bound |
| --- | --- | --- |
| reflection | 0 | 1.0000000000000001e-15 |
| backward_autodiff | 4.4408920985006262e-16 | 5.0000000000000004e-16 |
| backward_numpy | 4.4408920985006262e-16 | 5.0000000000000004e-16 |
| lipschitz | 1.0443310539518174 | 1.05 |
| inflection | 0 | 1.0000000000000001e-15 |

The cached backward graphs have zero transcendental operations, divisions, and inverse square roots. All 33 unit tests and the complete Lean build pass. See `purity.json`, `pytest.log`, and `lean-build.log`.

## Limits

These are initial-state residual-network experiments, not transformer training. The width-64 and non-residual failures remain published. Float32/BF16 TPU numerical checks, million-sample experiments, deep trials, and the throughput/parameter-memory comparisons still require the real 16-chip slice.
