# Phase 2 failures and repairs

All evidence paths below are relative to this directory. None of the failed
scientific results is counted as a PASS.

| Iteration | Evidence | Classification and action |
| --- | --- | --- |
| Initial implementation | `iterations/pytest-initial.log`, `iterations/pytest-implementation.log` | 54 tests passed. Stable negative conjugate branch, exactly three squarings, FP32 intermediates for BF16, and cached VJP including AVN chain rule. |
| Original scientific claims | `iterations/original-gates.json`, `iterations/original-noise.npz`, `counterexamples.json` | Local mathematical/specification defects: bound 2 is entrywise in normalized scores; exact sharpness is irrational; base-rho Lipschitz continuity does not imply 100× noise superiority for octic attention. Stronger matrix/raw-score claims have explicit counterexamples. |
| Lean strengthening | `iterations/lean-additions.log` | Added probability-product bounds, conditional entrywise bound, exact sharpness polynomial and positive sink-mass bound. Root build passes without warnings or placeholders. |
| Full CPU serialization failure | `iterations/cpu-serialization-failure/` | All samples and Jacobians completed, but a NumPy boolean was not JSON serializable. Raw arrays and the full traceback are preserved. Converted the Jacobian result to a Python bool and added an evidence-serialization regression test. |
| First TPU serialization failure | `iterations/tpu-serialization-failure/` | Numerical parity and all throughput gates passed; all seven noise/distribution studies failed. A NumPy float32 tolerance prevented writing JSON. The coordinator failed while peers waited at a barrier. The complete console output is preserved; this run is not final hardware evidence. |
| Recovery from failed TPU run | Same launcher log | TPU runtime intercepted SIGTERM; SSH retried after workers were killed. Stopped the owning local gcloud retry process and then only Python processes matching this exact failed snapshot. Other workloads were not targeted. No measurements from that automatic retry supply final evidence. |
| Scalar serialization repair | `iterations/pytest-serialization-repair.log` | The Phase 2 JSON writer converts NumPy scalar types without changing values, rejects NaN/Inf, and leaves existing files untouched on serialization errors. 56 tests pass, including false-status preservation and nonfinite rejection. |
| Full CPU repeat | `iterations/full-cpu-repair.log`, `metrics.json`, `monte-carlo.npz`, `jacobians.npz` | Complete rerun after repair; scientific failures are preserved as FAIL. No data generation, formula, seed, threshold or sample count changed. |
| Prescribed local sink repair | `metrics.json:sink_sweep` | All five sinks 0/.25/.5/.75/1 failed the noise and distribution requirements at length 128, 2,048 shared trials. |
| Prescribed scale/sink repair | `iterations/scale-sink-sweep.py`, `iterations/scale-sink-sweep.json` | Nine rsqrt(dk) scales × four allowed sinks, 2,048 shared trials. All 36 configurations failed. Scaling is applied to both arms and common noise. No selective successful regime is substituted for the original experiment. |
| Final TPU repeat | `iterations/tpu-repair-launch.log`, `tpu/metrics.json`, `tpu/latencies.json`, `tpu/monte-carlo.npz` | Full repeat from the repaired committed snapshot. Numerical/performance results and failed scientific gates are reported separately in FAIL.md. |

## Dependency tracing

The failure originates in the Phase 2 hypothesis, not Phase 1 AVN. The independent
float64 oracle, JAX CPU path, and TPU float32 path agree. The derivative chain rule
is tested directly against autograd and an independent quotient-rule VJP. No
change to the Phase 1 primitive or epsilon is justified by these failures.

No Phase 3–10 implementation exists. Future Phase 6 kernels and Phase 7–9 models
must retain last-axis uncentered AVN, the default epsilon/sink, the octic exponent,
FP32 intermediates and the full AVN derivative. They must not assume universal
100× quantization robustness, softmax distribution parity, or entropy bounds for
all score vectors. Phase 2 has not authorized advancement to later phases.

Mathematical clarification v2 in `phases/phase2.md` corrects only the exact
sharpness expression, derivative-coordinate interpretation, and finite-precision
simplex comparison. The entropy, 5% distribution, 100× noise and 90% throughput
gates remain unchanged. `theory.md` withdraws the stronger unsupported claims.
