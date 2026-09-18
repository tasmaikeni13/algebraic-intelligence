# Phase 4 PASS — Algebraic Loss Functionals & Information Metrics

Verified 2026-09-18 on CPU and the four-host, 16-chip Cloud TPU v4 Pod slice
`my-tpu-v4` in `us-central2-b`. The corrected OACE implementation is the
strictly proper non-local power score; the older local-only objective and its
scale-mismatched gradient claims are not part of this PASS record.

## Gate inventory

| Gate | Verified outcome | Evidence |
| :--- | :--- | :--- |
| Lean certificates | 1526 jobs; no errors, `sorry`, `admit`, or added axioms | [`lean-build.log`](lean-build.log), [`Loss.lean`](../../formal/AlgebraicTheory/Loss.lean) |
| Zero-transcendental audit | No source, token, or traced-graph violations | [`metrics.json`](metrics.json) |
| Strict propriety | Truth score at most $5.33\times10^{-15}$; minimum sampled competitor regret $9.97\times10^{-3}$; stationarity error $2.22\times10^{-16}$ over $10^5$ soft targets | [`metrics.json`](metrics.json) |
| Probability/composed gradients | Probability gradient matches its formula to $5.78\times10^{-16}$ scaled error; the complete AVN + A-Softmax + OACE score gradient is finite | [`metrics.json`](metrics.json) |
| Label-noise characterization | $10^5$ same-domain logit-gradient trials; analytical/autodiff error $1.11\times10^{-15}$; all finite | [`metrics.json`](metrics.json) |
| Fisher equivalence | Mean Hessian ratio $2.0$ with zero observed error over $10^5$ samples | [`metrics.json`](metrics.json) |
| Classification parity | OACE accuracy $74.75\%$ vs CE $75.25\%$; $0.5$ percentage-point delta | [`metrics.json`](metrics.json) |
| TPU parity | 16/16 FP32 and BF16 configurations passed against the float64 oracle | [`tpu/metrics.json`](tpu/metrics.json) |
| TPU throughput | 8/8 configurations passed the $\ge90\%$ gate; observed ratios $1.0376$–$1.0716$ | [`tpu/metrics.json`](tpu/metrics.json), [`tpu/latencies.json`](tpu/latencies.json) |

## Correct score and gradient contract

For $\alpha=1/8$, the production score is

$$
\mathcal L_{\mathrm{OACE}}(\mathbf p,\mathbf y)=\gamma\left[
8\sum_i y_i p_i^{-1/8}+\frac87\sum_i p_i^{7/8}
-\frac{64}{7}\sum_i y_i^{7/8}\right].
$$

The correction terms make the score strictly proper for hard and soft targets.
Before multiplying by $\gamma$, its probability derivative is

$$
\frac{\partial\mathcal L}{\partial p_i}
=p_i^{-1/8}-y_i p_i^{-9/8}.
$$

This derivative is singular as $p_i\to0$; it is not a bounded probability-space
gradient. At a target probability of $10^{-9}$ with the default $\gamma=2$, the
verified magnitude is $1.3335214308\times10^{10}$. The training-stability gate
instead verifies that the full AVN-bounded A-Softmax composition has finite
score gradients (maximum observed magnitude $6794.73$ in the configured sweep).

The prediction term uses exactly three `rsqrt` operations:

$$
z_1=p_i^{-1/2},\qquad z_2=p_i^{1/4},\qquad z_3=p_i^{-1/8},
\qquad p_i^{7/8}=p_i z_3.
$$

## Same-domain label-noise result

CE and OACE were compared in the same logit domain over $100{,}000$ trials.
The measured gradient-norm variance ratio was $2.278$ (OACE/CE). This value is
descriptive, not a superiority gate. The withdrawn variance-reduction result
compared gradients in different domains and is intentionally not reproduced.

## TPU throughput

Each row used 10 warmups and 100 synchronized repetitions on all 16 chips.

| Dtype | Classes | OACE fwd (ms) | CE fwd (ms) | Fwd ratio | OACE fwd+bwd (ms) | CE fwd+bwd (ms) | Fwd+bwd ratio |
| :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FP32 | 256 | 0.3496 | 0.3701 | 1.0585 | 0.4293 | 0.4522 | 1.0532 |
| FP32 | 1024 | 0.3455 | 0.3630 | 1.0505 | 0.4287 | 0.4450 | 1.0379 |
| FP32 | 4096 | 0.3419 | 0.3654 | 1.0688 | 0.4280 | 0.4539 | 1.0606 |
| FP32 | 32000 | 0.3491 | 0.3655 | 1.0468 | 0.4343 | 0.4506 | 1.0376 |
| BF16 | 256 | 0.3503 | 0.3671 | 1.0479 | 0.4331 | 0.4518 | 1.0431 |
| BF16 | 1024 | 0.3411 | 0.3656 | 1.0716 | 0.4259 | 0.4476 | 1.0508 |
| BF16 | 4096 | 0.3517 | 0.3762 | 1.0697 | 0.4373 | 0.4607 | 1.0535 |
| BF16 | 32000 | 0.3502 | 0.3691 | 1.0540 | 0.4398 | 0.4595 | 1.0447 |

The machine-readable JSON files are authoritative for unrounded values and
source hashes.
