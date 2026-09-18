# Phase 5 PASS — Algebraic Optimization & Rational Scheduling

Verified 2026-09-18 on CPU and the four-host, 16-chip Cloud TPU v4 Pod slice
`my-tpu-v4` in `us-central2-b`.

## Gate inventory

| Gate | Verified outcome | Evidence |
| :--- | :--- | :--- |
| Lean certificates | 1526 jobs; no errors, `sorry`, `admit`, or added axioms | [`lean-build.log`](lean-build.log), [`Curvature.lean`](../../formal/AlgebraicTheory/Curvature.lean) |
| Zero-transcendental audit | No source, token, or traced-graph violations | [`metrics.json`](metrics.json) |
| Optax-compatible default | Default update matches $\hat m/(\sqrt{\hat v}+\epsilon)$ while constructing $\sqrt{\hat v}$ as $\hat v\operatorname{rsqrt}(\hat v)$ with an explicit zero branch | [`src/optimizer.py`](../../src/optimizer.py) |
| Compiled radical audit | Every ARDS step has 0 raw `sqrt` instructions and 2 `rsqrt` instructions in FP32 and BF16 | [`tpu/metrics.json`](tpu/metrics.json), [`tpu/float32_ards_step.mlir`](tpu/float32_ards_step.mlir) |
| Ill-conditioned sweep | Minimum and mean loss reduction both $100\%$ across $10^4$ trials | [`metrics.json`](metrics.json) |
| Non-convex parity | Rosenbrock delta $-2.3923\%$; Rastrigin delta $+0.0815\%$ versus cosine | [`metrics.json`](metrics.json) |
| ARDS properties | Strict decay after warmup; asymptotic ratio $1.0009509$ | [`metrics.json`](metrics.json) |
| TPU numerical parity | 8/8 FP32/BF16 configurations passed | [`tpu/metrics.json`](tpu/metrics.json) |
| TPU throughput | 4/4 configurations passed the $\ge90\%$ gate; ratios $0.9973$–$1.0029$ | [`tpu/metrics.json`](tpu/metrics.json), [`tpu/latencies.json`](tpu/latencies.json) |

## Optimizer contract

`algebraic_adamw(..., use_rsqrt=False)` preserves the standard Optax/AdamW
denominator $\sqrt{\hat v}+\epsilon$. It avoids a raw square-root opcode via

$$
\sqrt{\hat v}=\begin{cases}
\hat v\operatorname{rsqrt}(\hat v), & \hat v>0,\\
0, & \hat v=0.
\end{cases}
$$

The optional `use_rsqrt=True` mode retains the separate algebraic variant
$\hat m\operatorname{rsqrt}(\hat v+\epsilon^2)$. Both modes use the permitted
inverse-square-root radical. Bias corrections use fixed-width integer binary
exponentiation, and weight decay remains decoupled.

ARDS evaluates

$$
\eta(t)=\eta_{\max}\min\left(1,\frac{t}{T_{\mathrm{warm}}}\right)
\operatorname{rsqrt}\left(1+\alpha\left[
\frac{\max(0,t-T_{\mathrm{warm}})}{T_{\mathrm{decay}}}\right]^2\right),
$$

with strict post-warmup monotonicity and the verified $\mathcal O(1/t)$ tail.

## TPU throughput

Each row used 10 warmups and 100 synchronized repetitions on all 16 chips.

| Dtype | Shape | ARDS step (ms) | Cosine step (ms) | Cosine / ARDS |
| :---: | :---: | ---: | ---: | ---: |
| FP32 | $1024^2$ | 0.603930 | 0.602290 | 0.997284 |
| FP32 | $2048^2$ | 0.630229 | 0.631385 | 1.001834 |
| BF16 | $1024^2$ | 0.602730 | 0.603195 | 1.000772 |
| BF16 | $2048^2$ | 0.613005 | 0.614800 | 1.002928 |

The machine-readable JSON files are authoritative for unrounded values and
source hashes.
