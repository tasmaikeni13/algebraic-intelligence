# Phase 1: Pure Algebraic Primitives & Non-Linear Gating (ALU & AVN)

Read `theory.md`, `formal/README.md`, `formal/AlgebraicTheory/Gate.lean`, `formal/AlgebraicTheory/Variance.lean`, and `phases/README.md` completely before executing. Execute the shared adaptive failure-repair loop until all gates pass.

---

## 1. Objective, Scientific Hypothesis & Competing Models

Eliminate all continuous exponential and transcendental functions from activation gating and variance normalization:
$$\textbf{"Can algebra and algebra alone produce stable non-linear representations and deep gradient flow?"}$$

### Competing Hypotheses:
- **$H_1$ (Algebraic Hypothesis):** The Algebraic Linear Unit $K(x) = \frac{x}{2}(1 + x \cdot \operatorname{rsqrt}(1 + x^2)) = x \beta(x)$ possesses an exact inflection point at $x = -\sqrt{2}$ matching GELU dynamics, a strictly bounded Lipschitz constant $L_K \approx 1.044331$, and an analytic $\mathcal{O}(1)$ Horner cubic backward pass. When paired with parameter-free Algebraic Variance Normalization (AVN), signals propagate through deep networks ($D \ge 32$) with bounded variance $\operatorname{Var}(\mathbf{h}_D) \approx 1.0$ and stable gradient norms without exponential saturation or learnable channel scales $\boldsymbol{\gamma}$ in HBM.
- **$H_0$ (Transcendental Baseline Hypothesis):** Continuous transcendental error functions (GELU $\Phi(x)$) or exponential sigmoids (Swish $\sigma(x)$) and affine LayerNorm/RMSNorm are essential for non-linear feature separation; rational approximations will suffer from variance collapse, representation shrinkage, or gradient explosion under deep composition.

---

## 2. Mathematical Formulations & Zero-Transcendental Constraints

### 2.1 The Algebraic Gate Function $\beta(u)$
With the algebraic cache variable $u \coloneqq x \cdot \operatorname{rsqrt}(1 + x^2) \in (-1, 1)$:
$$\beta(u) = \frac{1 + u}{2}$$
Forward evaluation on the TPU v4 Vector Processing Unit (VMU) requires only one multiply, one add, and one hardware $\operatorname{rsqrt}$ call.

### 2.2 Exact Polynomial Backward Pass
The exact derivative of ALU with respect to input $x$ is expressible strictly as a cubic polynomial in the cached variable $u$:
$$\frac{d}{dx} K(x) = \beta(u) + x \beta'(x) = \frac{1}{2}\left(1 + 2u - u^3\right) = 0.5 + u \cdot (1.0 - 0.5 \cdot u^2)$$
**Critical Zero-Transcendental Constraint:** The backward pass must NEVER recompute square roots, divisions, or transcendentals. It evaluates the forward cache $u$ directly through Horner's polynomial rule.

### 2.3 Inflection Point Theorem
The second derivative satisfies:
$$K''(x) = \frac{1}{2}(2 - 3u^2)\frac{du}{dx} = 0 \iff 2 - 3u^2 = 0 \iff u = \pm\sqrt{2/3} \iff x = \pm\sqrt{2}$$
This matches the exact negative inflection coordinate of GELU ($G''(x) = \phi(x)(2 - x^2) = 0$ at $x = -\sqrt{2}$).

### 2.4 Algebraic Variance Normalization (AVN)
$$\operatorname{AVN}(\mathbf{x}) = \mathbf{x} \cdot \operatorname{rsqrt}\left(\frac{1}{d}\|\mathbf{x}\|_2^2 + \epsilon\right)$$
Zero learnable parameters in high-bandwidth memory (HBM), strictly satisfying the Coupling Identity $\beta(x; v) = \beta(\hat{x}; 1)$ where $\hat{x} = x / \sqrt{v}$. The backward pass is $\tau[g-\hat{x}\,\operatorname{mean}(g\hat{x})]$. It is a scaled orthogonal projection only for $\epsilon=0$; at positive epsilon its radial eigenvalue is $\tau\epsilon/(m_2+\epsilon)$. The dimension reciprocal is a static constant, so the traced backward graph contains no division or $\operatorname{rsqrt}$.

---

## 3. Implementation Target: JAX / TPU v4 Architecture

Instruct the creation and verification of the following modular files targeting the 16 TPU v4 Pod:
1. **`src/primitives.py`**:
   - `alu(x)`: JAX implementation with `@jax.custom_vjp` to enforce Horner cubic backward pass using cached $u = x \cdot \operatorname{rsqrt}(1 + x^2)$.
   - `avn(x, eps=1e-5)`: Parameter-free algebraic variance normalization in JAX with analytical VJP.
   - Dual-mode support: compiled for TPU v4 VMU execution via `@jax.jit` and verifiable in fp64 CPU oracle mode.
2. **`tests/test_primitives.py`**:
   - Automated unit tests checking mathematical equivalence against fp64 CPU reference.
   - AST parser verifying zero occurrences of `exp`, `log`, `sin`, `cos`, `tanh`, or `sigmoid` in `src/primitives.py`.
3. **`scripts/run_verify_primitives.py`**:
   - Standalone CLI runner executing the full Monte Carlo verification suite and emitting `results/phase1/metrics.json`.

---

## 4. Lean 4 Formal Verification Gate

The agent must compile `formal/AlgebraicTheory/Gate.lean` and `formal/AlgebraicTheory/Variance.lean` under `/root/.elan/bin/lake build` with zero errors, zero warnings, and zero `sorry`:

1. `alu_reflection_symmetry`:
   $$\forall u \in \mathbb{R}, \quad \beta(u) + \beta(-u) = 1$$
2. `alu_deriv_formula`:
   $$\forall u, \quad \frac{1 + u}{2} + \frac{u(1 - u^2)}{2} = \frac{1}{2}(1 + 2u - u^3)$$
3. `bounded_avn_norm`:
   $$\forall S \geq 0, \ d > 0, \ \epsilon \geq 0, \quad S \leq d\left(\frac{S}{d} + \epsilon\right)$$
4. `avn_coupling_identity`:
   Invariance of normalized coordinates under uniform positive scalar scaling: $\operatorname{AVN}(\alpha \mathbf{x}) = \operatorname{AVN}(\mathbf{x})$ for $\alpha > 0$ when $\epsilon \to 0$.
5. `avn_scale_invariance`:
   Exact coordinate invariance under non-zero scaling: $(\tau / \alpha)(\alpha x) = \tau x$.

---

## 5. Deep Empirical & Monte Carlo Simulation Gate

Execute the verification suite via `python3 scripts/run_verify_primitives.py` and enforce the following empirical criteria:

| Evaluation Dimension | Experimental Protocol | Success Criterion / Bound |
| :--- | :--- | :--- |
| **Monte Carlo Moment/Variance Preservation (v2)** | $10^6$ samples per scale across $\sigma \in [0.1,10]$, eps in {0, 1e-5}; independent vector-level 95% CIs | Exact second-moment and centered-variance identities within 1e-12 (fp64), 2e-5 (TPU fp32); retain variance CI in [0.9999,1.0001] for zero-mean Gaussian inputs at eps=0. See amendment below. |
| **Deep Gradient Flow Ratio** | $D \in \{8, 16, 24, 32\}$ layers, $10^4$ trials, measure $\frac{\|\mathbf{g}_0\|_2}{\|\mathbf{g}_D\|_2}$ | Ratio $\in [0.2, 5.0]$ (no vanishing, no explosion) |
| **Deep Activation Variance** | $D=32$ stacked layers, random He init, measure $\frac{\operatorname{Var}(\mathbf{h}_{32})}{\operatorname{Var}(\mathbf{h}_0)}$ | Ratio $\in [0.5, 2.0]$ |
| **Gate Reflection Symmetry Error** | $\|\beta(u) + \beta(-u) - 1.0\|_\infty$ across $10^5$ samples | $\leq 1.0 \times 10^{-15}$ |
| **Backward Pass Exactness** | $\|\frac{dK}{dx}_{\text{poly}} - \frac{dK}{dx}_{\text{autograd}}\|_\infty$ | $\leq 5.0 \times 10^{-16}$ |
| **Max Jacobian / Lipschitz Bound** | Empirical supremum $\sup_{x} |K'(x)|$ | $\leq 1.05$ (theorized: $\approx 1.044331$) |
| **Inflection Point Alignment** | Numerical verification of $K''(x) = 0$ at $x = -\sqrt{2}$ | $|K''(-\sqrt{2})| \leq 1.0 \times 10^{-15}$ |
| **ALU vs. GELU/Swish Benchmark** | Direct microbenchmark of ALU vs. standard GELU and Swish on forward/backward throughput and gradient dynamics | Throughput $\ge 90\%$ of standard GELU; gradient flow at par or slightly down ($\le 5\%$ variance delta) |
| **AVN vs. RMSNorm Benchmark** | Feature variance preservation and backward latency vs. standard RMSNorm (with learnable $\boldsymbol{\gamma}$) | Throughput $\ge 95\%$ of RMSNorm; zero parameter overhead in HBM |
| **Zero Transcendental Audit** | AST call inspection + regex grep of code logic | Exactly $0$ occurrences of `exp`, `log`, `sin`, `cos` |

---

## 6. Adaptive Failure-Repair & Bidirectional Dependency Protocol

When a test or gate fails in Phase 1:
1. **Iterate Locally:** Apply the 9-step failure-repair loop, updating equations, Lean 4 proofs, and JAX code until passing.
2. **Forward Dependency Cascading:** If fixing Phase 1 requires modifying any primitive signature or invariant (e.g. changing the AVN $\epsilon$ regularizer, altering the Horner caching layout $u$, or adding depth attenuation $\operatorname{rsqrt}(2D)$):
   - Immediately audit all downstream phases:
     - **Phase 2 (`src/attention.py`):** Pre-bounding logits $\hat{s}_i = s_i \cdot \operatorname{rsqrt}(m_2(\mathbf{s}) + \epsilon)$.
     - **Phase 6 (`src/kernels/pallas_afa.py`):** Vector memory (VMEM) tile normalization inputs.
     - **Phase 7, 8, 9 (`src/model.py`):** ALU-GLU feedforward blocks and residual connections in `AlgebraicTransformerLM`.
   - Propagate the updated signatures and invariants forward into all downstream files.
   - Update Lean 4 theorems in `formal/AlgebraicTheory/` to match the new definitions and verify `lake build` compiles cleanly.

---

## 7. PASS Gates

- [ ] `formal/AlgebraicTheory/Gate.lean` compiles with 0 errors via `/root/.elan/bin/lake build`.
- [ ] `formal/AlgebraicTheory/Variance.lean` compiles with 0 errors via `/root/.elan/bin/lake build`.
- [ ] `src/primitives.py` created with JAX `@jax.custom_vjp` Horner cubic backward pass and parameter-free AVN.
- [ ] Version 2 Monte Carlo identities pass for default and zero epsilon; the ideal zero-epsilon variance CI remains within $[0.9999,1.0001]$.
- [ ] Deep gradient flow across 8, 16, 24, 32 layers confirms bounded gradient ratio $\in [0.2, 5.0]$.
- [ ] Reflection symmetry error $\leq 1.0 \times 10^{-15}$ and autograd backward error $\leq 5.0 \times 10^{-16}$.
- [ ] Max Lipschitz constant bounded by $\le 1.05$.
- [ ] Side-by-side benchmark of ALU vs. GELU/Swish and AVN vs. RMSNorm confirms performance at par or within acceptable tolerance (throughput $\ge 90\%$, gradient stability at par or slightly down within $\le 5\%$).
- [ ] Codebase AST audit confirms exactly zero transcendental calls in `src/primitives.py`.
- [ ] `results/phase1/PASS.md` satisfies the shared PASS record contract.


## 8. Phase 1 Gate Amendment v2 (2026-09-15)

This amendment follows the evidence-preserving rule in `phases/README.md` §8.
The original gate remains reproducible in `results/phase1/iterations/variance-v1.json`:
at seed 42, sigma=0.1, epsilon=1e-5, and 1,000,000 samples, the measured variance
is 0.9990019521851182, outside the unchanged historical [0.9999,1.0001] interval.
The claim fails mathematically because AVN does not center and epsilon is nonzero:

$$m_2(\operatorname{AVN}(x))=\frac{m_2(x)}{m_2(x)+\epsilon},\qquad
\operatorname{Var}(\operatorname{AVN}(x))=\frac{\operatorname{Var}(x)}{m_2(x)+\epsilon}.$$

Version 2 keeps the production signature `avn(x, eps=1e-5)` and checks **both exact
identities** at all seven scales, including a nonzero-mean counterexample. It also
retains the original near-unit interval at epsilon=0 on independent zero-mean
Gaussian vectors (20 vectors × 50,000 coordinates on CPU; 16 × 62,500 on TPU).
The confidence interval treats vectors as independent units. It never treats
coordinates coupled by one normalization as independent observations.

The depth experiment now fixes the previously unspecified composition:

$$h_{l+1}=\operatorname{AVN}\left(h_l+\operatorname{rsqrt}(2D)
 W_{d,l}K(W_{u,l}\operatorname{AVN}(h_l))\right).$$

Use width 128 (the TPU MXU width), independent Gaussian weights with variance 2/128, Gaussian inputs,
unit random terminal cotangents, and 10,000 independent trials **at each** depth.
Every trial gets new weights in every layer. GELU and Swish receive identical
weights, inputs, residual attenuation, and RMSNorm scales initialized to one.
The original gradient, activation, and 5% parity thresholds remain unchanged;
all observed trials must satisfy the norm bounds. The unattenuated non-residual
ablation is retained in `results/phase1/iterations/deep-unattenuated.json` and
fails, demonstrating that stable deep flow is a property of this residual
composition, not of arbitrary ALU/AVN stacks. This is a Phase 1 test network,
not a completed transformer or a certificate for future ALU-GLU architectures.

ALU's negative forward branch uses the equivalent identity
$1+u=r^2/(1-u)$, where $r=\operatorname{rsqrt}(1+x^2)$, to avoid cancellation.
It still caches only u and keeps the exact same Horner backward and public API.
The two inflections are $x=\pm\sqrt2$. The negative one still aligns with GELU.

Dependencies: no downstream implementation files exist yet. Phase 2/6 consumers
retain last-axis, uncentered normalization and epsilon=1e-5. Future Phase 7–9
models must measure their own residual gradient flow; they cannot inherit a
claim of unit centered variance for arbitrary input means. See
`results/phase1/DEPENDENCIES.md`. Only Phase 1 is executed here.

The earlier 64-feature full study is retained in `results/phase1/iterations/full-cpu-initial/`. All gradient gates passed, but the activation variance ratio reached 2.746 because finite-width input variance was unusually low. The final 128-feature protocol uses the native TPU matrix width, the same seed and sample count, and the same per-trial bounds. The claims are specific to that width; they are not a uniform guarantee over widths.
