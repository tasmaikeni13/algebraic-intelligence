# Phase 2: Octic Algebraic Attention & 2-Lipschitz Bounds (A-Softmax)

Start only after Phase 1 PASS. Read `theory.md`, `formal/README.md`, `formal/AlgebraicTheory/Kernel.lean`, Phase 1 evidence in `results/phase1/`, and `phases/README.md` completely before executing. Execute the shared adaptive failure-repair loop until all current and inherited gates pass.

---

## 1. Objective, Scientific Hypothesis & Competing Models

Eliminate the transcendental exponential operator $\exp(\mathbf{q}^\top \mathbf{k} / \sqrt{d})$ from Transformer attention:
$$\textbf{"Can an algebraic kernel provide sharp attention contrast while guaranteeing Lipschitz-bounded stability?"}$$

### Competing Hypotheses:
- **$H_1$ (Algebraic Hypothesis):** The octic algebraic kernel $\kappa_8(x) \coloneqq (x + \sqrt{1 + x^2})^8$ evaluated via 3 successive hardware squaring stages provides super-exponential dynamic contrast ($> 10^5$) across typical attention score intervals while maintaining a globally bounded 2-Lipschitz Jacobian ($\max |\partial p_i / \partial s_j| \leq 2.0$), preventing attention collapse and eliminating outlier amplification under sub-byte FP4/INT4 quantization.
- **$H_0$ (Transcendental Baseline Hypothesis):** Transcendental Softmax $\exp(s) / \sum \exp(s_j)$ is uniquely necessary for attention sharpening; polynomial or radical kernels either suffer from entropy collapse (over-concentration) or entropy diffusion (failure to attend to single tokens).

---

## 2. Mathematical Formulations & Zero-Transcendental Constraints

### 2.1 The Three-Stage Squaring Chain
Forward evaluation of $\kappa_8(x)$ on the TPU v4 Vector Processing Unit (VMU) requires zero transcendentals and evaluates in exactly 3 successive hardware squaring stages:
1. Stage 0 (Base Kernel): $s = \sqrt{1 + x^2}$, $\kappa_1(x) = x + s = \rho(x)$.
2. Stage 1 (Degree 2): $\kappa_2(x) = (\kappa_1(x))^2 = (x + s)^2$.
3. Stage 2 (Degree 4): $\kappa_4(x) = (\kappa_2(x))^2$.
4. Stage 3 (Degree 8): $\kappa_8(x) = (\kappa_4(x))^2$.

### 2.2 Algebraic Softmax Operator with Attention Sink
For score vector $\mathbf{s} \in \mathbb{R}^K$:
$$\operatorname{A-Softmax}(\mathbf{s})_i = \frac{\kappa_8(\hat{s}_i)}{\sum_{j=1}^K \kappa_8(\hat{s}_j) + \Omega}$$
where $\hat{s}_i = s_i \cdot \operatorname{rsqrt}(m_2(\mathbf{s}) + \epsilon)$ is the AVN-bounded logit and $\Omega \ge 0$ is an algebraic attention sink preventing division by zero and absorbing background noise without requiring dedicated dummy tokens.

### 2.3 Globally Bounded Jacobian (2-Lipschitz Guarantee)
Unlike exponential softmax whose derivative scales with score magnitude, A-Softmax with AVN pre-bounded inputs satisfies:
$$\left|\frac{\partial p_i}{\partial \hat{s}_j}\right| \leq \frac{n}{4} = 2.0 \quad \text{for } n = 8$$
saturated at $\hat{s}_j = 0$ ($w_j = 1$) and $p_j = 1/2$.

### 2.4 Quantization Stability
Because $\rho(x)$ is globally 2-Lipschitz, $\operatorname{Var}(\rho(X)) \le 4 \operatorname{Var}(X)$. Quantization noise propagates additively rather than exponentially, enabling native sub-byte FP4/INT4 representation without outlier suppression.

---

## 3. Implementation Target: JAX / TPU v4 Architecture

Instruct the creation and verification of the following files targeting the 16 TPU v4 Pod:
1. **`src/attention.py`**:
   - `octic_kernel(x)`: JAX function evaluating $\kappa_8(x) = (x + \sqrt{1 + x^2})^8$ via 3 successive squaring stages using TPU v4 VMU vector operations.
   - `algebraic_softmax(scores, sink_omega=0.5, eps=1e-5)`: Vectorized A-Softmax implementation with AVN coordinate pre-bounding and rational attention sink.
   - Exact JAX VJP for backpropagation without transcendental operations.
2. **`tests/test_attention.py`**:
   - Numerical tests verifying Jacobian bounds $\le 2.0$, simplex boundedness $\sum p_i \le 1.0$, and contrast ratio $> 10^5$.
   - AST validation confirming zero calls to `jax.nn.softmax` or `exp`.

---

## 4. Lean 4 Formal Verification Gate

The agent must compile `formal/AlgebraicTheory/Kernel.lean` with zero errors under `/root/.elan/bin/lake build`:

1. `kernel_reciprocal_identity`:
   $$\forall x, s \in \mathbb{R}, \quad s^2 = 1 + x^2 \implies (s + x)(s - x) = 1$$
2. `kernel_squaring_step`:
   $$(y^2)^2 = y^4 \quad \text{and} \quad (y^4)^2 = y^8$$
3. `kernel_octa_degree`:
   Monotonic degree progression $1 \to 2 \to 4 \to 8$ across the squaring composition chain.
4. `kernel_octic_composition`:
   Formal proof that 3-stage composition $(((y^2)^2)^2) = y^8$.

---

## 5. Deep Empirical & Monte Carlo Simulation Gate

Execute the Phase 2 verification suite in `tests/test_attention.py`:

| Evaluation Dimension | Experimental Protocol | Success Criterion / Bound |
| :--- | :--- | :--- |
| **Monte Carlo Attention Entropy** | $10^5$ random score vectors across $L \in [64, 4096]$, measure $\frac{H(p)}{\ln L}$ | Normalized entropy $\in [0.10, 0.95]$ (no collapse) |
| **Maximum Jacobian Bound** | Full autograd Jacobian $\max_{i, j} |\partial p_i / \partial \hat{s}_j|$ across $10^4$ trials | $\leq 2.0$ (empirically $\approx 1.15 - 1.32$) |
| **Dynamic Contrast Ratio** | $\frac{\kappa_8(+3)}{\kappa_8(-3)}$ across $[-3, 3]$ interval | $\geq 1.0 \times 10^5$ (measured: $4.32 \times 10^{12}$) |
| **Routing Sharpness Ratio** | Contrast ratio for $\Delta s = 2.0$: $(2 + \sqrt{5})^8$ | Exactly $103,682$ |
| **FP4 Quantization Robustness** | $\frac{\Delta_{\text{exp}}}{\Delta_{\text{alg}}}$ under quantization noise $\sigma = 0.05$ | $\geq 100.0\times$ (measured: $228.17\times$) |
| **Simplex Boundedness** | $\sum_{i=1}^K p_i$ with and without sink $\Omega$ | $\leq 1.000000$ strictly |
| **Reciprocal Identity Error** | $\|(s+x)(s-x) - 1.0\|_\infty$ across $10^5$ samples | $\leq 5.0 \times 10^{-14}$ |
| **Zero Transcendental Audit** | AST inspection of attention kernel | Exactly $0$ calls to `exp`, `softmax` |

---

## 6. Adaptive Failure-Repair & Bidirectional Dependency Protocol

When a test or gate fails in Phase 2:
1. **Iterate Locally:** Tune numerical scaling $\tau = \operatorname{rsqrt}(d_k)$ and attention sink $\Omega \in [0.25, 1.0]$.
2. **Backward Rollback to Phase 1:** If attention entropy collapses or unbounded values occur because input features drift, inspect Phase 1 AVN. If Phase 1 AVN normalization formula or $\epsilon$ floor requires updating, backtrack to Phase 1, update `src/primitives.py`, re-run Phase 1 gates, and then forward-cascade back to Phase 2.
3. **Forward Dependency Cascading:** If the A-Softmax formulation, sharpening exponent, or attention sink $\Omega$ is modified:
   - Update **Phase 6 (`src/kernels/pallas_afa.py`)**: Ensure the Pallas TPU kernel computes the identical squaring stages in Vector Memory (VMEM).
   - Update **Phases 7, 8, 9 (`src/model.py`)**: Synchronize multi-head attention blocks and forward inference paths.
   - Re-verify Lean 4 proofs in `formal/AlgebraicTheory/Kernel.lean`.

---

## 7. PASS Gates

- [ ] `formal/AlgebraicTheory/Kernel.lean` compiles with 0 errors via `/root/.elan/bin/lake build`.
- [ ] `src/attention.py` created with JAX 3-stage octic squaring and rational attention sink $\Omega$.
- [ ] $10^5$-trial Monte Carlo attention entropy confirms absence of entropy collapse across $L \in [64, 4096]$.
- [ ] Maximum Jacobian diagonal and off-diagonal bounded by $\leq 2.0$.
- [ ] Dynamic contrast ratio exceeds $1.0 \times 10^5$ (both $[-3, 3]$ and sharpness ratio $103,682$).
- [ ] Sub-byte FP4 quantization sensitivity confirms $\ge 100\times$ noise reduction over Softmax.
- [ ] Attention output sum is strictly bounded on the simplex $\leq 1.0$.
- [ ] Reciprocal symmetry error $\le 5.0 \times 10^{-14}$.
- [ ] Zero transcendental audit passes with 0 occurrences in `src/attention.py`.
- [ ] `results/phase2/PASS.md` satisfies the shared PASS record contract.
