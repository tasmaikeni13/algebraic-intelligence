# Phase 1: Pure Algebraic Primitives & Non-Linear Gating (ALU & AVN)

Read `theory.md`, `formal/README.md`, `formal/AlgebraicTheory/Gate.lean`, `formal/AlgebraicTheory/Variance.lean`, and `phases/AUTONOMY_PROTOCOL.md` completely before executing. Execute the shared failure-repair loop until all gates pass.

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
Forward evaluation requires only one multiply, one add, and one hardware $\operatorname{rsqrt}$ call.

### 2.2 Exact Polynomial Backward Pass
The exact derivative of ALU with respect to input $x$ is expressible strictly as a cubic polynomial in the cached variable $u$:
$$\frac{d}{dx} K(x) = \beta(u) + x \beta'(x) = \frac{1}{2}\left(1 + 2u - u^3\right) = 0.5 + u \cdot (1.0 - 0.5 \cdot u^2)$$
**Critical Zero-Transcendental Constraint:** The backward pass must NEVER recompute square roots, divisions, or transcendentals. It evaluates the forward cache $u$ directly through Horner's polynomial rule.

### 2.3 Inflection Point Theorem
The second derivative satisfies:
$$K''(x) = \frac{1}{2}(2 - 3u^2)\frac{du}{dx} = 0 \iff 2 - 3u^2 = 0 \iff u = -\sqrt{2/3} \iff x = -\sqrt{2}$$
This matches the exact negative inflection coordinate of GELU ($G''(x) = \phi(x)(2 - x^2) = 0$ at $x = -\sqrt{2}$).

### 2.4 Algebraic Variance Normalization (AVN)
$$\operatorname{AVN}(\mathbf{x}) = \mathbf{x} \cdot \operatorname{rsqrt}\left(\frac{1}{d}\|\mathbf{x}\|_2^2 + \epsilon\right)$$
Zero learnable parameters in HBM, strictly satisfying the Coupling Identity $\beta(x; v) = \beta(\hat{x}; 1)$ where $\hat{x} = x / \sqrt{v}$. The backward pass is an orthogonal projection along $\hat{\mathbf{x}}$ computable in dense matrix-vector operations with zero divisions and zero $\operatorname{rsqrt}$.

---

## 3. Lean 4 Formal Verification Gate

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

## 4. Deep Empirical & Monte Carlo Simulation Gate

Execute the verification suite in `analysis/verify_algebraic_primitives.py` and enforce the following empirical criteria:

| Evaluation Dimension | Experimental Protocol | Success Criterion / Bound |
| :--- | :--- | :--- |
| **Monte Carlo Variance Preservation** | $10^6$ samples across $\sigma \in [0.1, 10.0]$, measure $\operatorname{Var}(\operatorname{AVN}(\mathbf{x}))$ | $\operatorname{Var} \in [0.9999, 1.0001]$ ($95\%$ CI) |
| **Deep Gradient Flow Ratio** | $D \in \{8, 16, 24, 32\}$ layers, $10^4$ trials, measure $\frac{\|\mathbf{g}_0\|_2}{\|\mathbf{g}_D\|_2}$ | Ratio $\in [0.2, 5.0]$ (no vanishing, no explosion) |
| **Deep Activation Variance** | $D=32$ stacked layers, random He init, measure $\frac{\operatorname{Var}(\mathbf{h}_{32})}{\operatorname{Var}(\mathbf{h}_0)}$ | Ratio $\in [0.5, 2.0]$ |
| **Gate Reflection Symmetry Error** | $\|\beta(u) + \beta(-u) - 1.0\|_\infty$ across $10^5$ samples | $\leq 1.0 \times 10^{-15}$ |
| **Backward Pass Exactness** | $\|\frac{dK}{dx}_{\text{poly}} - \frac{dK}{dx}_{\text{autograd}}\|_\infty$ | $\leq 5.0 \times 10^{-16}$ |
| **Max Jacobian / Lipschitz Bound** | Empirical supremum $\sup_{x} |K'(x)|$ | $\leq 1.05$ (theorized: $\approx 1.044331$) |
| **Inflection Point Alignment** | Numerical verification of $K''(x) = 0$ at $x = -\sqrt{2}$ | $|K''(-\sqrt{2})| \leq 1.0 \times 10^{-15}$ |
| **Zero Transcendental Audit** | AST call inspection + regex grep of code logic | Exactly $0$ occurrences of `exp`, `log`, `sin`, `cos` |

---

## 5. Autonomous Failure Ledger & Self-Correction Playbook

- **Symptom: Activation variance expands across depth ($D \ge 16$):**
  - *Root Cause:* Absence of AVN pre-bounding before projection layers.
  - *Correction:* Enforce pre-layer AVN normalization: $\mathbf{h}_{\ell+1} = \mathbf{h}_\ell + \operatorname{ALU}(\mathbf{W}_1 \operatorname{AVN}(\mathbf{h}_\ell)) \odot \mathbf{W}_2 \operatorname{AVN}(\mathbf{h}_\ell)$.
- **Symptom: Backward pass autograd mismatch ($> 5.0 \times 10^{-16}$):**
  - *Root Cause:* Forward cache variable $u$ precision mismatch or premature rounding in Horner sequence.
  - *Correction:* Ensure $u = x / \sqrt{1 + x^2}$ is evaluated in FP64 and the Horner sequence `0.5 + u * (1.0 - 0.5 * u * u)` is computed without intermediate casting.
- **Symptom: Gradient vanishing across 32 layers:**
  - *Root Cause:* Un-damped residual accumulation causing effective weights to decay.
  - *Correction:* Apply rational depth scaling $\operatorname{rsqrt}(2D)$ on residual additions.

---

## PASS Gates

- [ ] `formal/AlgebraicTheory/Gate.lean` compiles with 0 errors via `/root/.elan/bin/lake build`.
- [ ] `formal/AlgebraicTheory/Variance.lean` compiles with 0 errors via `/root/.elan/bin/lake build`.
- [ ] $10^6$-sample Monte Carlo variance simulation passes within $[0.9999, 1.0001]$.
- [ ] Deep gradient flow across 8, 16, 24, 32 layers confirms bounded gradient ratio $\in [0.2, 5.0]$.
- [ ] Reflection symmetry error $\leq 1.0 \times 10^{-15}$ and autograd backward error $\leq 5.0 \times 10^{-16}$.
- [ ] Max Lipschitz constant bounded by $\le 1.05$.
- [ ] Codebase audit confirms exactly zero transcendental calls in ALU and AVN.
- [ ] `results/phase1/PASS.md` satisfies the shared PASS record contract.
