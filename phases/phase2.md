# Phase 2 — Primitive Mechanism Separation & Matched Baselines

Start only after Phase 1 PASS. Read `theory.md`, Phase 1 evidence in `results/phase1/`, and `phases/AUTONOMY_PROTOCOL.md` completely before acting. Execute the failure-repair loop until every current and inherited gate passes.

The scientific purpose of this phase is **not to artificially force the Algebraic Stack to win**. It is to cleanly separate and isolate the exact mechanisms of each algebraic primitive against its standard transcendental counterpart under fair, matched resource budgets.

---

## 1. Matched Baselines & Primitive Ablations

Implement equation-verified versions of both algebraic and transcendental counterparts:

| Component Category | Algebraic Candidate | Matched Transcendental Baseline | Minimal Control / Ablation |
| :--- | :--- | :--- | :--- |
| **Activation Function** | Algebraic Linear Unit (ALU) | GELU ($x \Phi(x)$) & Swish ($x \sigma(x)$) | ReLU and Identity |
| **Attention Kernel** | Octic A-Softmax ($\kappa_8(x)$, $\Omega$) | Exponential Softmax ($\exp(x)$) | Linear Attention ($\phi(x) = 1 + x$) |
| **Positional Encoding** | AGO (Cayley transform on $\mathfrak{so}(2)$) | RoPE (Rotary $\cos \theta, \sin \theta$) | No Positional Encoding (Absolute) |
| **Normalization** | AVN (Parameter-free $\mathrm{rsqrt}$) | RMSNorm & LayerNorm (Learned $\boldsymbol{\gamma}$) | Un-normalized Baseline |
| **Loss Functional** | OACE ($\mathcal{L}_{1/8}$) & AD ($D_A$) | Cross-Entropy ($-\ln p$) & KL Divergence | Quadratic Loss ($\frac{1}{2}\|y - p\|^2$) |
| **Curvature Optimizer** | ACO (Factorized $\mathcal{O}(d_{\text{out}} + d_{\text{in}})$) | AdamW (Full $\mathcal{O}(d_{\text{out}} \cdot d_{\text{in}})$) | Standard SGD with Momentum |
| **LR Schedule** | ARDS (Rational $\operatorname{rsqrt}$) | Cosine Annealing ($\frac{1}{2}(1 + \cos)$) | Step Decay / Constant |

Verify each baseline equation on hand-computable toy cases before benchmarking.

---

## 2. Four Fair Budget Views

Never collapse evaluations into a single uncalibrated score. Evaluate every primitive pair across **four distinct resource views**:
1. **View 1: Same Feature Dimension:** Identical tensor widths ($d, d_k, d_v, d_{\text{ff}}$) and layer depths.
2. **View 2: Same Parameter Count:** Calibrated within $\pm 1\%$ parameter parity. (Note: AVN has 0 parameters, freeing budget for projection layers).
3. **View 3: Same Live-State Memory (HBM Bytes):** Compare active activation cache and optimizer state footprint. (ACO stores only $d_{\text{out}} + d_{\text{in}}$ curvature scalars vs $d_{\text{out}} \cdot d_{\text{in}}$ for AdamW).
4. **View 4: Same Measured FLOPs:** Equalized floating-point operation budgets.

---

## 3. Targeted Falsification Suites

### 3.1 Sub-Byte Quantization Robustness (FP4 / INT4 / FP8)
- Inject stochastic uniform quantization noise $\Delta \sim \mathcal{U}(-\delta, \delta)$ with $\delta \in [0.01, 0.20]$ onto attention logits and activations.
- Measure output probability displacement $\|\mathbf{p}_{\text{quant}} - \mathbf{p}_{\text{exact}}\|_1$ for A-Softmax vs. Exponential Softmax.
- **Hypothesis to Falsify:** A-Softmax's 2-Lipschitz property prevents outlier variance explosion, delivering $\ge 100\times$ lower output displacement than Exponential Softmax under FP4 noise.

### 3.2 Dynamic Contrast & Attention Sink Regime
- Measure the dynamic contrast ratio on logit intervals $[-s_{\max}, s_{\max}]$ for $s_{\max} \in [1.0, 5.0]$.
- Evaluate the role of the attention sink $\Omega \in [0.0, 1.0]$ in suppressing background noise when queries match no relevant key.
- Contrast against standard Softmax with artificial sink tokens.

### 3.3 Deep Signal Propagation & Variance Stability
- Stack $D \in \{8, 16, 24, 32, 48\}$ residual blocks:
  - ALU + AVN vs. GELU + RMSNorm vs. Swish + LayerNorm.
  - Initialize with He normal initialization. Measure activation variance $\operatorname{Var}(\mathbf{h}_\ell)$ and gradient norm ratio $\|\mathbf{g}_0\|_2 / \|\mathbf{g}_D\|_2$.
- Confirm whether AVN pre-bounding guarantees bounded variance without learnable channel scales.

### 3.4 Long-Context Positional Equivariance & Drift
- Sweep sequence lengths $L \in [128, 8192]$:
  - Compare AGO Cayley rotation powers $(\mathbf{R}_k)^m$ vs. RoPE trigonometric rotations $(\cos m\theta, \sin m\theta)$.
  - Measure matrix norm preservation $|\|\mathbf{R}^m \mathbf{v}\|_2 - \|\mathbf{v}\|_2|$, determinant error $|\det(\mathbf{R}^m) - 1.0|$, and relative shift equivariance error $\|\mathbf{R}_m^\top \mathbf{R}_n - \mathbf{R}_{n-m}\|_\infty$.

### 3.5 Optimizer Memory Footprint & Ill-Conditioned Convergence
- Compare ACO vs. AdamW across weight matrix dimensions $d \in \{512, 1024, 2048, 4096, 8192\}$.
- Measure exact HBM state bytes: verify the factorized second-moment compression factor $\frac{d_{\text{out}} d_{\text{in}}}{d_{\text{out}} + d_{\text{in}}} \ge 1024\times$ at $d=4096$.
- Benchmark convergence speed on ill-conditioned non-convex functions (Rosenbrock, Rastrigin) with condition numbers $\kappa \in [10^2, 10^5]$.

---

## 4. Research Repair Requirement

If any empirical regime reveals that an algebraic primitive underperforms its transcendental baseline by more than the preregistered threshold:
1. Research the underlying mechanism in the primary literature (e.g. eigenvalue spectra of factorized preconditioning, contrast curves of polynomial kernels).
2. Derive the mathematical explanation and determine whether the discrepancy arises from an uncalibrated scale parameter (e.g. attention temperature, FFN expansion ratio, or rational warmup schedule).
3. If an algebraic adjustment is derived, formalize it in Lean 4, implement it in the fp64 oracle, and update all regression suites. If a regime legitimately favors transcendentals, preserve the negative result honestly in the PASS report.

---

## PASS Gates

- [ ] Every baseline and ablation passes its own standalone equation verification tests.
- [ ] Sub-byte FP4 quantization benchmark confirms $\ge 100\times$ lower output distribution displacement for A-Softmax over Exponential Softmax.
- [ ] Deep signal propagation across 32 layers confirms bounded variance $\operatorname{Var}(\mathbf{h}_{32}) \in [0.5, 2.0]$ for ALU + AVN without learnable $\boldsymbol{\gamma}$ parameters.
- [ ] AGO Cayley rotations maintain exact shift equivariance error $\le 1.0 \times 10^{-6}$ up to sequence length $L = 4096$.
- [ ] ACO demonstrates $\ge 1024\times$ second-moment memory compression at $d = 4096$ while converging within $5\%$ of AdamW on non-convex test surfaces.
- [ ] OACE loss gradient remains strictly bounded without gradient clipping on extreme label noise ($\epsilon = 0.3$).
- [ ] All four budget views (dimension, parameters, HBM bytes, FLOPs) are fully populated and reported.
- [ ] All Lean 4 proofs compile cleanly via `/root/.elan/bin/lake build`.
- [ ] All Phase 0 and Phase 1 inherited gates pass.
- [ ] `results/phase2/PASS.md` satisfies the shared PASS record contract.
