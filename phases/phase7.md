# Phase 7: Architectural Integration & Pilot Pretraining (10M–30M LM)

## 1. Objective & Research Scope
Assemble all 12 modules of the Algebraic Stack into a unified, end-to-end autoregressive language model architecture: `AlgebraicTransformerLM`.
- Conduct pilot pretraining runs at scale (10M–30M parameters) across $10^5$ optimization steps.
- Demonstrate unbroken training stability with **zero NaNs, zero Infs, and zero loss spikes** under pure algebraic optimization (ACO + ARDS + OACE).
- Establish empirical scaling curves and calibrate hyperparameter ranges prior to frontier 125M pretraining.

---

## 2. Model Architecture & Hyperparameter Configuration

### 2.1 Complete Algebraic Transformer Stack Specification
- **Embedding:** Pure byte embedding with AVN normalization.
- **Attention Layer:** Multi-head Algebraic Attention with A-Softmax ($n=8$, $\Omega = 1.0$), pre-attention AVN, and AGO Cayley rotary encoding.
- **Feed-Forward Layer:** ALU-GLU with expansion ratio $8/3$, pre-FFN AVN, and residual connection.
- **Output Head:** Un-tied linear projector directly evaluated under Octo-Algebraic Cross-Entropy ($\mathcal{L}_{1/8}$).
- **Optimizer:** Full Algebraic Curvature Optimizer (ACO) with ARDS rational decay schedule.

### 2.2 Baseline Matching Configuration (25M Parameters)
| Hyperparameter | Algebraic Model | Transcendental Baseline |
| :--- | :--- | :--- |
| **Layers** | 8 | 8 |
| **Hidden Dimension ($d$)** | 512 | 512 |
| **Attention Heads** | 8 | 8 |
| **Head Dimension ($d_k$)** | 64 | 64 |
| **FFN Intermediate Dimension** | 1365 (ALU-GLU) | 1365 (SwiGLU) |
| **Context Length** | 1024 | 1024 |
| **Activation** | ALU | GELU |
| **Attention Kernel** | A-Softmax ($\kappa_8$) | Exponential Softmax |
| **Positional Encoding** | AGO (Cayley $\mathrm{SO}(2)$) | RoPE ($\sin/\cos$) |
| **Loss Function** | OACE ($\mathcal{L}_{1/8}$) | Cross-Entropy ($-\sum y \ln p$) |
| **Optimizer** | ACO (Factorized $\mathcal{O}(d_{\text{out}} + d_{\text{in}})$) | AdamW ($\beta_1=0.9, \beta_2=0.95$) |
| **Learning Rate Schedule** | ARDS (Rational $\operatorname{rsqrt}$) | Cosine Annealing |

---

## 3. Empirical Passing Gate

The agent must execute `analysis/pilot_pretraining.py` on WikiText-103 or TinyStories and satisfy:

| Metric | Target Value | Tolerance / Bound |
| :--- | :--- | :--- |
| **Optimization Stability** | Loss spikes ($\Delta \mathcal{L} > 2.0$) over $10^5$ steps | Exactly $0$ spikes |
| **Gradient Norm Stability** | $\max_t \|\mathbf{G}_t\|_2$ | $\leq 5.0$ without explicit gradient clipping |
| **Perplexity Parity vs. Baseline** | $\frac{\operatorname{PPL}_{\text{algebraic}}}{\operatorname{PPL}_{\text{baseline}}}$ at step $50,000$ | $\leq 1.08$ (within 8% parity) |
| **Peak VRAM Consumption** | Model + Optimizer State | $\geq 25\%$ reduction vs. AdamW baseline |
| **Zero Transcendental Audit** | Automated AST inspection of execution trace | Exactly $0$ transcendental operations |

---

## 4. Failure Modes & Self-Correction Playbook

- **Symptom: Loss stalls early at high perplexity ($> 50$):**
  *Root Cause:* The attention sink constant $\Omega$ is too large relative to the score sum $\sum \kappa_8(s)$, diluting attention weights.
  *Correction:* Calibrate $\Omega \in [0.1, 1.0]$ or dynamically scale $\Omega$ inversely with sequence length $L$.
- **Symptom: Residual stream variance explodes at depth $> 12$:**
  *Root Cause:* Accumulating un-scaled residual additions: $\mathbf{x}_{\ell+1} = \mathbf{x}_\ell + \operatorname{SubLayer}(\mathbf{x}_\ell)$.
  *Correction:* Apply AVN normalization to the residual input before adding, or scale the residual branch by the rational depth factor $\frac{1}{\sqrt{2L}}$.

---

## 5. Passing Gate Checklist
- [ ] Pilot 25M model completes $10^5$ training steps without numerical instability.
- [ ] Validation perplexity matches the standard Transformer within the target bound ($\leq 1.08\times$).
- [ ] VRAM reduction confirmed via CUDA memory tracker.
- [ ] Complete AST execution audit confirms 100% pure algebraic operations.
