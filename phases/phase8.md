# Phase 8: Frontier Pretraining: 125M Parameters on 1B Tokens of FineWeb-Edu (3 Seeds on 1x MI300X)

## 1. Objective, Scientific Hypothesis & Competing Models
Execute the first definitive empirical evaluation of the core thesis:
$$\textbf{"Can algebra and algebra alone give rise to intelligence at the 125M scale?"}$$

Train a **125M-parameter Pure Algebraic Transformer** alongside a **standard 125M Transformer baseline** on **1 Billion tokens of FineWeb-Edu** across **three independent random seeds** (Seeds 42, 1337, 2026) strictly utilizing this server's **1x AMD Instinct MI300X (192 GB HBM3)**:
- Total runs: $2 \text{ architectures} \times 3 \text{ seeds} = 6 \text{ complete pretraining runs}$.
- Compute mean $\pm$ standard error of the mean (SEM) for all metrics.
- Establish whether pure algebra matches standard transcendental Transformers in convergence speed, final perplexity, and downstream reasoning.

---

## 2. Experimental Setup on 1x AMD Instinct MI300X

### 2.1 Hardware Allocation & Execution Model
- **Accelerator:** 1x AMD Instinct MI300X GPU (`gfx942`, 192 GB HBM3, 5.3 TB/s bandwidth).
- **VRAM Budget:** Total static state $< 800\text{ MB}$, leaving over $190\text{ GB}$ of local HBM3 for large micro-batches and activations.
- **Batch Pipeline:** Global batch size $\approx 1.05\times 10^6$ tokens (512 sequences of length 2048) with native CDNA3 Wave64 kernels.
- **Seeds:** Seed 42, Seed 1337, Seed 2026 for both models.

### 2.2 Model Specifications (125M Parameters)
| Hyperparameter | Algebraic Transformer (Ours) | Standard Transformer Baseline |
| :--- | :--- | :--- |
| **Layers** | 12 | 12 |
| **Hidden Dimension ($d$)** | 768 | 768 |
| **Attention Heads** | 12 | 12 |
| **Head Dimension ($d_k$)** | 64 | 64 |
| **FFN Dimension** | 2048 (ALU-GLU) | 2048 (SwiGLU) |
| **Sequence Length** | 2048 | 2048 |
| **Activation** | ALU ($x \cdot \beta(x)$) | GELU ($x \cdot \Phi(x)$) |
| **Attention** | A-Softmax ($\kappa_8(x)$, $\Omega=1.0$) | Exponential Softmax ($\exp(x)$) |
| **Positional Representation** | AGO (Rational Cayley $\mathrm{SO}(2)$) | RoPE ($\cos, \sin$) |
| **Normalization** | AVN | RMSNorm |
| **Loss Functional** | OACE ($\mathcal{L}_{1/8}$) | Cross-Entropy ($-\sum y \ln p$) |
| **Optimizer** | ACO (Factorized) | AdamW ($\beta_1=0.9, \beta_2=0.95$) |
| **Schedule** | ARDS (Rational $\operatorname{rsqrt}$) | Cosine Annealing |

---

## 3. Empirical Passing Gate & Acceptance Criteria (3-Seed Mean)

The runs across the 3 seeds must meet:

| Evaluation Dimension | Metric (Mean over 3 Seeds) | Success Threshold vs. Transformer Baseline |
| :--- | :--- | :--- |
| **Validation Perplexity** | Perplexity on FineWeb-Edu test set | $\frac{\operatorname{Mean\ PPL}_{\text{alg}}}{\operatorname{Mean\ PPL}_{\text{base}}} \leq 1.08$ (within $8\%$ parity) |
| **Perplexity Variance** | Standard Error of Mean (SEM) | $\operatorname{SEM} \leq 0.15$ across seeds |
| **Training Loss Stability** | Loss spikes / NaNs / Infs across all 6 runs | Exactly $0$ |
| **Downstream Zero-Shot** | ARC-Easy, HellaSwag, PIQA, LAMBADA | Mean accuracy within $2.0\%$ absolute of baseline |
| **Quantization Robustness** | FP4 / INT8 post-training degradation | $\leq 0.5$ PPL degradation (vs. $\geq 3.0$ PPL for baseline) |
| **Zero Transcendental Audit** | AST & memory check on all checkpoints | Exactly $0$ transcendental operations |

---

## 4. Autonomous Failure & Self-Correction Protocol

### If Phase 8 Passing Gate Fails:
The autonomous agent **MUST NOT PROCEED TO PHASE 9**. It must trigger the **Phase 8 Self-Correction Loop**:
1. **Diagnose Failure Mechanism:**
   - *Case 1: Early Optimization Instability / Divergence in Seed runs:* Recalibrate rational decay parameter $\alpha$ and add rational warmup.
   - *Case 2: Consistent Perplexity Gap ($> 8\%$ behind baseline):* Sweep $\Omega \in [0.2, 1.0]$ and temperature scalar $\tau = \frac{1}{\sqrt{d_k}}$ in Phase 7 pilot, then re-execute Phase 8.
   - *Case 3: Variance Accumulation Across 12 Layers:* Enforce AVN on residual branches or apply rational depth factor $\frac{1}{\sqrt{2D}}$.
2. **Formal Verification & Regression Testing:** If any mathematical primitive is altered, update `theory.md`, re-prove in Lean 4, run `lake build`, and verify that Phase 1–7 unit tests pass.
3. **Re-run Phase 8:** Rerun all 3 seeds for both models. Advance to Phase 9 **only** when all Section 3 criteria pass.

---

## 5. Passing Gate Checklist
- [ ] 3 random seed pretraining runs completed for 125M Algebraic model on 1B tokens.
- [ ] 3 random seed pretraining runs completed for 125M Baseline Transformer on 1B tokens.
- [ ] Statistical significance table (mean $\pm$ SEM) logged for perplexity and downstream benchmarks.
- [ ] Parity threshold $\leq 1.08\times$ validated.
- [ ] Strict zero-transcendental compliance confirmed on all checkpoints.
