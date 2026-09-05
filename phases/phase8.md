# Phase 8: Frontier Pretraining: 125M Parameters on 1B Tokens of FineWeb-Edu on 1x MI300X

## 1. Objective & Research Scope
Execute the definitive empirical experiment answering the foundational question:
$$\textbf{"Can algebra and algebra alone give rise to intelligence?"}$$

Train a **125M-parameter Pure Algebraic Transformer** alongside a **standard 125M Transformer baseline** on **1 Billion tokens of FineWeb-Edu** strictly utilizing this server's **1x AMD Instinct MI300X (192 GB HBM3)**:
- Leverage the massive 192 GB single-socket HBM3 memory to train with high batch sizes and context length 2048 without multi-node communication bottlenecks.
- Compare training dynamics, final validation perplexity, downstream zero-shot reasoning, and hardware throughput under strictly controlled, identical FLOP budgets and identical hardware.

---

## 2. Experimental Setup on 1x AMD Instinct MI300X

### 2.1 Hardware Utilization Strategy
- **Accelerator:** 1x AMD Instinct MI300X GPU (`gfx942`, 192 GB HBM3, 5.3 TB/s bandwidth).
- **Zero Inter-GPU Communication:** Running on a single 192 GB socket eliminates all NCCL/RCCL communication overhead, allowing pure compute and memory bandwidth measurement.
- **Batch Pipeline:** Large global batch size enabled by 192 GB VRAM (e.g. 512 sequences of length 2048 = ~1.05M tokens per batch).
- **Kernels:** Optimized HIP C++ and AMD Triton kernels for both architectures:
  - Algebraic Model: Fused AFA, octic $\kappa_8$ in LDS, ALU-GLU.
  - Baseline Model: FlashAttention-2 ROCm/HIP, SwiGLU.

### 2.2 Model Specifications (125M Parameters)
| Hyperparameter | Algebraic Transformer (Ours) | Standard Transformer Baseline |
| :--- | :--- | :--- |
| **Layers** | 12 | 12 |
| **Hidden Dimension ($d$)** | 768 | 768 |
| **Attention Heads** | 12 | 12 |
| **Head Dimension ($d_k$)** | 64 | 64 |
| **FFN Intermediate Dimension** | 2048 | 2048 |
| **Sequence Length** | 2048 | 2048 |
| **Activation** | ALU ($x \cdot \beta(x)$) | GELU ($x \cdot \Phi(x)$) |
| **Attention Mechanism** | A-Softmax ($\kappa_8(x)$, $\Omega=1.0$) | Exponential Softmax ($\exp(x)$) |
| **Positional Representation** | AGO (Rational Cayley $\mathrm{SO}(2)$) | RoPE ($\cos m\theta, \sin m\theta$) |
| **Normalization** | AVN ($\mathbf{x} \cdot \operatorname{rsqrt}(m_2 + \epsilon)$) | RMSNorm |
| **Loss Functional** | OACE ($\mathcal{L}_{1/8}$) | Cross-Entropy ($-\sum y \ln p$) |
| **Optimizer** | ACO (Factorized $\mathcal{O}(d_{\text{out}} + d_{\text{in}})$) | AdamW ($\beta_1=0.9, \beta_2=0.95, \lambda=0.1$) |
| **Learning Rate Schedule** | ARDS ($\eta_0 \cdot \operatorname{rsqrt}(1 + \alpha t^2)$) | Cosine Annealing |

---

## 3. Empirical Passing Gate & Acceptance Criteria on MI300X

The pretraining run on this MI300X server must satisfy:

| Evaluation Dimension | Metric | Success Threshold vs. Transformer Baseline |
| :--- | :--- | :--- |
| **Validation Perplexity** | Perplexity on FineWeb-Edu test set | At par or within $5\%$ to $8\%$ of baseline ($\frac{\operatorname{PPL}_{\text{alg}}}{\operatorname{PPL}_{\text{base}}} \leq 1.08$) |
| **Training Loss Stability** | NaN / Inf occurrences over 1B tokens | Exactly $0$ |
| **Downstream Zero-Shot** | ARC-Easy / HellaSwag / PIQA / LAMBADA | Within $2.0\%$ absolute accuracy of baseline |
| **MI300X Sustained Throughput** | Tokens / second on 1x MI300X | Competitive with baseline Transformer (within 10%) |
| **Optimizer Memory (HBM)** | Optimizer state allocated in 192 GB VRAM | $\geq 50\%$ lower memory footprint than AdamW |
| **Quantization Robustness** | FP4 / INT8 post-training degradation | $\leq 0.5$ PPL degradation (vs. $\geq 3.0$ PPL for baseline) |
| **Zero Transcendental Audit** | Full execution trace inspection | Exactly $0$ transcendental operations |

---

## 4. Failure Modes & Self-Correction Playbook

- **Symptom: Validation loss lags baseline by $> 10\%$ in first 200M tokens:**
  *Root Cause:* Initial learning rate in ARDS too conservative compared to AdamW's warmup.
  *Correction:* Implement a rational warmup schedule: $\eta_t = \eta_0 \cdot \frac{t}{T_{\text{warm}}}$ for $t \leq T_{\text{warm}}$, transitioning smoothly into rational decay:
  $$\eta_t = \eta_0 \cdot \operatorname{rsqrt}\left(1 + \alpha (t - T_{\text{warm}})^2\right)$$
- **Symptom: Memory bandwidth throttling on MI300X:**
  *Root Cause:* Sub-optimal tile access patterns across memory channels.
  *Correction:* Align memory access strides to 128-byte boundaries and ensure LDS tiles are padded to avoid bank conflicts.

---

## 5. Passing Gate Checklist
- [ ] 125M Algebraic model successfully trained on 1B tokens of FineWeb-Edu on 1x MI300X.
- [ ] 125M Baseline Transformer successfully trained on identical 1B tokens on 1x MI300X.
- [ ] Loss curves and validation perplexities logged and plotted.
- [ ] Downstream zero-shot evaluation completed and tabulated.
- [ ] Strict zero-transcendental verification confirmed on all model checkpoints.
