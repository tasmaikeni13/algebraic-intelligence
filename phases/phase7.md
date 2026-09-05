# Phase 7: Architectural Integration & Pilot Pretraining (10M–30M LM on MI300X)

## 1. Objective & Research Scope
Assemble all 12 modules of the Algebraic Stack into a unified, end-to-end autoregressive language model architecture: `AlgebraicTransformerLM`.
- Configure and verify the **ROCm / PyTorch environment** on this server's **1x AMD Instinct MI300X (192 GB)**.
- Conduct pilot pretraining runs at scale (10M–30M parameters) across $10^5$ optimization steps on the MI300X.
- Demonstrate unbroken training stability with **zero NaNs, zero Infs, and zero loss spikes** under pure algebraic optimization (ACO + ARDS + OACE).
- Benchmark throughput and establish empirical scaling curves against a matching standard Transformer on the same accelerator.

---

## 2. Model Architecture & Hyperparameter Configuration

### 2.1 Hardware Configuration on MI300X
- **GPU:** 1x AMD Instinct MI300X (192 GB HBM3, `gfx942`).
- **Precision:** Pure algebraic forward in BF16/FP32, denominator scratchpad in FP32.
- **Batching:** Exploiting the massive 192 GB HBM3 capacity to run sequence length 2048 with micro-batch size 32–64 without CPU offloading or pipeline parallelization overhead.

### 2.2 Model Specifications (25M Parameters)
| Hyperparameter | Algebraic Model | Transcendental Baseline |
| :--- | :--- | :--- |
| **Layers** | 8 | 8 |
| **Hidden Dimension ($d$)** | 512 | 512 |
| **Attention Heads** | 8 | 8 |
| **Head Dimension ($d_k$)** | 64 | 64 |
| **FFN Intermediate Dimension** | 1365 (ALU-GLU) | 1365 (SwiGLU) |
| **Context Length** | 2048 | 2048 |
| **Activation** | ALU (rational gate) | GELU (erf/exponential) |
| **Attention Kernel** | A-Softmax ($\kappa_8$, $\Omega = 1.0$) | Exponential Softmax |
| **Positional Encoding** | AGO (Cayley $\mathrm{SO}(2)$) | RoPE ($\sin/\cos$) |
| **Loss Function** | OACE ($\mathcal{L}_{1/8}$) | Cross-Entropy ($-\sum y \ln p$) |
| **Optimizer** | ACO (Factorized $\mathcal{O}(d_{\text{out}} + d_{\text{in}})$) | AdamW ($\beta_1=0.9, \beta_2=0.95$) |
| **Learning Rate Schedule** | ARDS (Rational $\operatorname{rsqrt}$) | Cosine Annealing |

---

## 3. Empirical Passing Gate on MI300X

The agent must execute `analysis/pilot_pretraining.py` on this MI300X server and satisfy:

| Metric | Target Value on MI300X | Tolerance / Bound |
| :--- | :--- | :--- |
| **Optimization Stability** | Loss spikes ($\Delta \mathcal{L} > 2.0$) over $10^5$ steps | Exactly $0$ spikes |
| **Gradient Norm Stability** | $\max_t \|\mathbf{G}_t\|_2$ | $\leq 5.0$ without explicit gradient clipping |
| **Perplexity Parity vs. Baseline** | $\frac{\operatorname{PPL}_{\text{algebraic}}}{\operatorname{PPL}_{\text{baseline}}}$ at step $50,000$ | $\leq 1.08$ (within 8% parity) |
| **MI300X Token Throughput** | Tokens / second (BF16, sequence 2048) | Within $10\%$ of standard Transformer |
| **VRAM Consumption** | Model + Optimizer State at $d=512$ | $\geq 25\%$ reduction vs. AdamW baseline |
| **Zero Transcendental Audit** | Automated AST inspection of execution trace | Exactly $0$ transcendental operations |

---

## 4. Failure Modes & Self-Correction Playbook

- **Symptom: PyTorch ROCm device allocation stall:**
  *Root Cause:* Memory fragmentation or HIP driver IPC initialization issue.
  *Correction:* Configure environment variable: `export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` and verify device visibility with `torch.cuda.get_device_name(0)` returning MI300X.
- **Symptom: Loss stalls early at high perplexity ($> 50$):**
  *Root Cause:* Attention sink constant $\Omega$ too large relative to score sum $\sum \kappa_8(s)$.
  *Correction:* Calibrate $\Omega \in [0.1, 1.0]$ or dynamically scale $\Omega$ inversely with sequence length $L$.

---

## 5. Passing Gate Checklist
- [ ] PyTorch detects the 1x AMD Instinct MI300X (192 GB) GPU.
- [ ] Pilot 25M model completes $10^5$ training steps without numerical instability.
- [ ] Validation perplexity matches the standard Transformer within the target bound ($\leq 1.08\times$).
- [ ] Throughput and VRAM logged directly on MI300X hardware.
