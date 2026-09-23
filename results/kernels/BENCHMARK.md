# Historical Kernel Microbenchmark (Superseded)

These measurements predate the production-kernel wiring corrections. They are retained for provenance and must not be cited as current architecture or pretraining evidence. The recurrence below is a Taylor/diagonal-feature approximation, and the former pretraining telemetry was hard-coded rather than measured by this benchmark.

---

## 1. Complete $2 \times 2$ Attention Matrix

| Attention Regime | Context Length ($T$) | Standard Baseline | Pure Algebraic Transformer | Speedup | Dominant Physical Mechanism |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Unfused JIT** | $T = 2048$ | 145.03 ms | **95.52 ms** | **1.52$\times$** | Zero transcendental $\exp$ calls in silicon |
| **Unfused JIT** | $T = 8192$ | 2108.74 ms | **1637.5 ms** | **1.29$\times$** | Polynomial evaluation avoids SFU cycle penalty |
| **Tiled Micro-Kernel** | $T = 2048$ | 348.37 ms (standard online-softmax) | **367.78 ms (Octic AFA)** | **0.95$\times$** | Zero online exponential rescaling barriers |
| **Tiled Micro-Kernel** | $T = 8192$ | 5631.04 ms (standard online-softmax) | **5605.67 ms (Octic AFA)** | **1.0$\times$** | Pure additive accumulator updates in SRAM/VMEM |

---

## 2. Projection Head & Loss Function Head-to-Head

Both architectures are evaluated with the repository's vocabulary-chunked loss implementations without materializing the full logit tensor in High Bandwidth Memory:

| Metric | Standard Fused Cross-Entropy (Liger / Megatron) | Fused Linear + OACE (Algebraic Stack) | Comparison / Architectural Tradeoff |
| :--- | :--- | :--- | :--- |
| **Full-Batch Logit Materialization** | $105.4\text{ GB}$ (Unfused) $\to$ **12.8 MB** (Fused) | $105.4\text{ GB}$ (Unfused) $\to$ **12.8 MB** (Fused) | **$6500\times$ Memory Reduction (Equal Parity)** |
| **Micro-Batch Execution Latency** | 87.06 ms | **120.73 ms** | **0.72$\times$ (Relative)** |
| **Throughput (Tokens / Sec)** | 5881.0 tok/s | **4240.9 tok/s** | **0.72$\times$ Throughput Ratio** |
| **Transcendental Instructions** | Materializes $\ln(\sum e^z)$ across vocabulary | **Zero $\ln$, Zero $\exp$** (3-rsqrt cascade) | Eliminates SFU stall cycles |

---

## 3. Inference Scaling: Recurrent State Space Duality ($O(1)$ Memory)

The measured recurrence uses a degree-8 Taylor truncation with coordinate-wise features. Its state size is constant in context length, but it is not equivalent to exact octic AFA:

| Context Length ($T$) | Step Latency | Working Memory per Step | Complexity |
| :--- | :--- | :--- | :--- |
| **1,024** | 47.16 $\mu$s | **66820 bytes** | $O(1)$ memory & $O(1)$ compute per token |
| **2,048** | 36.51 $\mu$s | **66820 bytes** | $O(1)$ memory & $O(1)$ compute per token |
| **4,096** | 40.1 $\mu$s | **66820 bytes** | $O(1)$ memory & $O(1)$ compute per token |
| **8,192** | 41.69 $\mu$s | **66820 bytes** | $O(1)$ memory & $O(1)$ compute per token |
