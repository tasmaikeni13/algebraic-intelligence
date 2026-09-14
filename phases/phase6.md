# Phase 6: Hardware-Fused Kernels & Algebraic FlashAttention (AFA on 16 TPU v4 Pod via Pallas / XLA HLO)

Start only after Phase 5 PASS. Read `theory.md`, official Google Cloud TPU v4, JAX Pallas, and XLA HLO compiler documentation, and `phases/README.md`. Execute the adaptive failure-repair loop until PASS.

---

## 1. Objective, Scientific Hypothesis & Competing Models

Eliminate inter-tile synchronization barriers and transcendental online rescaling of FlashAttention on TPU v4 systolic hardware:
$$\textbf{"Can Algebraic FlashAttention achieve near-roofline memory bandwidth on Google Cloud TPU v4 hardware via JAX Pallas and XLA HLO?"}$$

### Competing Hypotheses:
- **$H_1$ (Algebraic Hypothesis):** AFA replaces running maximum subtraction $\exp(m_{\text{old}} - m_{\text{new}})$ with pure additive tile accumulation. When implemented in JAX Pallas targeting TPU v4 TensorCore Vector Memory (VMEM) and 128×128 Matrix Multiply Units (MXUs), AFA streams tiles without inter-tile log-sum-exp synchronization barriers, sustaining $> 70\%$ peak HBM bandwidth ($> 840\text{ GB/s}$ per chip) and enabling lock-free distributed Ring Attention across the 16 TPU v4 chips over the 3D Torus Inter-Chip Interconnect (ICI).
- **$H_0$ (Transcendental Baseline Hypothesis):** Standard FlashAttention-2 online exponential rescaling is uniquely optimal for hardware scratchpad caching; additive algebraic kernels will encounter VMEM pressure or numerical overflow during long sequence tile streaming.

---

## 2. TPU v4 Hardware Architecture & XLA HLO Execution Model

To instruct the autonomous agent or engineer in building hardware-optimal kernels for the Google Cloud TPU v4, understand the exact mapping to TPU v4 TensorCore silicon:

```mermaid
graph TD
    subgraph "TPU v4 TensorCore Architecture"
        HBM["High-Bandwidth Memory (HBM2e)<br/>32 GB/chip @ 1.2 TB/s"]
        
        subgraph "On-Chip Scratchpad & Compute"
            VMEM["Vector Memory (VMEM)<br/>16 MB per TensorCore"]
            SMEM["Scalar Memory (SMEM)"]
            
            MXU["Matrix Multiply Units (MXU)<br/>Dual 128x128 Systolic Arrays (BF16)"]
            VMU["Vector Processing Unit (VMU)<br/>Elementwise +, -, *, /, rsqrt"]
        end
    end

    HBM -- "Async DMA Stream (BlockSpec Tiling)" --> VMEM
    VMEM -- "Systolic Matmul: Q @ K^T" --> MXU
    MXU -- "Raw Scores S_bc" --> VMU
    VMU -- "Octic Squaring (κ₈) & Additive Accumulation" --> VMEM
    VMEM -- "Systolic Matmul: P_bc @ V_c" --> MXU
    MXU -- "Output Accumulator O_b" --> VMEM
    VMEM -- "Final Division: O_b / D_b" --> VMU
    VMU -- "Writeout Y_b" --> HBM
```

### 2.1 Memory Spaces on TPU v4
1. **HBM2e (Off-Chip Memory):** 32 GB per TPU v4 chip, ~1.2 TB/s bandwidth. Global tensor storage for activations, model parameters, and optimizer states.
2. **VMEM (`pltpu.VMEM`):** Fast on-chip 16 MB Vector Memory per TensorCore. Pallas kernels load sub-blocks of $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ directly into VMEM.
3. **SMEM (`pltpu.SMEM`):** Scalar Memory used for loop induction variables, grid index arithmetic, and scalar constants.

### 2.2 Compute Engines on TPU v4
1. **MXU (Matrix Multiply Unit):** Dual 128×128 systolic matrix multiplier arrays per core. Peak efficiency requires tile dimensions to be exact multiples of 128 (e.g., $B_q = 128, B_k = 128$).
2. **VMU (Vector Processing Unit):** SIMD vector execution unit capable of elementwise arithmetic, vector loads/stores, and hardware-native $\operatorname{rsqrt}$. The 3-stage octic squaring $\kappa_8$ evaluates entirely in VMU vector registers.

---

## 3. Pallas TPU Kernel Implementation & XLA Lowering Instructions

Instruct the creation of `src/kernels/pallas_afa.py` following these concrete implementation steps:

### 3.1 Kernel Tiling & Pallas Signature
The kernel partitions the sequence length $L$ into blocks of size $B_q = 128$ and $B_k = 128$, matching the 128×128 MXU systolic dimension:

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

def afa_kernel(q_ref, k_ref, v_ref, o_ref, d_ref, *, sink_omega: float, scale: float):
    """
    Pure Additive Algebraic FlashAttention Tile Kernel for TPU v4.
    Executed directly inside TPU v4 TensorCore VMEM.
    """
    # q_ref: (B_q, d) in VMEM
    # k_ref: (B_k, d) in VMEM
    # v_ref: (B_k, d) in VMEM
    # o_ref: (B_q, d) in VMEM (output accumulator)
    # d_ref: (B_q,) in VMEM (denominator accumulator)
    
    # 1. Systolic Matrix Multiply on MXU: S_bc = (Q_b @ K_c^T) * scale
    s_bc = jnp.matmul(q_ref[...], k_ref[...].T) * scale
    
    # 2. Causal Masking (if applicable) using rational negative floor:
    # Under Zero-Transcendental Axiom, masked positions receive large negative value
    
    # 3. Three-Stage Squaring Kernel on VMU (Zero Transcendentals):
    # Base: rho = s + sqrt(1 + s^2)
    s_sq = s_bc * s_bc
    rho_1 = s_bc + jnp.sqrt(1.0 + s_sq)
    rho_2 = rho_1 * rho_1         # Degree 2
    rho_4 = rho_2 * rho_2         # Degree 4
    p_bc = rho_4 * rho_4          # Degree 8 (Octic A-Softmax)
    
    # 4. Pure Additive Tile Accumulation (Zero running-max subtraction!):
    # Accumulate into numerator matrix: O_b += P_bc @ V_c (MXU matmul)
    o_ref[...] += jnp.matmul(p_bc.astype(v_ref.dtype), v_ref[...])
    
    # Accumulate into denominator vector: D_b += sum_j P_bc[:, j] (VMU reduction)
    d_ref[...] += jnp.sum(p_bc, axis=-1)
```

### 3.2 Pallas Grid & Memory BlockSpecs
```python
def pallas_afa_forward(q, k, v, sink_omega=0.5):
    """
    Full forward call orchestrating Pallas TPU execution.
    q: (B, H, L, D), k: (B, H, L, D), v: (B, H, L, D) in HBM.
    """
    batch_size, num_heads, seq_len, head_dim = q.shape
    B_q, B_k = 128, 128
    scale = 1.0 / jnp.sqrt(head_dim)
    
    # Grid: (Batch * Heads, Num_Q_Blocks)
    grid = (batch_size * num_heads, seq_len // B_q)
    
    # Define VMEM BlockSpecs with index maps
    in_specs = [
        pl.BlockSpec((B_q, head_dim), lambda b_h, i, j: (b_h // num_heads, b_h % num_heads, i, 0)),
        pl.BlockSpec((B_k, head_dim), lambda b_h, i, j: (b_h // num_heads, b_h % num_heads, j, 0)),
        pl.BlockSpec((B_k, head_dim), lambda b_h, i, j: (b_h // num_heads, b_h % num_heads, j, 0)),
    ]
    out_specs = [
        pl.BlockSpec((B_q, head_dim), lambda b_h, i, j: (b_h // num_heads, b_h % num_heads, i, 0)),
        pl.BlockSpec((B_q,), lambda b_h, i, j: (b_h // num_heads, b_h % num_heads, i)),
    ]
    
    # Invoke Pallas TPU call
    out_o, out_d = pl.pallas_call(
        afa_kernel,
        out_shape=[
            jax.ShapeDtypeStruct(q.shape, q.dtype),
            jax.ShapeDtypeStruct((batch_size, num_heads, seq_len), jnp.float32),
        ],
        grid=grid,
        in_specs=in_specs,
        out_specs=out_specs,
        scratch_shapes=[pltpu.VMEM((B_q, B_k), jnp.float32)],
        compiler_params=pltpu.TPUCompilerParams(dimension_semantics=("parallel", "parallel")),
    )(q, k, v, sink_omega=sink_omega, scale=scale)
    
    # Final normalization in VMU: Y = O / (D + sink_omega)
    return out_o / (out_d[..., None] + sink_omega)
```

---

## 4. XLA HLO Lowering & Static Opcode Audit

Instruct the creation of `scripts/audit_xla_hlo.py` to inspect the compiled XLA High-Level Optimizer (HLO) representation:

### 4.1 How to Lower and Dump XLA HLO
```python
import jax
from src.kernels.pallas_afa import pallas_afa_forward

def inspect_hlo(q, k, v):
    # Lower to XLA HLO
    lowered = jax.jit(pallas_afa_forward).lower(q, k, v)
    hlo_module = lowered.compile().runtime_executable()
    hlo_text = lowered.as_text()
    return hlo_text
```

### 4.2 Strict XLA HLO Acceptance Criteria
1. **MXU Systolic Mapping:** Verify that matrix multiplies are lowered to `dot` or `custom-call(tpu_custom_call)` targeting the 128×128 systolic MXU.
2. **VMU Radical Instructions:** Verify that $\sqrt{1 + s^2}$ is lowered to `rsqrt` / `multiply` on the VMU without library call indirection.
3. **Zero-Transcendental HLO Instruction Check:**
   - Run regex search on the emitted HLO text:
     ```python
     forbidden_hlo_ops = ['exponential', 'logarithm', 'sine', 'cosine', 'tanh', 'sigmoid']
     for op in forbidden_hlo_ops:
         assert op not in hlo_text.lower(), f"Forbidden transcendental HLO op found: {op}"
     ```
   - Must return exactly **0 occurrences**.
4. **No Inter-Tile Rescaling Logic:** Verify that the HLO loop body does not contain subtraction of max values ($m_{\text{new}} - m_{\text{old}}$) or intermediate exponential rescaling multiplications.

---

## 5. Distributed Ring Attention across 16 TPU v4 Chips over ICI

Because AFA accumulator $\mathbf{O}_b = \sum_c \mathbf{P}_{bc} \mathbf{V}_c$ and $\mathbf{D}_b = \sum_c \sum_j \mathbf{P}_{bc, \cdot j}$ are strictly additive, Ring Attention across the 16 TPU v4 chips requires **zero inter-tile normalization barriers**:
1. Each TPU v4 chip processes its local query and key/value shard.
2. Sequence parallel chunks are rotated across the 16-chip 3D Torus ICI interconnect using `jax.lax.psum_scatter` and `jax.lax.all_gather`.
3. A single global `jax.lax.psum` over the ICI mesh sums the partial numerators $\mathbf{O}_b^{(p)}$ and partial denominators $\mathbf{D}_b^{(p)}$.
4. Final tile normalization evaluates in a single pass at the end of the ring traversal:
   $$\mathbf{Y}_b = \frac{\sum_{p=1}^{16} \mathbf{O}_b^{(p)}}{\Omega + \sum_{p=1}^{16} \mathbf{D}_b^{(p)}}$$

---

## 6. Lean 4 Formal Verification Gate

The agent must compile `formal/AlgebraicTheory/Kernel.lean` and `formal/AlgebraicTheory/Gate.lean` under `/root/.elan/bin/lake build`:
1. Single-pass additive associativity: $\sum (P_1 V_1 + P_2 V_2) = (\sum P_1 V_1) + (\sum P_2 V_2)$.
2. Numerator-denominator scaling invariance: $(\alpha O) / (\alpha D) = O / D$ for $\alpha > 0$.

---

## 7. Hardware Benchmarking & Passing Gate on 16 TPU v4 Pod

Benchmark `src/kernels/pallas_afa.py` via `scripts/run_benchmark_pallas.py` on the 16 TPU v4 Pod:

| Evaluation Dimension | Target on 16 TPU v4 Pod | Tolerance / Bound |
| :--- | :--- | :--- |
| **Numerical Accuracy vs. Float64** | $\|\mathbf{Y}_{\text{AFA}} - \mathbf{Y}_{\text{exact}}\|_\infty / \|\mathbf{Y}_{\text{exact}}\|_\infty$ | $\leq 1.0 \times 10^{-6}$ |
| **Inter-Tile Rescaling FLOPs** | Transcendental $\exp(m_{\text{old}} - m_{\text{new}})$ calls in AFA | Exactly $0$ |
| **Kernel Throughput at $L=4096$** | TFLOPS per chip (BF16 forward) | $\geq 85\%$ of baseline FlashAttention-2 |
| **HBM Memory Bandwidth Utilization** | Sustained GB/s during tile streaming | $\geq 70\%$ of theoretical peak ($> 840\text{ GB/s}$/chip) |
| **Distributed Ring Attention Relative Error** | 16 TPU v4 chips, additive accumulation error | $\leq 1.0 \times 10^{-6}$ |
| **XLA HLO Transcendental Audit** | Grep of compiled HLO instructions | Exactly $0$ transcendental opcodes |

---

## 8. Adaptive Failure-Repair & Bidirectional Dependency Protocol

When a test or gate fails in Phase 6:
1. **Iterate Locally:**
   - If VMEM allocation exceeds 16 MB scratchpad capacity, tune tile dimensions from $(128, 128)$ to $(128, 64)$ or adjust scratchpad memory hints.
   - If XLA compiler reports tiling mismatch, confirm that sequence length is padded to multiples of 128.
2. **Backward Rollback to Phase 2/1:**
   - If numerical instability occurs during long sequence streaming, inspect Phase 2 A-Softmax logit scaling $\tau$ or attention sink $\Omega$.
   - If Phase 2 A-Softmax or Phase 1 AVN must be modified, backtrack to Phase 2 (or Phase 1), update the core primitives, re-prove Lean 4 theorems, pass their regression gates, and forward-cascade the updates back to Phase 6.
3. **Forward Dependency Cascading:**
   - **Phases 7, 8, 9 (`src/model.py`, pretraining pipelines):** Integrate `pallas_afa` as the attention engine in `AlgebraicTransformerLM`. Any changes to tile signatures or sharding constraints must be propagated into `src/model.py` and `src/mesh.py`.

---

## 9. PASS Gates

- [ ] `formal/AlgebraicTheory/Kernel.lean` compiles with 0 errors via `/root/.elan/bin/lake build`.
- [ ] `src/kernels/pallas_afa.py` created with JAX Pallas TPU implementation for TPU v4 VMU/MXU.
- [ ] Relative numerical accuracy against float64 un-tiled reference is $\le 1.0 \times 10^{-6}$.
- [ ] Head-to-head throughput benchmark executed against FlashAttention-2 on 16 TPU v4 Pod.
- [ ] Sustained HBM bandwidth exceeds $70\%$ of theoretical peak ($> 840\text{ GB/s}$ per chip).
- [ ] Distributed Ring Attention executes across 16 TPU v4 chips over ICI with relative error $\le 1.0 \times 10^{-6}$.
- [ ] XLA HLO inspection dumps confirm zero transcendental library calls or opcodes.
- [ ] `results/phase6/PASS.md` satisfies the shared PASS record contract.
