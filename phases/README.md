# Autonomous Agent Research Protocol: The Algebraic Stack

This directory contains the operational execution blueprints, formal verification specifications, empirical benchmarking protocols, and release standards for the **Algebraic Stack** research project.

The ultimate scientific question investigated across this repository is singular, foundational, and uncompromising:
$$\textbf{Can algebra and algebra alone give rise to intelligence?}$$

We mandate the strict **Zero-Transcendental Axiom**:
$$\text{No } e^x, \quad \text{No } \ln(x), \quad \text{No } \sin(x), \quad \text{No } \cos(x), \quad \text{No continuous exponential EMAs}, \quad \text{No cosine schedules.}$$
Every layer, activation, normalization, attention score, positional rotation, loss functional, divergence, and optimizer update must consist solely of rational operations, polynomial compositions, and a single hardware-native algebraic radical: $\operatorname{rsqrt}(x) = 1/\sqrt{x}$.

---

## 1. The Autonomous Constitution

All automated research execution in this repository is governed by:
$$\textbf{\href{AUTONOMY_PROTOCOL.md}{AUTONOMY\_PROTOCOL.md}}$$

Every numbered phase inherits all earlier gates and must strictly adhere to:
- **The Evidence Hierarchy:** Raw execution traces, checkpoints, and hardware measurements are authoritative over prose and expectations.
- **The 9-Step Failure-Repair Loop:** Freeze evidence $\to$ Classify failure $\to$ Research primary literature $\to$ Derive mathematical repair $\to$ Formalize in Lean 4 $\to$ Implement twice (fp64 CPU oracle vs. optimized path) $\to$ Test mechanism $\to$ Run all inherited gates $\to$ Iterate.
- **Gate Amendment Discipline:** Thresholds are never lowered for convenience or disappointing results; gates may only be versioned if evidence refutes the theoretical claim or invalidates the evaluation harness.
- **The Reproducibility Contract:** Deterministic seed pinning, fp64 oracle standard, synchronized GPU timing, and equal-budget baseline comparisons.
- **Formal Verification Contract:** Zero-axiom Lean 4 compilation via Mathlib4 with zero `sorry` and zero `admit`.
- **PASS Record:** Each phase officially concludes only upon generating `results/phaseN/PASS.md`.

---

## 2. Hardware Substrate: 1x AMD Instinct MI300X (ROCm / HIP)

All empirical simulation, kernel execution, and pretraining phases run on this machine's dedicated hardware accelerator:
- **Accelerator:** 1x AMD Instinct MI300X GPU (CDNA3 architecture, `gfx942`).
- **Memory Capacity:** **192 GB unified HBM3** with **5.3 TB/s** peak memory bandwidth.
- **Software Stack:** ROCm toolchain (`HIP 6.2+`, `hipcc` at `/opt/rocm/bin/hipcc`, AMD Clang, AMD Triton).
- **Substrate Advantage:**
  - With **192 GB of high-bandwidth memory on a single socket**, 125M and 350M models train locally without distributed pipeline stalls or multi-node communication bottlenecks.
  - Custom CDNA3 Wave64 kernels eliminate inter-tile log-sum-exp synchronization barriers.

---

## 3. The Ten Research & Verification Phases

The autonomous research lifecycle is organized into **exactly ten sequential phases** (`phase0.md` through `phase9.md`):

| Phase File | Phase Title | Primary Focus & Verification Milestone |
| :--- | :--- | :--- |
| [**`phase0.md`**](phase0.md) | **Algebraic Reference Substrate & MI300X Audit** | Zero-transcendental AST audit, fp64 CPU reference paths, MI300X/HIP environment verification, and fail-fast test runner. |
| [**`phase1.md`**](phase1.md) | **Mathematical Oracle & Pathology Gate** | Falsification of exact algebraic equations under extreme conditioning ($\kappa = 10^6$), boundary logits ($p = 10^{-9}$), and Lean 4 formal verification. |
| [**`phase2.md`**](phase2.md) | **Primitive Mechanism Separation & Matched Baselines** | Controlled 4-view comparison (dimension, params, bytes, FLOPs) of ALU, A-Softmax, AGO, AVN, OACE, and ACO vs. GELU, Softmax, RoPE, RMSNorm, CE, and AdamW. |
| [**`phase3.md`**](phase3.md) | **Learned Representations & Sequence Induction** | 7 algorithmic task families across 5 paired seeds: associative memory (MQAR), induction heads, selective copy, and effective rank preservation. |
| [**`phase4.md`**](phase4.md) | **Long-Context Extrapolation & Quantization Gate** | $16\times$ horizon extrapolation ($L=16384$), multi-hop pointer chains (up to 16 hops), and sub-byte FP4/INT4 quantization stability ($\ge 100\times$ noise reduction). |
| [**`phase5.md`**](phase5.md) | **MI300X Systems Gate & Kernel Optimization** | Fused CDNA3 Wave64 HIP C++ and AMD Triton kernels for AFA, sustaining $> 3.5\text{ TB/s}$ HBM3 bandwidth and pure additive tile accumulation. |
| [**`phase6.md`**](phase6.md) | **Language-Model Viability & 3-Way Comparative Gate** | 3-way publication comparison at 125M and 350M scales: Algebraic Transformer vs. Causal Transformer vs. SSM-Attention Hybrid on FineWeb-Edu. |
| [**`phase7.md`**](phase7.md) | **Matched Multi-Seed Pretraining Study (125M / 1.0B Tokens)** | Head-to-head pretraining across Seeds 42, 1337, and 2026 on 1B FineWeb-Edu tokens on 1x MI300X; statistical significance (mean $\pm$ SEM) across downstream probes. |
| [**`phase8.md`**](phase8.md) | **Scaled Pretraining Study (350M / 3.0B Tokens)** | Scaling to 24 layers, width 1024, and 3B tokens; neural scaling law power-law validation and Hierarchical Scaling Back-Propagation Loop. |
| [**`phase9.md`**](phase9.md) | **Clean-Room Reproduction & Release Audit** | Clean-room fresh-clone replication on MI300X, standalone paper finalization (`theory.md`), full completion matrix, and open-source release package. |

---

## 4. Dual-Pillar Verification Paradigm

```mermaid
graph TD
    A["Phase Specification (phaseX.md)"] --> B["Pillar 1: Lean 4 Formal Logic"]
    B --> C{"lake build (0 Sorry, 0 Admit)?"}
    C -- "Failed" --> D["Analyze Proof Tactic / Reformulate Lemma"]
    D --> B
    C -- "Passed" --> E["Pillar 2: Deep Empirical & Systems Benchmarking"]
    E --> F{"Empirical Gates (fp64 Parity, Convergence, Noise)?"}
    F -- "Failed" --> G["Trigger 9-Step Failure-Repair Loop"]
    G --> B
    F -- "Passed" --> H["Hardware Execution on 1x AMD Instinct MI300X"]
    H --> I{"Throughput, Memory & Parity Validated?"}
    I -- "Failed" --> J["Execute Systems / Scaling Remediation"]
    J --> E
    I -- "Passed" --> K["Generate results/phaseX/PASS.md & Advance"]
```

Every phase enforces the **Dual-Pillar Verification Contract**:
1. **Pillar 1: Machine-Checked Formal Verification:** Lean 4 modules in `formal/AlgebraicTheory/` compiled cleanly with Mathlib4 via `lake build`.
2. **Pillar 2: Rigorous Empirical & Hardware Testing:** High-sample Monte Carlo simulations, deep gradient flow tracking, sub-byte quantization noise stress, and full pretraining on 1x AMD Instinct MI300X.

---

## 5. Execution Commands

```bash
# Run Phase 0 verification (Environment audit, AST check, fp64 tests, Lean build):
python3 analysis/verify_algebraic_primitives.py

# Run comparative empirical benchmarks:
python3 analysis/benchmark_algebraic_vs_transcendental.py

# Compile formal Lean 4 proofs:
cd formal && /root/.elan/bin/lake build
```
