# Phase 5 — MI300X/ROCm Systems Gate & Hardware-Fused Kernel Optimization

Start only after Phase 4 PASS. Read all prior artifacts, current official AMD ROCm / MI300X and PyTorch HIP documentation, and `phases/AUTONOMY_PROTOCOL.md`. Execute the failure-repair loop until PASS.

This phase optimizes the verified algebraic mechanisms on the dedicated single-node accelerator: **1x AMD Instinct MI300X GPU (192 GB HBM3, CDNA3 architecture, `gfx942`)**. A kernel that is faster but numerically divergent from the fp64 specification is a failure.

---

## 1. Execution Paths to Evaluate on MI300X

Compare, on identical quantized inputs and random seeds:
1. **fp64 CPU Reference Path:** Authoritative double-precision ground truth.
2. **PyTorch Eager ROCm Path:** Clean baseline in `torch.bfloat16` and `torch.float32`.
3. **`torch.compile` / Inductor (ROCm backend):** Native Inductor lowering with `mode="max-autotune"`.
4. **Native CDNA3 Wave64 HIP C++ Extension:** High-performance fused C++ kernel compiled via `hipcc --offload-arch=gfx942 -O3 -ffast-math` (`analysis/kernels/algebraic_attention_hip.cpp`).
5. **AMD Triton GPU Kernel:** Fused kernel using AMD Triton (`triton-amdgpu`) (`analysis/kernels/algebraic_attention_triton.py`).
6. **ROCm FlashAttention-2 Comparator:** Official ROCm FlashAttention-2 implementation.

---

## 2. Kernel Optimization & CDNA3 Architectural Targets

Exploit the unique architectural features of the AMD Instinct MI300X:
- **Wave64 Execution:** Design kernels for native CDNA3 Wave64 execution (`warp_size = 64`), matching matrix core dimensions.
- **Register Budget & Spilling:** Restrict tile dimensions so vector general-purpose register (VGPR) allocation stays $\le 256$ registers per work-item, eliminating scratch memory spills.
- **Local Data Share (LDS):** Optimize tile dimensions $(B_q, B_k) \in \{(64, 64), (128, 64)\}$ to fit within LDS without bank conflicts.
- **Single-Pass Additive Tile Accumulation:** Eliminate running-maximum subtraction $\exp(m_{\text{old}} - m_{\text{new}})$. Accumulate numerators $\mathbf{N} = \sum \kappa_8(\mathbf{S}) \mathbf{V}$ and denominators $D = \Omega + \sum \kappa_8(\mathbf{S})$ additively in LDS and FP32 scratch registers, normalizing in a single pass: $\mathbf{O} = \mathbf{N} / D$.
- **Asynchronous Ring Attention Simulation:** Simulate an 8-node ring where each node accumulates partial $(\mathbf{N}^{(p)}, D^{(p)})$ independently. Verify that sequence-wide attention is resolved via a single global $\mathrm{AllReduce}$ sum with zero inter-tile synchronization barriers.

---

## 3. Profiling & Systems Benchmarking Protocol

Benchmark across matrix dimensions:
- Sequence lengths $L \in \{512, 1024, 2048, 4096, 8192, 16384\}$;
- Head dimensions $d_k \in \{64, 128\}$ and heads $H \in \{12, 16\}$;
- Batch sizes $B \in \{1, 2, 4, 8, 16\}$;
- Target precision: BF16 forward/backward, FP32 accumulator.

### Required Telemetry:
1. **Wall-clock latency:** Synchronized with `torch.cuda.synchronize()`, reporting p50, p95, p99 across 100 steady-state iterations after 20 warmup iterations.
2. **Throughput:** Effective TFLOPS and tokens/second.
3. **Memory Bandwidth Utilization:** Sustained HBM3 bandwidth (GB/s) vs theoretical peak ($5.3\text{ TB/s}$).
4. **VRAM Footprint:** Static and peak dynamically allocated HBM bytes.
5. **Numerical Error:** Max absolute error and relative Frobenius error against the fp64 oracle.
6. **Zero-Transcendental ISA Audit:** Disassemble compiled kernel `.hsaco` via `llvm-objdump -d` to confirm zero transcendental math opcodes.

---

## 4. Systems Baselines & Competitors

Compare directly against:
- Standard ROCm FlashAttention-2;
- Standard Sliding-Window Attention;
- Standard PyTorch SDPA (Scaled Dot-Product Attention) on ROCm.

Compare across identical parameter counts, context lengths, and live decode memory states.

---

## PASS Gates

- [ ] Custom CDNA3 Wave64 HIP C++ kernel and AMD Triton kernel compile cleanly with zero errors under ROCm 6.2+ targeting `gfx942`.
- [ ] Accelerated kernels match fp64 CPU reference within condition-aware BF16 tolerance: relative error $\le 1.0 \times 10^{-5}$ in float32 and $\le 1.0 \times 10^{-3}$ in bfloat16.
- [ ] Zero running-maximum subtraction $\exp(m_{\text{old}} - m_{\text{new}})$ executed in AFA: ISA disassembly of compiled kernel binary confirms exactly 0 transcendental instructions.
- [ ] AFA kernel throughput on 1x MI300X reaches $\ge 85\%$ of ROCm FlashAttention-2 at $L=4096$, and matches or exceeds it at $L \ge 8192$.
- [ ] Sustained HBM3 memory bandwidth during tile streaming exceeds $3.5\text{ TB/s}$ ($> 65\%$ of theoretical peak).
- [ ] Simulated 8-node Ring Attention confirms single-pass AllReduce additive accumulation with relative error $< 1.0 \times 10^{-6}$ vs un-tiled attention.
- [ ] Zero GPU memory leaks, segmentation faults, or non-finite outputs across all benchmark configurations.
- [ ] All Lean 4 formal proofs compile cleanly via `/root/.elan/bin/lake build`.
- [ ] All inherited Phase 0–4 gates pass.
- [ ] `results/phase5/PASS.md` satisfies the shared PASS record contract.
