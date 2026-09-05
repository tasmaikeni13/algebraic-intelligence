# Phase 0 — Algebraic Reference Substrate, Independent Oracles, & ROCm/MI300X Audit

Read `theory.md`, `formal/README.md`, every file in `formal/AlgebraicTheory/`, and `phases/AUTONOMY_PROTOCOL.md` completely before acting. Execute the shared failure-repair loop until this phase passes.

This phase establishes the verified numerical, formal, and hardware substrate for all subsequent phases. Do not train a language model or claim large-scale scaling superiority yet.

---

## 0.1 Zero-Transcendental Static & Runtime Audit

Enforce the Zero-Transcendental Axiom across all tracked files in the repository:
1. Implement a comprehensive Python AST (Abstract Syntax Tree) static analyzer (`scripts/audit_zero_transcendentals.py`) that scans all Python modules for forbidden transcendental primitives:
   - Built-in math functions: `math.exp`, `math.log`, `math.sin`, `math.cos`, `math.tan`, `math.sinh`, `math.cosh`, `math.tanh`;
   - PyTorch operators: `torch.exp`, `torch.log`, `torch.sin`, `torch.cos`, `torch.tan`, `torch.sigmoid`, `torch.softmax`, `F.softmax`, `F.sigmoid`, `F.gelu`, `F.silu`;
   - NumPy operators: `np.exp`, `np.log`, `np.sin`, `np.cos`.
2. Ensure case-insensitive `rg` across `analysis/`, `benchmarks/`, and `src/` detects zero un-whitelisted transcendental invocations outside of baseline comparison modules explicitly named `baseline_*` or `benchmark_*`.
3. Delete or archive stale experimental runs whose equations or logs do not match the current Algebraic Stack formulation in `theory.md`. Do not cosmetically relabel old data.

---

## 0.2 Independent fp64 CPU Reference Paths & Oracles

Construct a self-contained, transparent fp64 CPU reference implementation with immutable state for all foundational primitives:
1. **ALU Reference:**
   - Forward: $K(x) = \frac{x}{2}(1 + u)$ with $u = x / \sqrt{x^2 + 1}$;
   - Backward: Exact Horner cubic polynomial $K'(x) = 0.5 + u \cdot (1.0 - 0.5 \cdot u^2)$;
   - Numerical Autograd Oracle: Verified against central finite differences in fp64 with step $h = 10^{-7}$.
2. **AVN Reference:**
   - Forward: $\hat{\mathbf{x}} = \mathbf{x} / \sqrt{\frac{1}{d}\|\mathbf{x}\|^2 + \epsilon}$;
   - Backward: Analytic projection $\tau (\mathbf{g} - \frac{\langle \mathbf{g}, \hat{\mathbf{x}} \rangle}{d}\hat{\mathbf{x}})$;
   - Coupling Identity verification: $\beta(x; v) = \beta(\hat{x}; 1)$.
3. **A-Softmax Reference:**
   - Kernel $\rho(x) = x + \sqrt{x^2 + 1}$;
   - Power of 8 via 3 squarings: $\rho^2 = \rho \cdot \rho$, $\rho^4 = \rho^2 \cdot \rho^2$, $\rho^8 = \rho^4 \cdot \rho^4$;
   - Closed-form Jacobian: $\frac{\partial p_i}{\partial \hat{s}_j} = 8 w_j p_i (\delta_{ij} - p_j)$ where $w_j = (1 + \hat{s}_j^2)^{-1/2}$;
   - Attention sink $\Omega \ge 0$ handling.
4. **OACE ($\mathcal{L}_{1/8}$) & Algebraic Divergence ($D_A$):**
   - 3-rsqrt loss evaluation: $z_1 = p_k^{-1/2}$, $z_2 = p_k^{-1/4}$, $z_3 = p_k^{-1/8}$, $\mathcal{L}_{1/8} = 8(z_3 - 1)$;
   - Analytic gradient: $-8 w_j p_k^{-1/8}(\delta_{kj} - p_j)$;
   - Pearson divergence: $D_A(\mathbf{y} \| \mathbf{p}) = \sum y_i^2 / p_i - 1$.
5. **AGO Cayley Rotation:**
   - Static rational rotation matrix $\mathbf{R}_k = \frac{1}{1 + \omega_k^2} \begin{pmatrix} 1 - \omega_k^2 & -2\omega_k \\ 2\omega_k & 1 - \omega_k^2 \end{pmatrix}$;
   - Autoregressive update: $\mathbf{R}_k(m) = \mathbf{R}_k \mathbf{R}_k(m-1)$ with periodic algebraic re-normalization;
   - Relative shift equivariance: $\langle \mathbf{Q}_m, \mathbf{K}_n \rangle = \mathbf{x}_q^\top \mathbf{R}_k^{n-m} \mathbf{x}_k$.
6. **AFA Additive Accumulation:**
   - Full un-tiled reference vs tiled additive accumulator $\mathbf{N} = \sum \mathbf{N}^{(t)}, D = \sum D^{(t)}, \mathbf{O} = \mathbf{N} / D$.
7. **ACO Factorized Curvature:**
   - Factorized row-column accumulators $\mathbf{r}_t \in \mathbb{R}^{d_{\text{out}}}, \mathbf{c}_t \in \mathbb{R}^{d_{\text{in}}}$;
   - Rank-1 synthesis $\sqrt{\hat{r}_i \hat{c}_j}$ evaluated via single $\operatorname{rsqrt}$;
   - Rational momentum and polynomial debiasing $\delta_1(t) = 1 - \beta_1^t, \delta_2(t) = 1 - \beta_2^t$;
   - ARDS rational decay schedule $\eta_t = \eta_{\max} \operatorname{rsqrt}(1 + \alpha ((t - T_{\text{warm}})/T_{\text{decay}})^2)$.

The streaming/tiled and numerical oracle paths must not share state-update logic. Double-precision fp64 on CPU is the sole source of numerical truth.

---

## 0.3 ROCm / MI300X Hardware Substrate Audit

The designated server target is one AMD Instinct MI300X GPU (192 GB HBM3, CDNA3 architecture, `gfx942`). Inspect and log the full hardware and software environment before running benchmarks:
1. **Hardware Metadata:**
   - GPU device name (`AMD Instinct MI300X VF`), compute capability / target architecture (`gfx942`);
   - Total HBM3 memory capacity (192 GB), peak theoretical bandwidth (5.3 TB/s);
   - Host CPU architecture, cores, and system RAM.
2. **Software Stack & ROCm Environment:**
   - ROCm/HIP version via `torch.version.hip` and `hipcc --version`;
   - Verify `torch.cuda.is_available() == True` and `torch.cuda.device_count() >= 1`;
   - Check available backends: rocBLAS, rocSOLVER, TorchInductor ROCm backend, AMD Triton (`triton-amdgpu`);
   - Zero-NVIDIA check: confirm absence of CUDA libraries, `nvcc`, and CUDA toolkits.
3. **Synchronized GEMM Benchmark:**
   - Run warmup and steady-state GEMM benchmarks ($4096 \times 4096$) across `torch.bfloat16`, `torch.float16`, `torch.float32`, and `torch.float64` with explicit `torch.cuda.synchronize()`;
   - Record achieved TFLOPS and verify arithmetic sanity.

---

## 0.4 Unified Phase 0 Verification Pipeline

Implement a unified fail-fast test runner (`python3 analysis/verify_algebraic_primitives.py`) that executes in strict sequence:
1. Zero-transcendental AST static code audit;
2. MI300X hardware environment and HIP runtime probe;
3. Unit and property tests of all 12 algebraic primitives in fp64;
4. Machine-checked Lean 4 compilation (`/root/.elan/bin/lake build` in `formal/`);
5. Small-scale algebraic forward-backward autograd check.

---

## PASS Gates

- [ ] Static AST audit confirms exactly 0 transcendental function calls across production modules.
- [ ] fp64 CPU reference implementations pass against autograd and numerical finite differences with condition-aware error $< 5.0 \times 10^{-15}$.
- [ ] Algebraic Gate reflection symmetry $\beta(u) + \beta(-u) = 1$ is exact in fp64 (error $= 0.0$).
- [ ] ALU inflection point matches $-\sqrt{2}$ with residual $< 1.0 \times 10^{-15}$.
- [ ] AVN output variance satisfies $\operatorname{Var}(\operatorname{AVN}(\mathbf{x})) \in [0.9999, 1.0001]$ across input variance spreads.
- [ ] A-Softmax diagonal Jacobian is bounded by $\le 2.0$ across $10^4$ trials, and routing contrast on $\Delta s = 2.0$ matches $(2 + \sqrt{5})^8 = 103,682$ exactly.
- [ ] AGO Cayley rotation satisfies $\det(\mathbf{R}) = 1$ (error $< 1.0 \times 10^{-15}$), column orthogonality $< 1.0 \times 10^{-15}$, and shift equivariance error $< 1.0 \times 10^{-6}$.
- [ ] ACO factorized curvature preconditioning passes on ill-conditioned quadratics ($\kappa = 1000$) with zero loss divergence.
- [ ] 1x AMD Instinct MI300X (`gfx942`) is detected and validated via HIP, with no NVIDIA/CUDA dependency.
- [ ] Lean 4 formal build passes via `/root/.elan/bin/lake build` with 0 errors, 0 warnings, and 0 `sorry`.
- [ ] `results/phase0/PASS.md` is generated and committed, satisfying the shared PASS record contract.
