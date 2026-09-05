# Mathematical Analysis & Verification: The Algebraic Stack

This directory contains the reference Python / PyTorch implementation and empirical benchmark suite for the **Algebraic Stack**.

## Files Overview

1. **`algebraic_stack.py`**:
   - Complete reference PyTorch library implementing every layer of the Algebraic Stack with pure algebra ($0$ transcendental functions).
   - Core primitives:
     - `AlgebraicLinearUnit` (ALU): rational gating function with exact partition of unity.
     - `AlgebraicVarianceNorm` (AVN): reciprocal $\mathrm{rsqrt}$-based activation normalization.
     - `AlgebraicSoftmax` (A-Softmax): octic algebraic kernel with constant normalization.
     - `OptimalAlgebraicCrossEntropy` (OACE / $\mathcal{L}_{1/8}$): octic algebraic divergence via 3 hardware $\mathrm{rsqrt}$ ops.
     - `AlgebraicGeometricOscillator` (AGO): Cayley rational rotation matrix generator in $\mathrm{SO}(2)$.
     - `ALUGLU`: Algebraic gated linear unit feedforward network.
     - `AlgebraicMoE`: Sparse Mixture of Experts with Algebraic Noise Turbulance (ANT).
     - `AlgebraicCurvatureOptimizer` (ACO): Rank-1 factorized preconditioner requiring $\mathcal{O}(d_{\text{out}} + d_{\text{in}})$ memory.
     - `AlgebraicTransformerLM`: Full autoregressive language model stack.

2. **`verify_algebraic_primitives.py`**:
   - Unit test suite numerically verifying algebraic theorems against PyTorch autograd and numerical bounds.
   - Tests:
     - Reflection symmetry: $\beta(u) + \beta(-u) = 1$ ($0.0$ error).
     - Reciprocal symmetry: $(s+x)(s-x) = 1$ ($2.02 \times 10^{-14}$ error).
     - ALU derivative accuracy vs. PyTorch autograd ($2.22 \times 10^{-16}$ error).
     - A-Softmax Jacobian diagonal boundedness ($\leq 2.0$).
     - OACE monotonicity and contrast ratio ($> 10^5$).
     - AGO shift equivariance error ($< 10^{-6}$).
     - ACO convergence on ill-conditioned quadratics ($\kappa = 1000$).

3. **`benchmark_algebraic_vs_transcendental.py`**:
   - Head-to-head empirical comparison of the pure Algebraic Stack against the standard Transcendental Transformer (Softmax, GELU, RoPE, AdamW, Cross-Entropy).
   - Benchmarks:
     - **In-Context Sequence Intelligence**: Evaluates associative recall and sequence induction on dynamic key-value retrieval tasks.
     - **Optimizer Memory Footprint**: Compares ACO factorized state vs. AdamW full second moment across dimension scales ($512$ to $8192$).
     - **Sub-Byte Quantization Robustness**: Evaluates FP8/INT8/INT4 quantization degradation between A-Softmax and standard exponential Softmax.
     - **Asynchronous Ring Attention**: Verifies single-pass un-synchronized FlashAttention with zero communication overhead.

## Running the Analyses

```bash
# Run primitive mathematical verification:
python3 verify_algebraic_primitives.py

# Run comparative empirical benchmarks:
python3 benchmark_algebraic_vs_transcendental.py
```
