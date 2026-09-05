# The Algebraic Stack: Can Algebra and Algebra Alone Give Rise to Intelligence?

### A Complete Transcendental-Free Architecture for Memory-Efficient, Synchronization-Free, and Quantization-Robust Deep Learning

**Tasmai Keni**  
`tas.ken.rt25@dypatil.edu`

---

## Abstract

Contemporary deep learning architectures are saturated with transcendental functions. The exponential $e^{x}$ appears in every softmax attention layer and in every Swish/GELU feed-forward gate; the logarithm $\ln x$ appears in every cross-entropy loss and Kullback-Leibler divergence; the trigonometric pair $(\sin,\cos)$ appears in every sinusoidal and rotary positional encoding; and continuous exponential moving averages of squared gradients underlie every state-of-the-art adaptive optimizer. The conventional view treats this as a mere pipeline-latency question: transcendentals compile to range-reduced minimax polynomials and consume Special-Function-Unit (SFU) cycles. On modern accelerators, however, the inverse square root $\mathrm{rsqrt}$ and $e^{x}$ have nearly identical SFU pipeline latencies, and SFU throughput is rarely the binding bottleneck of a trained Transformer at scale. The true cost of transcendentals on contemporary silicon is not arithmetic; it is structural. Transcendental operations are mathematically hostile to the three hardware resources that actually limit large-scale training and inference—high-bandwidth memory (HBM) capacity, cross-node synchronization bandwidth, and low-precision numerical range—and to the two foundational algorithmic requirements that govern sequence modeling and optimization: shift equivariance in positional encoding and curvature-aware preconditioning in optimization.

This paper addresses a foundational research question: **Can algebra and algebra alone give rise to intelligence?** Specifically, can an artificial neural system acquire reasoning, sequence induction, associative recall, and hierarchical representation when every forward pass, backward pass, normalization layer, attention mechanism, loss functional, and optimizer update is restricted strictly to rational operations, polynomial compositions, and a single hardware-native algebraic radical—the inverse square root $\mathrm{rsqrt}(x) = 1 / \sqrt{x}$—with zero exponential, logarithmic, or trigonometric functions?

We answer this question affirmatively by constructing the **Algebraic Stack**: a complete, mathematically rigorous architecture for deep learning whose every operation is algebraic. The stack provides twelve foundational algebraic primitives, proved from first principles:
1. **HBM capacity.** Algebraic Variance Normalization (AVN) is parameter-free, eliminating the $d$-dimensional learnable scale vector loaded from HBM per token in LayerNorm/RMSNorm, while its Coupling Identity allows downstream algebraic gates to evaluate dynamically without redundant normalizations.
2. **Cross-node synchronization.** The Algebraic Softmax (A-Softmax) kernel $\rho(x) = x + \sqrt{x^{2} + 1}$ is strictly positive and bounded above under AVN pre-bounding, so Algebraic Flash Attention (AFA) requires no running-maximum subtraction. Its tile accumulation is purely additive, enabling asynchronous, lock-free Ring Attention across multi-node clusters with a single global All-Reduce, eliminating the per-tile serialization barrier that dominates large-context training.
3. **Quantization robustness.** The algebraic kernel $\rho$ is globally 2-Lipschitz, guaranteeing $\operatorname{Var}(\rho(X)) \leq 4 \operatorname{Var}(X)$ for any input distribution. After AVN pre-bounding, the A-Softmax operator has a Jacobian magnitude strictly bounded by $n / 4$, where $n$ is the algebraic sharpening exponent. Setting $n = 8 = 2^{3}$ yields routing ratios of order $10^{5}$ in three hardware squaring operations while preserving a 2-Lipschitz operator. Routing logits, attention scores, and MoE expert weights are natively representable in INT4 and FP4 without per-group calibration scales or QAT-time outlier suppression, provided block-level denominators are accumulated in FP32 SRAM scratch.
4. **Shift equivariance.** Algebraic Geometric Ordering (AGO) is constructed from static, per-head, content-independent frequency generators $\mathbf{A}_k = \omega_k \mathbf{J}$ on the rank-2 skew subalgebra $\mathfrak{so}(2)$. The Cayley transform yields a closed-form orthogonal rotation matrix $\mathbf{R}_k$ whose $m$-th power computes the absolute position-$m$ encoding and whose Gram product $\mathbf{R}_k^{n - m}$ enforces the relative attention identity $\langle \mathbf{Q}_m, \mathbf{K}_n \rangle = \mathbf{x}_q^{\top} \mathbf{R}_k^{n - m} \mathbf{x}_k$. Autoregressive decoding maintains $\mathcal{O}(1)$ complexity per token through cached $\mathbf{R}_k^{m - 1} \mapsto \mathbf{R}_k \mathbf{R}_k^{m - 1}$ matrix-vector updates without evaluating $\sin$ or $\cos$.
5. **Loss-landscape curvature without transcendentals.** The Algebraic Curvature Optimizer (ACO) remakes AdamW entirely within algebra. It replaces the full $\mathcal{O}(d_{\mathrm{out}} \cdot d_{\mathrm{in}})$ second-moment tensor with factorized algebraic row-column preconditioning in $\mathcal{O}(d_{\mathrm{out}} + d_{\mathrm{in}})$ HBM memory, computes moment updates and polynomial debiasing via exact rational algebra, schedules learning rates via an inverse-square-root rational decay schedule (ARDS) without cosine annealing, and enforces decoupled algebraic weight decay.

Every component is given complete mathematical derivations: closed-form gradients, Lipschitz certificates, shift-equivariance and orthogonality theorems, factorized preconditioning guarantees, and explicit backward graphs that contain no rsqrt, no division, and no transcendental function. We provide formal machine-checked proofs in Lean 4 and rigorous empirical verification in Python, demonstrating that pure algebraic architectures converge stably, match the expressive capacity of transcendental Transformers, and open a new foundation for memory-frugal, synchronization-free machine intelligence.

---

## 1 Introduction

### 1.1 The Structural Cost of Transcendentals

A contemporary large-scale Transformer is, at its arithmetic core, a composition of transcendental functions layered atop dense matrix multiplication. The attention mechanism normalizes query-key scores through a softmax whose per-element kernel is the exponential $e^{x}$. Every feed-forward block, in its dominant SwiGLU or GeGLU variant, gates through a Swish or GELU non-linearity, both of which embed $e^{x}$ inside a tensor four to eight times wider than the model dimension. Positional encoding—whether sinusoidal or rotary (RoPE)—requires $\sin$ and $\cos$. The training loss evaluates $-\ln p$ in its cross-entropy component and $\ln (y / p)$ whenever a Kullback-Leibler divergence is present. Mixture-of-experts routing contributes $E$ exponentials per token per layer, plus two logarithms per sample for Gumbel noise. Finally, the optimizer (AdamW) updates parameters using exponential moving averages and relies on transcendental cosine annealing schedules for convergence.

It is tempting to view this stack of transcendentals primarily through the lens of arithmetic latency. Each invocation of $e^{x}$ or $\ln x$ on contemporary silicon compiles to a range-reduced minimax polynomial, typically requiring tens of pipeline cycles per element, executed on a dedicated Special Function Unit (SFU). At the scale of frontier models with hundreds of billions of parameters, the cumulative SFU cost is non-trivial.

On modern accelerators, however, the inverse square root $\mathrm{rsqrt}$ and $e^{x}$ have nearly identical SFU pipeline latencies, and SFU throughput is rarely the binding constraint of a well-tuned training step. The genuinely dominant resources are high-bandwidth memory (HBM) capacity, cross-node synchronization bandwidth, and the dynamic range of sub-byte numerical formats. To these three hardware regimes we must add two purely algorithmic requirements that determine whether a deep architecture can learn long-horizon representations: shift equivariance in positional encoding (so that the model learns translation-invariant linguistic regularities and decodes autoregressively in $\mathcal{O}(1)$ per token), and curvature-aware preconditioning in optimization (so that gradient descent converges on ill-conditioned loss surfaces). It is in precisely these five regimes that transcendental operations exact their true structural cost.

### 1.2 Five Structural Hostilities

1. **HBM capacity inflation.** Standard normalization layers (LayerNorm, RMSNorm) compute a per-token statistic, divide by it, discard the statistic, and then multiply by a learnable channel-wise scale $\boldsymbol{\gamma} \in \mathbb{R}^d$. The scale vector is loaded from HBM for every token of every layer of every forward and backward pass. On memory-bandwidth-bound inference workloads (such as KV-cached autoregressive decoding), the cumulative bandwidth cost is substantial. Furthermore, AdamW inflates HBM footprint by maintaining an uncompressed $\mathcal{O}(d_{\mathrm{out}} \cdot d_{\mathrm{in}})$ second-moment tensor per weight matrix, doubling or tripling the memory required to host model state.
2. **Cross-node synchronization barriers.** Softmax, as implemented in every numerically stable framework, is not a pointwise function: it is a global reduction. The inner loop subtracts the row-wise maximum from every score before exponentiating, in order to prevent floating-point overflow in $e^x$. This max-subtraction is a strict synchronization barrier: no element of the row can be normalized until the maximum of the entire row has been established. Inside a single GPU's SRAM, this is an intra-tile reduction. Across a multi-node cluster running Ring Attention—where each device holds a slice of keys and values and must accumulate partial attention contributions sequentially around the ring—the max-reduction collapses to a strict serial dependence between tile transitions. Every tile transition demands cross-device communication of the running maximum and a multiplicative correction of the running denominator. At a scale of thousands of accelerators, this barrier dominates wall-clock time per attention call.
3. **Sub-byte quantization range inflation.** The exponential function is an extreme variance inflator: it maps linear input scales to exponential output scales, converting modest outliers in the logit distribution into outputs that dominate the entire softmax mass. For FP32 or FP16 inference, this is manageable. For INT4 and FP4 quantization of KV caches, attention scores, and MoE routing logits, a single outlier in the score distribution forces the per-group calibration scale to widen by orders of magnitude, destroying the precision of all non-outlier values in the same group. Quantization-Aware Training (QAT) recipes for sub-byte Transformers routinely mandate complex outlier suppression specifically to tame exponential amplification.
4. **Loss of shift equivariance under non-transcendental approximations.** Standard positional encodings rely on trigonometric rotations $(\sin m\theta, \cos m\theta)$ to achieve shift equivariance $\langle \mathbf{Q}_m, \mathbf{K}_n \rangle = f(\mathbf{x}_q, \mathbf{x}_k, n - m)$. Prior non-trigonometric approaches often introduced content-dependent or learned positional embeddings that break exact relative translation invariance, preventing length extrapolation and inflating decode latency. Any pure algebraic replacement must produce an orthogonal rotation per position, satisfy the Gram product identity $\mathbf{R}_m^\top \mathbf{R}_n = \mathbf{R}_{n - m}$, and extend autoregressively at $\mathcal{O}(1)$ cost per token without evaluating trigonometric series.
5. **Loss of curvature-aware preconditioning without transcendental optimizer state.** Adaptive optimizers succeed on Transformer training not because they reduce gradient noise, but because they precondition by an estimate of the inverse Fisher information matrix, restoring step size in directions where the loss surface is flat and shrinking step size where it is sharp. AdamW's $\sqrt{\hat{v}}$ denominator is a diagonal Fisher estimate; its cost is the full $\mathcal{O}(d_{\mathrm{out}} \cdot d_{\mathrm{in}})$ second-moment tensor per weight matrix in HBM, updated via continuous exponential moving averages. Furthermore, training relies on transcendental cosine annealing schedules to decay learning rates. Eliminating transcendentals while preserving curvature awareness requires a purely algebraic optimizer that factorizes curvature into $\mathcal{O}(d_{\mathrm{out}} + d_{\mathrm{in}})$ memory and schedules learning rates via rational decay.

### 1.3 The Core Research Direction: Can Algebra and Algebra Alone Give Rise to Intelligence?

The central question investigated in this work is:
$$\textbf{Can algebra and algebra alone give rise to intelligence?}$$

Historically, artificial intelligence has inherited its mathematical toolkit from 19th-century continuous physics:
- Statistical mechanics gave rise to the Boltzmann-Gibbs distribution $p_i \propto e^{-E_i / T}$, directly inspiring the softmax operator.
- Information theory defined entropy and mutual information via the logarithm $H(X) = -\sum p \ln p$, leading to log-likelihood objectives and cross-entropy loss.
- Continuous harmonic analysis motivated Fourier and sinusoidal positional representations $(\sin \omega t, \cos \omega t)$.
- Thermal relaxation kinetics motivated continuous exponential moving averages $e^{-\Delta t / \tau}$ in adaptive optimization.

We challenge the necessity of this transcendental foundation. We argue that cognitive intelligence—defined as the capacity for hierarchical abstraction, relational reasoning, associative memory retrieval, sequence induction, and in-context learning—does not fundamentally depend on transcendental calculus. Intelligence is fundamentally an algebraic phenomenon: it arises from the composition of multilinear maps, polynomial representations, projective normalizations, group-theoretic rotations, and rational optimization dynamics over ordered fields.

By eliminating every transcendental function, we demonstrate that algebraic operations are not merely approximations to continuous ideals; they provide strictly superior mathematical properties on discrete computing substrates: bounded derivatives, uniform Lipschitz guarantees, exact tile commutativity, rational attention sinks, and quantization stability.

### 1.4 Outline of the Algebraic Stack

The paper is organized around the foundational layers of the Algebraic Stack:
- **Section 2** establishes the three foundational primitives: the Algebraic Variance Scalar $\tau$, the Algebraic Gate $\beta$, and the Algebraic Kernel $\rho$.
- **Section 3** develops the Algebraic Linear Unit (ALU), establishing its $\mathcal{O}(1)$ backward pass, Lipschitz constant, and exact inflection point alignment with GELU.
- **Section 4** develops the Algebraic Softmax (A-Softmax) operator with algebraic power-sharpening ($n = 8$), the $\alpha$-Algebraic Cross-Entropy ($\alpha$-ACE) family with its canonical Octo-Algebraic Cross-Entropy (OACE) at $\alpha = 1/n$, proving the uniform Jacobian bound and FP4 quantization stability.
- **Section 5** develops the Algebraic Divergence (AD), establishing strict propriety on the simplex interior, Pearson $\chi^2$ equivalence, and elimination of gradient explosion under AVN pre-bounding.
- **Section 6** establishes Algebraic Variance Normalization (AVN), proving the zero-parameter HBM corollary and the Coupling Identity.
- **Section 7** presents Algebraic Geometric Ordering (AGO) via static rank-2 Cayley rotations, proving exact shift equivariance and $\mathcal{O}(1)$ autoregressive updates.
- **Sections 8 and 9** develop Algebraic Attention (AA) and Algebraic Flash Attention (AFA), proving single-pass exactness, tile commutativity, and asynchronous, lock-free Ring Attention across multi-node clusters.
- **Section 10** treats ALU-GLU, deriving its closed-form polynomial backward pass and universal approximation certificate.
- **Section 11** develops the Algebraic Mixture of Experts (A-MoE), introducing the Algebraic Noise Transform (ANT) and native FP4 sparse routing.
- **Section 12** presents the **Algebraic Curvature Optimizer (ACO)**: remaking AdamW with algebra alone, deriving factorized $\mathcal{O}(d_{\mathrm{out}} + d_{\mathrm{in}})$ curvature preconditioning, rational momentum, and the Algebraic Rational Decay Schedule (ARDS).
- **Section 13** develops Algebraic Byte Algebra (ABA), proving constant-bounded typo shatter against adversarial BPE tokenization.
- **Section 14** develops Algebraic Information Preservation (AIP), providing three structural anti-collapse certificates for latent-space training.
- **Section 15** presents a foundational mathematical and philosophical treatise addressing the core research question: "Can algebra and algebra alone give rise to intelligence?".
- **Section 16** provides an exhaustive structural comparison between empirical patches in frontier LLMs and native Algebraic Stack primitives.
- **Section 17** synthesizes the complete Algebraic Stack in comprehensive architectural tables.
- **Section 18** concludes the paper.

---

## 2 Algebraic Foundations

### 2.1 Notation and Field Structure

We work over the ordered field of real numbers $(\mathbb{R}, +, \cdot, \leq)$ with standard Euclidean inner product $\langle \cdot ,\cdot \rangle$. Vectors are denoted by bold lowercase letters $\mathbf{x},\mathbf{y},\mathbf{p},\dots$; matrices by bold uppercase $\mathbf{W},\mathbf{V},\mathbf{G},\dots$. The elementwise (Hadamard) product is $\odot$. For $\mathbf{x}\in \mathbb{R}^{d}$, the second raw moment is $m_{2}(\mathbf{x}) = \| \mathbf{x}\|^2 / d$. 

The hardware intrinsic $\mathrm{rsqrt}(z) = 1 / \sqrt{z}$ denotes the principal inverse square root; we assume $z > 0$ whenever applied. The closed $(K-1)$-simplex is $\Delta^{K - 1} = \{\mathbf{p}\in \mathbb{R}_{\geq 0}^{K} : \sum_{i=1}^K p_{i} = 1\}$ and its relative interior is $\operatorname{int}\Delta^{K - 1}$. The spectral norm of a matrix is $\| \cdot \|_{2}$ and the Frobenius norm is $\| \cdot \|_{F}$. For $\mathbf{v}\in \mathbb{R}^{n}$, $\mathbf{v}^{\odot 2}$ denotes elementwise squaring and $\mathrm{rsqrt}(\mathbf{v}) \coloneqq (\mathrm{rsqrt}(v_{1}),\dots,\mathrm{rsqrt}(v_{n}))$.

We define an operation as **purely algebraic** if it belongs to the algebraic closure of the field of rational functions $\mathbb{R}(x_1, \dots, x_k)$ under finite compositions of arithmetic operations $(+, -, \cdot, /)$ and radical extractions $(\sqrt[m]{\cdot})$. Crucially, the exponential $e^x$, logarithm $\ln x$, and trigonometric functions $(\sin x, \cos x)$ are transcendental over $\mathbb{R}(x)$ and are strictly forbidden.

### 2.2 The Algebraic Variance Scalar

A primitive computational object reused throughout the stack is the Algebraic Variance Scalar, a single hardware $\mathrm{rsqrt}$ applied to the second raw moment of an input vector.

**Definition 2.1 (Algebraic Variance Scalar).** For $\mathbf{x}\in \mathbb{R}^{d}$ and regularizer $\epsilon > 0$,
$$\tau (\mathbf{x};\epsilon)\coloneqq \mathrm{rsqrt}(m_{2}(\mathbf{x}) + \epsilon) = \frac{1}{\sqrt{\|{\mathbf{x}}\|^{2} / d + \epsilon}}. \quad (1)$$

The corresponding bounded vector is
$$\hat{\mathbf{x}}\coloneqq \tau (\mathbf{x};\epsilon)\mathbf{x}. \quad (2)$$

**Proposition 2.2 (Bounded Normalization).** For any $\mathbf{x}\in \mathbb{R}^{d}$ and any $\epsilon \geq 0$, $\|\hat{\mathbf{x}}\|^2 \leq d$, hence $|\hat{x}_{i}|\leq \sqrt{d}$ for every coordinate $i$. When $\epsilon = 0$ and $\mathbf{x}\neq \mathbf{0}$, $\|\hat{\mathbf{x}}\|^2 = d$ exactly.

*Proof.* $\|\hat{\mathbf{x}}\|^2 = \|\mathbf{x}\|^2 \tau^2 = \|\mathbf{x}\|^2 / (\|\mathbf{x}\|^2 / d + \epsilon) \leq \|\mathbf{x}\|^2 / (\|\mathbf{x}\|^2 / d) = d$, with equality at $\epsilon = 0$. Since $\hat{x}_i^2 \leq \|\hat{\mathbf{x}}\|^2 \leq d$, the coordinate-wise bound $|\hat{x}_i| \leq \sqrt{d}$ follows immediately. $\blacksquare$

Definition 2.1 is the unifying normalization primitive reused across A-Softmax (Section 4), AVN (Section 6), A-MoE routing (Section 11), and AIP anti-collapse regularizers (Section 14).

### 2.3 The Algebraic Gate

The foundational nonlinear activation of the entire Algebraic Stack is a smooth, bounded, monotone gate constructed from a single inverse-square-root evaluation.

**Definition 2.3 (Algebraic Gate).** For $x\in \mathbb{R}$ and smoothing constant $\kappa > 0$, the Algebraic Gate is
$$\beta (x;\kappa)\coloneqq \frac{1}{2}\left(1 + \frac{x}{\sqrt{x^2 + \kappa}}\right). \quad (3)$$
The canonical form is $\beta (x)\coloneqq \beta (x;1)$.

**Proposition 2.4 (Elementary Properties of the Algebraic Gate).** The gate $\beta (\cdot ;\kappa)$ satisfies for every $x\in \mathbb{R}$:
1. **Boundedness:** $\beta (x;\kappa)\in (0,1)$.
2. **Monotonicity:** $\beta^{\prime}(x;\kappa) = \frac{\kappa}{2(x^{2} + \kappa)^{3 / 2}} > 0$.
3. **Reflection symmetry:** $\beta (x;\kappa) + \beta (-x;\kappa) = 1$.
4. **Asymptotic saturation:** $\lim_{x\to +\infty}\beta (x;\kappa) = 1$ and $\lim_{x\to -\infty}\beta (x;\kappa) = 0$.
5. **Lipschitz bound:** $|\beta (x;\kappa) - \beta (y;\kappa)|\leq \frac{1}{2\sqrt{\kappa}} |x - y|$, with tight constant $1 / (2\sqrt{\kappa})$ saturated at $x = 0$.

*Proof.* Boundedness: Since $|x| < \sqrt{x^{2} + \kappa}$, the ratio $x / \sqrt{x^{2} + \kappa}\in (-1,1)$, hence $\beta \in (0,1)$. Monotonicity: Differentiating $x / \sqrt{x^2 + \kappa}$ yields $\kappa / (x^2 + \kappa)^{3/2} > 0$; dividing by 2 yields $\beta'$. Reflection: $\beta(x) + \beta(-x) = \frac{1}{2}(1 + x/\sqrt{x^2+\kappa}) + \frac{1}{2}(1 - x/\sqrt{x^2+\kappa}) = 1$. Saturation: As $x \to \pm \infty$, $x / \sqrt{x^2 + \kappa} \to \pm 1$. Lipschitz: $|\beta'(x;\kappa)| \leq \kappa / (2\kappa^{3/2}) = 1 / (2\sqrt{\kappa})$, achieved at $x = 0$; the mean value theorem completes the proof. $\blacksquare$

### 2.4 The Algebraic Linear Unit

**Definition 2.5 (Algebraic Linear Unit).** For $x\in \mathbb{R}$, the Algebraic Linear Unit (ALU) is
$$K(x)\coloneqq \frac{x}{2}\left(1 + \frac{x}{\sqrt{x^{2} + 1}}\right) = x\beta (x). \quad (4)$$
The defining cache variable is $u\coloneqq x / \sqrt{x^{2} + 1}\in (-1,1)$, computed once by a single $\mathrm{rsqrt}$ call, from which $K(x) = \frac{x}{2} (1 + u)$.

### 2.5 The Algebraic Kernel $\rho$

**Definition 2.6 (Algebraic Kernel).** For $x\in \mathbb{R}$ and $\kappa > 0$, the algebraic kernel is
$$\rho_{\kappa}(x)\coloneqq x + \sqrt{x^{2} + \kappa}. \quad (5)$$
The canonical form is $\rho (x)\coloneqq \rho_{1}(x) = x + \sqrt{x^{2} + 1}$.

**Proposition 2.7 (Properties of $\rho$).** The canonical kernel $\rho$ satisfies:
1. **Strict positivity:** $\rho (x) > 0$ for all $x\in \mathbb{R}$.
2. **Strict monotonicity:** $\rho '(x) = 1 + x / \sqrt{x^2 + 1} = 1 + u > 0$.
3. **Bounded derivative:** $\rho '(x)\in (0,2)$, so $\rho$ is globally 2-Lipschitz.
4. **Reciprocal symmetry:** $\rho (x)\cdot \rho (-x) = 1$, hence $1 / \rho (x) = \rho (-x) = \sqrt{x^2 + 1} - x$.
5. **Logarithmic derivative:** $\rho '(x) / \rho (x) = 1 / \sqrt{x^2 + 1}$.
6. **Asymptotics:** $\rho (x)\sim 2x$ as $x\rightarrow +\infty$; $\rho (x)\sim 1 / (2|x|)$ as $x\rightarrow -\infty$.
7. **Gate identity:** $\rho (x) = 2\sqrt{x^2 + 1}\beta (x)$.
8. **Power-sharpening identity:** For any positive integer $n$, $(\rho (x)^n)' = n\rho (x)^n /\sqrt{x^2 + 1}$.

*Proof.* (1) $\sqrt{x^2 + 1} > |x|$ implies $\rho(x) > 0$. (2) Direct differentiation: $\rho'(x) = 1 + x/\sqrt{x^2+1} = 1 + u > 0$ since $u \in (-1,1)$. (3) Since $u \in (-1, 1)$, $\rho'(x) \in (0, 2)$, so $\sup |\rho'(x)| = 2$. (4) $(x + \sqrt{x^2+1})(-x + \sqrt{x^2+1}) = (x^2 + 1) - x^2 = 1$. (5) $\rho'(x) / \rho(x) = (1 + u) / (\sqrt{x^2+1}(1 + u)) = 1 / \sqrt{x^2+1}$. (6) For $x \to -\infty$, $\sqrt{x^2+1} = |x|\sqrt{1 + 1/x^2} = |x| + 1/(2|x|) + \mathcal{O}(|x|^{-3})$; since $x = -|x|$, $\rho(x) = 1/(2|x|) + \mathcal{O}(|x|^{-3})$. (7) $2\sqrt{x^2+1}\beta(x) = 2\sqrt{x^2+1}\cdot \frac{1}{2}(1 + u) = \sqrt{x^2+1} + x = \rho(x)$. (8) Chain rule: $(\rho^n)' = n\rho^{n-1}\rho' = n\rho^n (\rho'/\rho) = n\rho^n / \sqrt{x^2+1}$ by (5). $\blacksquare$

---

## 3 The Algebraic Linear Unit (ALU)

### 3.1 Motivation and Definition

The smooth, non-monotonic activation functions Swish ($x\cdot \sigma(\beta x)$) and GELU ($x\cdot \Phi(x)$) have become standard in modern Transformer feed-forward networks. Both embed an exponential $e^x$ in their pointwise definition. Beyond SFU latency, this exponential incurs an HBM memory penalty: computing the backward pass requires saving $\sigma(x)$ or evaluating transcendental derivatives. The Algebraic Linear Unit (ALU), defined in equation (4), achieves the same geometric profile—unbounded positive linear growth, smooth non-monotonic suppression near the origin, and a bounded negative tail—via an entirely algebraic formula whose backward cache is a single scalar $u \in (-1, 1)$ per element.

### 3.2 Asymptotic Limits

**Theorem 3.1 (Asymptotic Limits of ALU).** The Algebraic Linear Unit satisfies:
$$\lim_{x\to +\infty} K(x) = x \quad \text{(identity at positive tail)}, \qquad \lim_{x\to -\infty} K(x) = 0 \quad \text{(suppression at negative tail)}. \quad (6)$$

*Proof.* As $x \to +\infty$, $u = x / \sqrt{x^2+1} \to 1$, hence $K(x) = \frac{x}{2}(1 + u) \to x$. As $x \to -\infty$, $u \to -1$, hence $K(x) \to \frac{x}{2}(1 - 1) = 0$. $\blacksquare$

### 3.3 The $\mathcal{O}(1)$ Polynomial Backward Pass

**Theorem 3.2 (Closed-Form Polynomial Backward Pass).** With the forward-pass cache variable $u = x / \sqrt{x^2 + 1}$, the derivative of ALU is
$$K^{\prime}(x) = \frac{1}{2}\big(1 + 2u - u^{3}\big). \quad (8)$$
The backward pass requires exactly two multiplications and two additions per element, with zero inverse square roots, zero divisions, and zero exponentials.

*Proof.* Differentiating $u = x / \sqrt{x^2 + 1}$:
$$\frac{du}{dx} = \frac{(x^2 + 1) - x^2}{(x^2 + 1)^{3/2}} = \frac{1}{(x^2 + 1)^{3/2}}. \quad (9)$$
By the product rule on $K(x) = \frac{x}{2}(1 + u)$:
$$K'(x) = \frac{1}{2}(1 + u) + \frac{x}{2}\frac{du}{dx} = \frac{1}{2}(1 + u) + \frac{x}{2(x^2 + 1)^{3/2}}.$$
Using the identity $1 / (x^2 + 1) = 1 - u^2$, we have $\frac{x}{(x^2+1)^{3/2}} = u(1 - u^2) = u - u^3$. Substituting yields:
$$K'(x) = \frac{1}{2}(1 + u) + \frac{1}{2}(u - u^3) = \frac{1}{2}(1 + 2u - u^3),$$
which is a pure cubic polynomial in $u$. $\blacksquare$

**Corollary 3.3 (Lipschitz Constant of ALU).** The Lipschitz constant of $K$ on $\mathbb{R}$ is $L_{K} = \frac{1}{2} (1 + 2u^{*} - (u^{*})^{3})$ with $u^{*} = \sqrt{2 / 3}$, giving $L_{K} = \frac{1}{2}(1 + \frac{4\sqrt{6}}{9}) \approx 1.04433$.

*Proof.* The maximum of $K'(x) = \frac{1}{2}(1 + 2u - u^3)$ over $u \in (-1, 1)$ occurs at $\frac{d}{du}K'(x) = \frac{1}{2}(2 - 3u^2) = 0$, giving $u^* = \sqrt{2/3}$. Evaluating $K'(u^*)$ gives $L_K \approx 1.04433$. $\blacksquare$

### 3.4 Inflection Point Theorem and Structural Alignment with GELU

**Theorem 3.4 (Inflection Point Theorem).** The second derivative of ALU satisfies $K''(x) = 0$ if and only if $u = -\sqrt{2/3}$, which occurs at $x = -\sqrt{2}$. This matches the inflection point of the Gaussian Error Linear Unit (GELU) $G(x) = x\Phi(x)$.

*Proof.* From Theorem 3.2, $K''(x) = \frac{1}{2}(2 - 3u^2)\frac{du}{dx}$. Since $\frac{du}{dx} = (x^2+1)^{-3/2} > 0$, $K''(x) = 0 \iff 2 - 3u^2 = 0 \iff u = \pm \sqrt{2/3}$. The negative inflection is $u = -\sqrt{2/3}$, which implies $x^2 / (x^2+1) = 2/3 \implies x^2 = 2 \implies x = -\sqrt{2}$. For GELU: $G''(x) = \phi(x)(2 - x^2)$, which vanishes exactly at $x = -\sqrt{2}$. Thus, setting $\kappa = 1$ in the algebraic gate uniquely matches the curvature profile of GELU. $\blacksquare$

---

## 4 Algebraic Softmax and Octo-Algebraic Cross-Entropy (A-Softmax and OACE)

### 4.1 Motivation

Standard softmax attention $p_i = e^{s_i} / \sum_j e^{s_j}$ incurs three structural liabilities:
1. **The max-reduction synchronization barrier:** Numerically stable implementations must subtract $\max_j s_j$ before exponentiating, preventing parallel tile accumulation across distributed Ring Attention nodes.
2. **Exponential variance inflation:** $e^X$ amplifies input variance exponentially, making sub-byte quantization (FP4/INT4) unstable due to large outlier scales.
3. **Sharpness-stability conflict:** Producing sharp routing requires unbounded logits, causing gradient explosion in the backward pass.

Algebraic Softmax (A-Softmax) resolves all three by replacing the exponential with the 2-Lipschitz algebraic kernel $\rho(x) = x + \sqrt{x^2 + 1}$, applied to AVN-pre-bounded logits and raised to an integer power $n$.

### 4.2 The AVN-Bounded A-Softmax Operator

**Definition 4.1 (A-Softmax Operator).** For input logits $\mathbf{s} \in \mathbb{R}^K$, sharpening exponent $n \geq 1$, and regularizer $\epsilon > 0$, the A-Softmax operator computes:
$$\tau = \mathrm{rsqrt}(m_2(\mathbf{s}) + \epsilon), \qquad \hat{s}_i = \tau s_i, \qquad \mathbf{S}_n(\mathbf{s})_i = \frac{\rho(\hat{s}_i)^n}{\sum_{j=1}^K \rho(\hat{s}_j)^n}. \quad (10)$$
The canonical A-Softmax uses sharpening exponent $n = 8 = 2^3$.

**Theorem 4.2 (Mathematical Validity of A-Softmax).** For every $\mathbf{s}\in \mathbb{R}^K$, $n \geq 1$, and $\epsilon > 0$:
1. $\mathbf{S}_n(\mathbf{s})_i > 0$ for all $i \in \{1, \dots, K\}$.
2. $\sum_{i=1}^K \mathbf{S}_n(\mathbf{s})_i = 1$.
3. $\mathbf{S}_n$ is $C^\infty$ on $\mathbb{R}^K \setminus \{\mathbf{0}\}$ and continuous on $\mathbb{R}^K$.
4. $\operatorname{argmax}_i \mathbf{S}_n(\mathbf{s})_i = \operatorname{argmax}_i s_i$.

*Proof.* Follows directly from Proposition 2.7(i, ii) and the strict monotonicity of $\tau s_i$ and the power function $z^n$ for $z > 0$. $\blacksquare$

### 4.3 Bounded-Input, Bounded-Output Geometry

**Theorem 4.3 (Strict Logit Bounding).** For every $\mathbf{s} \in \mathbb{R}^K$ and $\epsilon > 0$:
$$|\hat{s}_i| \leq \sqrt{K}, \quad \forall i \in \{1, \dots, K\}. \quad (13)$$

*Proof.* By Proposition 2.2, $\|\hat{\mathbf{s}}\|^2 \leq K$, hence $\hat{s}_i^2 \leq \|\hat{\mathbf{s}}\|^2 \leq K$. $\blacksquare$

**Corollary 4.4 (Strict Simplex Probability Floor).** For every $\mathbf{s}\in \mathbb{R}^K$, the A-Softmax probability is strictly bounded away from zero:
$$\mathbf{S}_n(\mathbf{s})_i \geq \frac{\rho(-\sqrt{K})^n}{K \rho(\sqrt{K})^n} = \frac{1}{K \rho(\sqrt{K})^{2n}}. \quad (14)$$

*Proof.* Monotonicity of $\rho$ implies $\rho(-\sqrt{K}) \leq \rho(\hat{s}_i) \leq \rho(\sqrt{K})$. The denominator satisfies $\sum_j \rho(\hat{s}_j)^n \leq K \rho(\sqrt{K})^n$. Using the reciprocal identity $\rho(-\sqrt{K})\rho(\sqrt{K}) = 1$ yields the bound. $\blacksquare$

### 4.4 Closed-Form Jacobian and the $n/4$ Lipschitz Bound

**Theorem 4.5 (Closed-Form Jacobian).** Let $\mathbf{p} = \mathbf{S}_n(\mathbf{s})$ and $w_j \coloneqq 1 / \sqrt{\hat{s}_j^2 + 1}$. The Jacobian of A-Softmax with respect to the AVN-bounded logits is:
$$\frac{\partial p_i}{\partial \hat{s}_j} = n w_j p_i (\delta_{ij} - p_j). \quad (15)$$

*Proof.* Follows from Proposition 2.7(viii): $(\rho(\hat{s})^n)' = n w \rho(\hat{s})^n$. Applying the quotient rule to $p_i = \rho(\hat{s}_i)^n / Z$ yields the result. $\blacksquare$

**Theorem 4.6 (Uniform Jacobian Operator Bound).** For every sharpening exponent $n \geq 1$, the diagonal Jacobian entry satisfies:
$$\left|\frac{\partial p_j}{\partial \hat{s}_j}\right| = n w_j p_j (1 - p_j) \leq \frac{n}{4}, \quad (17)$$
saturated at $\hat{s}_j = 0$ ($w_j = 1$) and $p_j = 1/2$. For the canonical $n = 8$, the diagonal derivative is bounded by 2, certifying that A-Softmax is a globally 2-Lipschitz operator.

### 4.5 Algebraic Sharpness at Bounded Logits

**Theorem 4.7 (Sharpness at Bounded Inputs).** For two AVN-bounded logits $\hat{s}_1 = 2$ and $\hat{s}_2 = 0$, the routing ratio under canonical $n = 8$ is:
$$\frac{p_1}{p_2} = \left(\frac{\rho(2)}{\rho(0)}\right)^8 = (2 + \sqrt{5})^8 \approx 1.044 \times 10^5. \quad (18)$$
Thus, a routing contrast exceeding $10^5$ is achieved within a bounded interval $[-2, 2]$ without requiring unbounded logits.

### 4.6 Hardware Efficiency: Power of Eight via Three Squarings

**Proposition 4.8 (Three-Squaring Power).** The power $\rho(\hat{s})^8$ is computed in exactly three sequential squaring operations:
$$\rho^2 = \rho \cdot \rho, \qquad \rho^4 = \rho^2 \cdot \rho^2, \qquad \rho^8 = \rho^4 \cdot \rho^4.$$
Zero transcendental function unit cycles are consumed.

### 4.7 Quantization Robustness and Rational Attention Sinks

**Corollary 4.10 (Native FP4/INT4 Quantization Stability).** Because $\rho$ is globally 2-Lipschitz, $\operatorname{Var}(\rho(X)) \leq 4\operatorname{Var}(X)$. Quantization noise on $\hat{s}$ propagates additively, not exponentially. A-Softmax logits, attention scores, and expert routing weights can be natively cast to FP4 without dynamic per-group scaling.

**Corollary 4.11 (Rational Attention Sinks).** As $\hat{s} \to -\sqrt{K}$, $\rho(\hat{s}) \sim 1 / (2|\hat{s}|)$. Irrelevant tokens contribute an algebraically suppressed tail $\mathcal{O}(|\hat{s}|^{-n})$ rather than underflowing to an absolute zero, serving as an automatic, native attention sink without manual sink tokens or large negative bias masks.

### 4.8 The $\alpha$-Algebraic Cross-Entropy Family and OACE

To eliminate the logarithm $-\ln p$ while maintaining proper scoring and bounded gradients, we define the $\alpha$-Algebraic Cross-Entropy family:

**Definition 4.12 ($\alpha$-Algebraic Cross-Entropy).** For target distribution $\mathbf{y} \in \Delta^{K-1}$, predicted distribution $\mathbf{p} = \mathbf{S}_n(\mathbf{s})$, and exponent $\alpha \in (0, 1]$,
$$\mathcal{L}_\alpha(\mathbf{s}, \mathbf{y}) \coloneqq \frac{1}{\alpha}\left(\sum_{i=1}^K \frac{y_i}{p_i^\alpha} - 1\right). \quad (20)$$
In the hard-label classification case $\mathbf{y} = \mathbf{e}_k$:
$$\mathcal{L}_\alpha(\mathbf{s}, \mathbf{e}_k) = \frac{1}{\alpha}\left(p_k^{-\alpha} - 1\right). \quad (21)$$
The canonical choice is $\alpha = 1/n$. For $n = 8$, this defines the **Octo-Algebraic Cross-Entropy (OACE)**:
$$\mathcal{L}_{\mathrm{OACE}}(\mathbf{s}, \mathbf{e}_k) = 8\left(p_k^{-1/8} - 1\right). \quad (22)$$

**Theorem 4.13 (Propriety and Non-Negativity).** For all $\alpha \in (0, 1]$, $\mathcal{L}_\alpha \geq 0$, with $\mathcal{L}_\alpha = 0$ if and only if $\mathbf{p} = \mathbf{y}$. In the limit $\alpha \to 0^+$, $\lim_{\alpha \to 0^+} \mathcal{L}_\alpha(\mathbf{s}, \mathbf{e}_k) = -\ln p_k$, recovering the classical cross-entropy as a continuous singular boundary of the algebraic family.

**Theorem 4.14 (Closed-Form Backward of OACE).** For hard label $\mathbf{y} = \mathbf{e}_k$, with $w_j = 1 / \sqrt{\hat{s}_j^2 + 1}$:
$$\frac{\partial \mathcal{L}_{1/n}}{\partial \hat{s}_j} = -n w_j p_k^{-1/n}(\delta_{kj} - p_j). \quad (23)$$

**Theorem 4.15 (Uniformly Bounded Gradient).** For every input $\mathbf{s}$, target $k$, and $n \geq 1$:
$$\left|\frac{\partial \mathcal{L}_{1/n}}{\partial \hat{s}_j}\right| \leq n p_k^{-1/n} \leq n K^{1/n} \rho(\sqrt{K})^2. \quad (25)$$
For $n = 8$, the $p_k^{-1}$ reciprocal singularity is tamed into an eighth-root $p_k^{-1/8}$, eliminating the gradient explosion problem of standard inverse-power losses.

**Proposition 4.16 (Three-Rsqrt Path for OACE).** The term $p_k^{-1/8}$ is computed in three sequential $\mathrm{rsqrt}$ operations:
$$z_1 = \mathrm{rsqrt}(p_k) = p_k^{-1/2}, \quad z_2 = \mathrm{rsqrt}(z_1^{-1}) = p_k^{-1/4}, \quad z_3 = \mathrm{rsqrt}(z_2^{-1}) = p_k^{-1/8}.$$
Thus, the forward loss and backward gradient are computed without logarithms, divisions, or transcendentals.

---

## 5 The Algebraic Divergence (AD)

### 5.1 Motivation and Definition

For soft-label supervision (distillation, RLHF policy regularization, label smoothing), the $\alpha = 1$ cross-entropy functional $\sum y_i / p_i - 1$ exhibits interior bias ($p_i^* \propto \sqrt{y_i}$). The Algebraic Divergence fixes this calibration while remaining completely algebraic.

**Definition 5.1 (Algebraic Divergence).** For $\mathbf{y} \in \Delta^{K-1}$ and $\mathbf{p} \in \operatorname{int} \Delta^{K-1}$:
$$D_A(\mathbf{y} \| \mathbf{p}) \coloneqq \sum_{i=1}^K \frac{y_i^2}{p_i} - 1. \quad (30)$$

**Theorem 5.2 (Pearson Equivalence and Strict Propriety).**
1. $D_A(\mathbf{y} \| \mathbf{p}) = \sum_{i=1}^K \frac{(y_i - p_i)^2}{p_i}$ is the Pearson $\chi^2$ divergence.
2. $D_A(\mathbf{y} \| \mathbf{p}) \geq 0$, with equality iff $\mathbf{p} = \mathbf{y}$.
3. **Fisher Equivalence:** At $\mathbf{p} = \mathbf{y}$, $\nabla_{\mathbf{p}}^2 D_A(\mathbf{y} \| \mathbf{p}) = 2 \nabla_{\mathbf{p}}^2 D_{\mathrm{KL}}(\mathbf{y} \| \mathbf{p})$. Gradient descent on $D_A$ shares the exact local Riemannian Fisher geometry of KL divergence up to a factor of 2.

**Theorem 5.3 (Closed-Form Gradient and Boundedness).** With $R = \sum_i y_i^2 / p_i$ and $r_j = y_j^2 / p_j$:
$$\frac{\partial D_A}{\partial \hat{s}_j} = n w_j (p_j R - r_j), \qquad \left|\frac{\partial D_A}{\partial \hat{s}_j}\right| \leq 2n K \rho(\sqrt{K})^{2n}. \quad (31, 33)$$
AVN pre-bounding guarantees that the gradient is uniformly bounded, completely preventing the underflow/overflow explosion of naive $\chi^2$ divergences.

---

## 6 Algebraic Variance Normalization (AVN)

### 6.1 The Zero-Parameter HBM Corollary

Standard LayerNorm and RMSNorm load a learnable scale vector $\boldsymbol{\gamma} \in \mathbb{R}^d$ from HBM on every pass. AVN eliminates $\boldsymbol{\gamma}$ entirely, operating as a pure geometric projection onto the $\sqrt{d}$-sphere.

**Definition 6.1 (AVN Layer).** For input $\mathbf{x} \in \mathbb{R}^d$ and regularizer $\epsilon > 0$:
$$v = m_2(\mathbf{x}) + \epsilon = \frac{1}{d}\|\mathbf{x}\|^2 + \epsilon, \qquad \tau = \mathrm{rsqrt}(v), \qquad \hat{\mathbf{x}} = \tau \mathbf{x}. \quad (34)$$
The normalized vector $\hat{\mathbf{x}}$ and the variance scalar $\tau$ are passed forward. Zero parameters are stored in or loaded from HBM.

### 6.2 The Coupling Identity

**Theorem 6.2 (Coupling Identity).** Let an algebraic gate have data-dependent smoothing constant $\kappa = v$. Then:
$$\beta(x; v) = \frac{1}{2}\left(1 + \frac{x}{\sqrt{x^2 + v}}\right) = \frac{1}{2}\left(1 + \frac{\tau x}{\sqrt{(\tau x)^2 + 1}}\right) = \beta(\hat{x}; 1). \quad (35)$$
Downstream gated activations reuse the pre-computed AVN scalar $\tau$ without computing a new $\mathrm{rsqrt}$.

**Theorem 6.3 (Closed-Form AVN Backward Pass).** For upstream gradient $\mathbf{g} = \partial \mathcal{L} / \partial \hat{\mathbf{x}}$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \tau \left(\mathbf{g} - \frac{\langle \mathbf{g}, \hat{\mathbf{x}} \rangle}{d} \hat{\mathbf{x}}\right). \quad (36)$$
The backward pass is an orthogonal projection along $\hat{\mathbf{x}}$, computable in dense matrix-vector operations with zero divisions and zero $\mathrm{rsqrt}$.

---

## 7 Algebraic Geometric Ordering (AGO)

### 7.1 The Shift-Equivariance Requirement Without Trigonometry

Rotary Position Embedding (RoPE) applies $2 \times 2$ rotation blocks:
$$\mathbf{R}_{\text{trig}}(\theta) = \begin{pmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{pmatrix},$$
which satisfy $\mathbf{R}_{\text{trig}}(m\theta)^\top \mathbf{R}_{\text{trig}}(n\theta) = \mathbf{R}_{\text{trig}}((n - m)\theta)$. This relative shift equivariance ensures that attention scores depend only on relative displacement $n - m$. We now prove that this exact Lie group structure can be realized through purely algebraic rational maps.

### 7.2 The Cayley Transform on $\mathfrak{so}(2)$

**Definition 7.1 (Cayley Transform).** For a skew-symmetric matrix $\mathbf{A} \in \mathfrak{so}(d)$ ($\mathbf{A}^\top = -\mathbf{A}$), the Cayley transform is:
$$\operatorname{Cay}(\mathbf{A}) \coloneqq (\mathbf{I} + \mathbf{A})(\mathbf{I} - \mathbf{A})^{-1}. \quad (37)$$

**Theorem 7.2 (Orthogonality of the Cayley Transform).** If $\mathbf{A}^\top = -\mathbf{A}$, then $\mathbf{R} = \operatorname{Cay}(\mathbf{A})$ is strictly orthogonal: $\mathbf{R}^\top \mathbf{R} = \mathbf{I}$, and $\det(\mathbf{R}) = 1$, so $\mathbf{R} \in \mathrm{SO}(d)$.

*Proof.* Note that $(\mathbf{I} + \mathbf{A})$ and $(\mathbf{I} - \mathbf{A})^{-1}$ commute because $(\mathbf{I} + \mathbf{A})(\mathbf{I} - \mathbf{A}) = \mathbf{I} - \mathbf{A}^2 = (\mathbf{I} - \mathbf{A})(\mathbf{I} + \mathbf{A})$. Therefore:
$$\mathbf{R}^\top = [(\mathbf{I} - \mathbf{A})^{-1}]^\top (\mathbf{I} + \mathbf{A})^\top = (\mathbf{I} + \mathbf{A})^{-1} (\mathbf{I} - \mathbf{A}).$$
Multiplying:
$$\mathbf{R}^\top \mathbf{R} = (\mathbf{I} + \mathbf{A})^{-1} (\mathbf{I} - \mathbf{A}) (\mathbf{I} + \mathbf{A}) (\mathbf{I} - \mathbf{A})^{-1} = (\mathbf{I} + \mathbf{A})^{-1} (\mathbf{I} + \mathbf{A}) (\mathbf{I} - \mathbf{A}) (\mathbf{I} - \mathbf{A})^{-1} = \mathbf{I}.$$
Thus $\mathbf{R}$ is orthogonal. $\blacksquare$

### 7.3 The Static Rank-2 Generator and Closed-Form Rational Rotation

For channel pair $k$ with base frequency $\omega_k > 0$, define the skew generator $\mathbf{A}_k = \omega_k \mathbf{J}$, where $\mathbf{J} = \begin{pmatrix} 0 & -1 \\ 1 & 0 \end{pmatrix}$.

**Theorem 7.3 (Closed-Form Rational Rotation Matrix).** The Cayley transform of $\mathbf{A}_k$ evaluates to the rational matrix:
$$\mathbf{R}_k = \operatorname{Cay}(\omega_k \mathbf{J}) = \frac{1}{1 + \omega_k^2} \begin{pmatrix} 1 - \omega_k^2 & -2\omega_k \\ 2\omega_k & 1 - \omega_k^2 \end{pmatrix}. \quad (40)$$
The matrix $\mathbf{R}_k$ is an exact rotation in $\mathrm{SO}(2)$ with rotation angle $\theta_k = 2 \arctan(\omega_k)$. Its entries are pure rational functions of $\omega_k$.

### 7.4 Positional Encoding by Repeated Multiplication and Shift Equivariance

**Definition 7.4 (AGO Positional Encoding).** The position-$m$ encoding matrix is the $m$-th matrix power:
$$\mathbf{R}_k(m) \coloneqq (\mathbf{R}_k)^m. \quad (41)$$
For a token vector $\mathbf{x} = (x_1, x_2)^\top$ at position $m$, the encoded vector is $\mathbf{x}^{(m)} = \mathbf{R}_k(m) \mathbf{x}$.

**Theorem 7.5 (Exact Shift Equivariance).** For any query position $m$ and key position $n$:
$$\langle \mathbf{Q}_m, \mathbf{K}_n \rangle = \mathbf{x}_q^\top \mathbf{R}_k(m)^\top \mathbf{R}_k(n) \mathbf{x}_k = \mathbf{x}_q^\top \mathbf{R}_k^{n - m} \mathbf{x}_k = f(\mathbf{x}_q, \mathbf{x}_k, n - m). \quad (42)$$
AGO satisfies the exact relative shift-equivariance contract of RoPE without evaluating any trigonometric function.

**Theorem 7.6 ($\mathcal{O}(1)$ Autoregressive Update).** During autoregressive inference, the rotation at step $m$ is updated from step $m-1$ via:
$$\mathbf{R}_k(m) = \mathbf{R}_k \cdot \mathbf{R}_k(m - 1), \quad (43)$$
requiring exactly 4 fused multiply-adds (FMAs) per channel pair and zero transcendentals.

---

## 8 Algebraic Attention (AA)

Algebraic Attention (AA) combines a local windowed A-Softmax track with a global linear associative memory track updated by an ALU-gated delta rule.

**Definition 8.1 (Global Associative Memory Track).** The memory matrix $\mathbf{S}_t \in \mathbb{R}^{d_v \times d_k}$ updates as:
$$\mathbf{S}_t = (1 - f_t)\mathbf{S}_{t-1} + \gamma_t (\mathbf{v}_t - \mathbf{S}_{t-1}\mathbf{k}_t)\mathbf{k}_t^\top, \quad (47)$$
where $f_t = \beta(\mathbf{w}_f^\top \mathbf{c}_t)$ is the algebraic forget gate and $\gamma_t = \beta(\mathbf{w}_\gamma^\top \mathbf{c}_t)$ is the algebraic write gate.

**Theorem 8.2 (Contractive Memory Stability).** Under normalized keys $\|\mathbf{k}_t\| = 1$, the transition matrix $\mathbf{M}_t = (1 - f_t)\mathbf{I} - \gamma_t \mathbf{k}_t \mathbf{k}_t^\top$ has eigenvalues in $(-1, 1)$, guaranteeing that $\|\mathbf{S}_t\|_F \leq \gamma_{\max} V / f_{\min}$ remains strictly bounded for all $t$.

---

## 9 Algebraic Flash Attention (AFA)

### 9.1 Elimination of the Max-Reduction Barrier

Standard FlashAttention maintains a running row-maximum $m_i$ across SRAM tiles to avoid $e^x$ overflow, requiring the cross-tile rescaling factor $e^{m_{\text{old}} - m_{\text{new}}}$. In distributed Ring Attention, communicating $m_i$ creates a serial cross-node barrier.

In Algebraic Flash Attention (AFA), because the kernel $\rho(\hat{s})^8$ is strictly positive and AVN pre-bounding guarantees $|\hat{s}| \leq \sqrt{N}$, $\rho(\hat{s})^8$ is naturally bounded in FP32. No max-subtraction is needed.

**Theorem 9.1 (Single-Pass Tile Additivity).** In AFA, partial numerators $\mathbf{N}_i^{(t)} = \sum_{j \in \text{tile}(t)} \rho(\hat{s}_{ij})^8 \mathbf{v}_j$ and partial denominators $D_i^{(t)} = \sum_{j \in \text{tile}(t)} \rho(\hat{s}_{ij})^8$ accumulate purely additively:
$$\mathbf{N}_i = \sum_{t} \mathbf{N}_i^{(t)}, \qquad D_i = \sum_{t} D_i^{(t)}, \qquad \mathbf{o}_i = \frac{\mathbf{N}_i}{D_i}. \quad (48)$$
The tiled accumulation is mathematically identical to un-tiled computation in exact arithmetic.

**Corollary 9.2 (Lock-Free Asynchronous Ring Attention).** Across $P$ distributed nodes, each node computes local sums $(\mathbf{N}_i^{(p)}, D_i^{(p)})$ independently. Sequence-wide attention requires only a single global $\mathrm{AllReduce}$ sum at the end, completely eliminating intermediate communication barriers between tiles.

---

## 10 ALU-GLU: The Algebraic Feed-Forward Block

**Definition 10.1 (ALU-GLU Block).** For input $\mathbf{x} \in \mathbb{R}^d$:
$$\mathbf{y} = \mathbf{W}_d [(\mathbf{W}_g \mathbf{x}) \odot K(\mathbf{W}_u \mathbf{x})], \quad (50)$$
where $K(b) = \frac{b}{2}(1 + b / \sqrt{b^2 + 1}) = b \beta(b)$ is the Algebraic Linear Unit.

**Theorem 10.2 (Closed-Form Polynomial Backward Graph).** With cached $u_j = b_j / \sqrt{b_j^2 + 1}$ and upstream gradient $\mathbf{g} = \partial \mathcal{L} / \partial \mathbf{y}$:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = (\mathbf{W}_d^\top \mathbf{g}) \odot \mathbf{a} \odot \frac{1}{2}(1 + 2\mathbf{u} - \mathbf{u}^{\odot 3}), \qquad \frac{\partial \mathcal{L}}{\partial \mathbf{a}} = (\mathbf{W}_d^\top \mathbf{g}) \odot K(\mathbf{b}). \quad (52)$$
Evaluating the backward pass requires zero $\mathrm{rsqrt}$ and zero transcendental calls.

**Theorem 10.3 (Universal Approximation).** Because the Algebraic Gate $\beta(x)$ is continuous, non-decreasing, and non-polynomial, feed-forward networks with ALU-GLU activations are universal approximators on compact subsets of $\mathbb{R}^d$ by the Leshno-Lin-Pinkus-Schocken theorem.

---

## 11 Algebraic Mixture of Experts (A-MoE)

### 11.1 The Algebraic Noise Transform (ANT)

Standard MoE routing uses Gumbel-Softmax noise: $g = -\ln(-\ln U)$. This requires two transcendental logarithms. We construct the **Algebraic Noise Transform (ANT)** via inverse transform sampling on the algebraic distribution:

**Definition 11.1 (Algebraic Distribution and ANT).** The algebraic distribution has CDF $F(\eta) = \beta(\eta) = \frac{1}{2}(1 + \eta / \sqrt{\eta^2 + 1})$. For uniform $U \in (0, 1)$ and regularizer $\epsilon_n > 0$, the ANT sample is:
$$\eta = F^{-1}(U) = \frac{2U - 1}{\sqrt{1 - (2U - 1)^2 + \epsilon_n}}. \quad (57)$$
Cost: 1 FMA, 1 $\mathrm{rsqrt}$, 1 multiplication. Zero logarithms.

### 11.2 The A-MoE Router

Logits are perturbed by ANT noise scaled by the token's AVN scalar $\tau_x$:
$$\tilde{r}_j = r_j + \tau_x \eta_j, \qquad \mathbf{p} = \mathbf{S}_8(\tilde{\mathbf{r}}). \quad (58, 59)$$
Tokens with high variance receive smaller perturbation (exploitation), while low-energy tokens receive larger noise (exploration).

**Theorem 11.3 (Algebraic Anti-Collapse).** The routing gradient on AVN-bounded logits satisfies:
$$\left|\frac{\partial p_j}{\partial \hat{r}_j}\right| \leq \frac{8 p_j}{\sqrt{\hat{r}_j^2 + 1}} \leq 8 p_j. \quad (60)$$
The term $1 / \sqrt{\hat{r}_j^2 + 1}$ attenuates gradients for over-confident experts, structurally mitigating routing collapse without auxiliary entropy regularizers.

---

## 12 The Algebraic Curvature Optimizer (ACO)

### 12.1 The Curvature Problem and the Memory Hostility of AdamW

The training of deep Transformer architectures exhibits severely ill-conditioned, non-convex loss landscapes characterized by anisotropic valleys with condition numbers $\kappa \gg 10^4$. First-order stochastic gradient descent (SGD) fails on these surfaces due to orthogonal gradient oscillation. AdamW resolves this by preconditioning updates with a diagonal estimate of the Fisher information matrix:
$$\theta_t = \theta_{t-1} - \eta_t \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right).$$

However, standard AdamW introduces three structural liabilities:
1. **Memory hostility:** It maintains two full state tensors ($m_t, v_t \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}$) in HBM for every 2D weight matrix, consuming $2 \times$ the parameter footprint and bottlenecking training on memory-capacity-limited hardware.
2. **Transcendental reliance:** The moving average updates are historically motivated by continuous exponential decay $e^{-\Delta t / \tau}$, and bias correction $(1 - \beta^t)$ is treated as exponential relaxation.
3. **Transcendental learning rate scheduling:** Practitioners rely on transcendental schedules, notably Cosine Annealing $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})(1 + \cos(\pi t / T))$, introducing $\cos$ into the optimization loop.

We now derive the **Algebraic Curvature Optimizer (ACO)**, remaking adaptive optimization entirely within algebra.

### 12.2 Purely Algebraic Rational Momentum and Debiasing

In ACO, moment updates are governed by rational constants $\beta_1, \beta_2 \in \mathbb{Q}$ (e.g., $\beta_1 = 9/10, \beta_2 = 999/1000$):
$$\mathbf{M}_t = \beta_1 \mathbf{M}_{t-1} + (1 - \beta_1) \mathbf{G}_t. \quad (61)$$
The bias-correction term is the rational polynomial:
$$\delta_1(t) = 1 - \beta_1^t, \qquad \delta_2(t) = 1 - \beta_2^t. \quad (62)$$
For integer step $t$, $\beta^t$ is an exact algebraic power computed in $\mathcal{O}(\log t)$ multiplications via binary exponentiation. The debiased first moment is $\hat{\mathbf{M}}_t = \mathbf{M}_t / \delta_1(t)$.

### 12.3 Factorized Algebraic Preconditioning ($\mathcal{O}(d_{\mathrm{out}} + d_{\mathrm{in}})$ Memory)

To eliminate the $\mathcal{O}(d_{\mathrm{out}} \cdot d_{\mathrm{in}})$ second-moment HBM storage, ACO decomposes the curvature into row and column marginal projections.

**Definition 12.1 (Factorized Curvature Accumulators).** For a weight gradient $\mathbf{G}_t \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}$, ACO maintains only two low-dimensional vectors in HBM:
$$\mathbf{r}_t = \beta_2 \mathbf{r}_{t-1} + (1 - \beta_2) \left(\frac{1}{d_{\mathrm{in}}} \sum_{j=1}^{d_{\mathrm{in}}} \mathbf{G}_{t, \cdot, j}^{\odot 2}\right) \in \mathbb{R}^{d_{\mathrm{out}}}, \quad (63)$$
$$\mathbf{c}_t = \beta_2 \mathbf{c}_{t-1} + (1 - \beta_2) \left(\frac{1}{d_{\mathrm{out}}} \sum_{i=1}^{d_{\mathrm{out}}} \mathbf{G}_{t, i, \cdot}^{\odot 2}\right) \in \mathbb{R}^{d_{\mathrm{in}}}. \quad (64)$$

The debiased marginal second moments are:
$$\hat{\mathbf{r}}_t = \frac{\mathbf{r}_t}{1 - \beta_2^t}, \qquad \hat{\mathbf{c}}_t = \frac{\mathbf{c}_t}{1 - \beta_2^t}. \quad (65)$$

**Definition 12.2 (ACO Preconditioned Update).** The algebraic preconditioner is synthesized on-the-fly inside SRAM via the rank-1 outer product and evaluated using a single $\mathrm{rsqrt}$:
$$\hat{\mathbf{V}}_{t, ij} = \sqrt{\hat{r}_{t, i} \hat{c}_{t, j}}, \qquad \mathbf{U}_{t, ij} = \hat{\mathbf{M}}_{t, ij} \cdot \mathrm{rsqrt}(\hat{r}_{t, i} \hat{c}_{t, j} + \epsilon^2). \quad (66)$$
The parameter update is:
$$\mathbf{W}_t = \mathbf{W}_{t-1} - \eta_t \mathbf{U}_t - \eta_t \lambda \mathbf{W}_{t-1}, \quad (67)$$
where $\lambda \in \mathbb{Q}$ is the decoupled algebraic weight decay factor.

**Theorem 12.3 (Memory Compression Guarantee).** For a weight tensor $\mathbf{W} \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}$, standard AdamW stores $2 d_{\mathrm{out}} d_{\mathrm{in}}$ optimizer state scalars. ACO stores $d_{\mathrm{out}} d_{\mathrm{in}}$ scalars for momentum plus $d_{\mathrm{out}} + d_{\mathrm{in}}$ scalars for curvature. When combined with rank-1 momentum factorization $\mathbf{M} \approx \mathbf{u}_m \mathbf{v}_m^\top$, the state scales as $\mathcal{O}(d_{\mathrm{out}} + d_{\mathrm{in}})$, achieving an exact compression factor of:
$$\frac{d_{\mathrm{out}} d_{\mathrm{in}}}{d_{\mathrm{out}} + d_{\mathrm{in}}} \approx \frac{d}{2} \gg 10^3. \quad (68)$$

**Theorem 12.4 (Kronecker Fisher Spectral Alignment).** Suppose the true gradient second moment follows a Kronecker-factored distribution $\mathbb{E}[\mathbf{G}_t^{\odot 2}] = \mathbf{a} \mathbf{b}^\top$ for positive vectors $\mathbf{a} \in \mathbb{R}^{d_{\mathrm{out}}}, \mathbf{b} \in \mathbb{R}^{d_{\mathrm{in}}}$. Then the factorized estimators $\hat{\mathbf{r}}_t$ and $\hat{\mathbf{c}}_t$ satisfy:
$$\mathbb{E}[\hat{\mathbf{r}}_t] = \bar{b} \cdot \mathbf{a}, \qquad \mathbb{E}[\hat{\mathbf{c}}_t] = \bar{a} \cdot \mathbf{b},$$
where $\bar{a} = \frac{1}{d_{\mathrm{out}}}\sum a_i$ and $\bar{b} = \frac{1}{d_{\mathrm{in}}}\sum b_j$. Consequently:
$$\mathbb{E}[\hat{\mathbf{r}}_{t, i}] \cdot \mathbb{E}[\hat{\mathbf{c}}_{t, j}] = (\bar{a}\bar{b}) \cdot (\mathbf{a}\mathbf{b}^\top)_{ij},$$
proving that the synthesized curvature $\sqrt{\hat{r}_i \hat{c}_j}$ is an exact spectral estimator of the true diagonal Fisher information matrix up to a global scalar.

### 12.4 Algebraic Rational Decay Schedule (ARDS)

Practitioners rely on transcendental cosine annealing $\frac{1}{2}(1 + \cos(\pi t / T))$ to decay the learning rate. We eliminate cosine entirely by introducing the **Algebraic Rational Decay Schedule (ARDS)**:

**Definition 12.5 (ARDS).** For maximum learning rate $\eta_{\max}$, warmup steps $T_{\mathrm{warm}}$, decay scale $T_{\mathrm{decay}}$, and curvature parameter $\alpha > 0$:
$$\eta(t) = \eta_{\max} \cdot \min\left(1, \frac{t}{T_{\mathrm{warm}}}\right) \cdot \mathrm{rsqrt}\left(1 + \alpha \left[\frac{\max(0, t - T_{\mathrm{warm}})}{T_{\mathrm{decay}}}\right]^2\right). \quad (69)$$

**Proposition 12.6 (Properties of ARDS).**
1. **Smoothness:** $\eta(t)$ is continuous and piecewise smooth.
2. **Warmup linearity:** For $t \leq T_{\mathrm{warm}}$, $\eta(t) = \eta_{\max}(t / T_{\mathrm{warm}})$.
3. **Algebraic decay:** For $t > T_{\mathrm{warm}}$, $\eta(t) \sim \mathcal{O}(1/t)$, matching the optimal theoretical rate for non-convex stochastic optimization.
4. **Hardware cost:** Evaluated in 1 subtraction, 1 square, 1 FMA, and 1 hardware $\mathrm{rsqrt}$. Zero trigonometric functions.

### 12.5 Global Convergence of the Algebraic Curvature Optimizer

**Theorem 12.7 (Convergence Bound on Smooth Non-Convex Objectives).** Let $\mathcal{L}: \mathbb{R}^P \to \mathbb{R}$ be $L$-Lipschitz smooth ($\|\nabla \mathcal{L}(\theta) - \nabla \mathcal{L}(\theta')\| \leq L\|\theta - \theta'\|$) and bounded below by $\mathcal{L}^*$. Let the stochastic gradient estimates have bounded variance $\mathbb{E}[\|\mathbf{G}_t - \nabla \mathcal{L}(\theta_t)\|^2] \leq \sigma^2$. Under the ACO update (Definition 12.2) with learning rate schedule $\eta_t = \eta_0 / \sqrt{t}$, the sequence of iterates satisfies:
$$\frac{1}{T} \sum_{t=1}^T \mathbb{E}[\|\nabla \mathcal{L}(\theta_t)\|^2] \leq \frac{C_1 (\mathcal{L}(\theta_0) - \mathcal{L}^*)}{\sqrt{T}} + \frac{C_2 \sigma^2 \ln(T)}{\sqrt{T}} = \mathcal{O}\left(\frac{1}{\sqrt{T}}\right), \quad (70)$$
guaranteeing convergence to a stationary point at the minimax optimal rate.

---

## 13 Algebraic Byte Algebra (ABA)

Byte-Pair Encoding (BPE) tokenization is a non-Lipschitz, discrete pre-processing step. A single-character typo can cause the tokenizer to partition an entire sequence into completely different token IDs, a vulnerability we term **Typo Shatter**.

**Theorem 13.1 (BPE Typo Shatter Lower Bound).** For any BPE tokenizer over an adversarial alphabet, a single-byte substitution creates an embedding displacement growing as $\Omega(\sqrt{L})$.

**Definition 13.2 (Algebraic Byte Algebra Layer).** ABA ingests raw bytes $b_j \in \{0, \dots, 255\}$:
$$\mathbf{e}_j = \mathbf{E}[b_j] + \mathbf{P}[j \bmod W], \quad \mathbf{h}_j = \mathbf{W}_d [(\mathbf{W}_g \mathbf{e}_j) \odot K(\mathbf{W}_u \mathbf{e}_j)], \quad \mathbf{x}_i = \mathbf{W}_o \left(\frac{1}{W}\sum_{j \in \mathrm{patch}(i)} \mathbf{h}_j\right). \quad (71)$$

**Theorem 13.3 (Constant-Bounded Typo Shatter).** Under a single-byte perturbation, the output perturbation of ABA satisfies:
$$\|F(\mathbf{b}) - F(\mathbf{b}')\|_F \leq \frac{1}{W}(1 + L_K)\|\mathbf{W}_g\|_2 \|\mathbf{W}_u\|_2 \|\mathbf{W}_o\|_2 (M_E + M_P) = \mathcal{O}(1), \quad (72)$$
which is strictly bounded by a constant independent of sequence length $L$.

---

## 14 Algebraic Information Preservation (AIP)

To prevent representation collapse in self-supervised or latent-predictive architectures without transcendental VICReg hinges or Barlow Twins logarithms, we introduce **Algebraic Information Preservation (AIP)**:

1. **Anti-Roughness (Lipschitz Certificate):** Algebraic power iteration uses $\mathrm{rsqrt}$ to estimate spectral norms without division:
   $$\mathbf{u}_{t+1} = \mathbf{W} \mathbf{v}_t \cdot \mathrm{rsqrt}(\mathbf{v}_t^\top \mathbf{W}^\top \mathbf{W} \mathbf{v}_t), \qquad \mathbf{v}_{t+1} = \mathbf{W}^\top \mathbf{u}_{t+1} \cdot \mathrm{rsqrt}(\mathbf{u}_{t+1}^\top \mathbf{W} \mathbf{W}^\top \mathbf{u}_{t+1}). \quad (74)$$
   Spectrally normalized layers ensure $\operatorname{Lip}(\Phi) \leq L_K^D \approx (1.0445)^D$.
2. **Anti-Dimensional Collapse (Algebraic Covariance Divergence):** For correlation matrix $\mathbf{C} = \frac{1}{N}\mathbf{Z}^\top \mathbf{Z}$ with unit diagonal:
   $$\mathcal{L}_{\mathrm{AIP}}(\mathbf{Z}) = \frac{1}{2}\left(\operatorname{Tr}(\mathbf{C}^2) - d\right) = \frac{1}{2}\|\mathbf{C} - \mathbf{I}\|_F^2 = \frac{1}{2}\sum_{i \neq j} C_{ij}^2. \quad (76)$$
   $\mathcal{L}_{\mathrm{AIP}} = 0 \iff \mathbf{C} = \mathbf{I}$, guaranteeing full-rank representations via a pure polynomial.
3. **Anti-Mode Collapse (AVN Repulsion Field):** Differentiating the AVN scalar $\tau = (v + \epsilon)^{-1/2}$ yields $|\partial \tau / \partial v| = \frac{1}{2}(v + \epsilon)^{-3/2} \to \frac{1}{2\epsilon^{3/2}}$ as $v \to 0$, providing an automatic repulsive potential that pushes collapsed clusters apart.

---

## 15 Foundational Analysis: Can Algebra and Algebra Alone Give Rise to Intelligence?

Having constructed the complete Algebraic Stack, we now directly address the core research direction:
$$\textbf{Can algebra and algebra alone give rise to intelligence?}$$

### 15.1 The Historical Transcendental Dogma

For over seven decades, machine learning has operated under the implicit assumption that transcendental functions are indispensable prerequisites for intelligence:
- The exponential $e^x$ was assumed necessary to define Gibbs-Boltzmann probability distributions over discrete states.
- The logarithm $\ln x$ was assumed necessary to quantify information, measure surprise, and compute Shannon entropy.
- Trigonometric functions $(\sin, \cos)$ were assumed necessary to represent rotational geometries and translation-invariant sequence coordinates.
- Continuous differential equations and exponential decays were assumed necessary to model temporal memory and adaptive optimization.

This assumption is historically contingent, not mathematically necessary. Transcendentals entered computation because 19th-century mathematicians lacked digital silicon and relied on continuous analytic functions whose infinitesimal derivatives admitted paper-and-pencil closed forms. Digital accelerators, however, operate on discrete finite-precision registers. In this regime, transcendentals become liabilities: they require range-reduction polynomials, suffer from exponential dynamic range explosion, and create serial communication barriers.

### 15.2 The Algebraic Completeness of Neural Representation

Can algebraic operations approximate arbitrary cognitive functions? We state and prove the Algebraic Universality Theorem:

**Theorem 15.1 (Algebraic Universality on Compact Sets).** Let $\mathcal{K} \subset \mathbb{R}^d$ be a compact domain, and let $f \in C(\mathcal{K}, \mathbb{R})$ be any continuous target function. Let $\mathcal{A}_{\mathrm{alg}}$ denote the family of single-hidden-layer networks whose activations are restricted to the Algebraic Linear Unit $K(x) = \frac{x}{2}(1 + x/\sqrt{x^2+1})$:
$$\mathcal{A}_{\mathrm{alg}} = \left\{ g(\mathbf{x}) = \sum_{i=1}^M c_i K(\mathbf{w}_i^\top \mathbf{x} + b_i) : c_i, b_i \in \mathbb{R}, \mathbf{w}_i \in \mathbb{R}^d, M \in \mathbb{N} \right\}.$$
Then $\mathcal{A}_{\mathrm{alg}}$ is uniformly dense in $C(\mathcal{K}, \mathbb{R})$: for every $\epsilon > 0$, there exists $g \in \mathcal{A}_{\mathrm{alg}}$ such that $\sup_{\mathbf{x} \in \mathcal{K}} |f(\mathbf{x}) - g(\mathbf{x})| < \epsilon$.

*Proof.* By the Leshno-Lin-Pinkus-Schocken Theorem (1993), an activation function $\sigma: \mathbb{R} \to \mathbb{R}$ achieves universal approximation in $C(\mathcal{K})$ if and only if $\sigma$ is not an algebraic polynomial. The ALU activation $K(x) = \frac{x}{2}(1 + x/\sqrt{x^2+1})$ involves the square root radical $\sqrt{x^2+1}$. Suppose for contradiction that $K(x)$ were a polynomial $P(x) \in \mathbb{R}[x]$. Then $x/\sqrt{x^2+1} = 2P(x)/x - 1$ would be a rational function $Q(x) \in \mathbb{R}(x)$, implying $x^2 / (x^2 + 1) = Q(x)^2$, so $x^2(Q_{\mathrm{den}}(x))^2 = (x^2+1)(Q_{\mathrm{num}}(x))^2$. But $x^2 + 1$ is irreducible over $\mathbb{R}[x]$ and has simple roots $\pm i$, whereas in any square of a polynomial in $\mathbb{R}[x]$, all complex roots have even multiplicity. This contradiction proves that $K(x)$ is not a polynomial. Since $K(x)$ is continuous and non-polynomial, $\mathcal{A}_{\mathrm{alg}}$ is dense in $C(\mathcal{K})$. $\blacksquare$

### 15.3 Group-Theoretic Equivariance via Pure Algebra

Intelligence requires representing symmetries—specifically the translation group $(\mathbb{R}, +)$ in temporal and spatial reasoning. RoPE relies on the Lie group isomorphism $(\mathbb{R}, +) \to \mathrm{SO}(2)$ via the transcendental exponential map $\theta \mapsto \exp(\theta \mathbf{J}) = \cos \theta \mathbf{I} + \sin \theta \mathbf{J}$.

Theorem 7.2 and 7.3 prove that the Cayley transform $\operatorname{Cay}(\omega \mathbf{J}) = (\mathbf{I} + \omega \mathbf{J})(\mathbf{I} - \omega \mathbf{J})^{-1}$ is an exact algebraic birational map from the Lie algebra $\mathfrak{so}(2)$ onto the Lie group $\mathrm{SO}(2) \setminus \{-\mathbf{I}\}$. The composition of group actions corresponds to repeated matrix multiplication:
$$\mathbf{R}_k(m) = (\mathbf{R}_k)^m.$$
Because $\mathbf{R}_k(m)^\top \mathbf{R}_k(n) = \mathbf{R}_k^{n-m}$, relative displacement is represented exactly without transcendentals. Symmetries in machine intelligence do not require continuous analytic functions; they are exact algebraic properties of orthogonal matrices.

### 15.4 Information Geometry Without Logarithms

Does information measurement require the logarithm? The historical justification for $-\sum p \ln p$ is Shannon's additivity axiom for independent events: $I(A \cap B) = I(A) + I(B)$. However, in neural training, the loss function is an optimization surrogate whose purpose is to provide a steep, non-vanishing gradient signal aligned with the Riemannian Fisher information metric.

Theorem 5.2 establishes that the Algebraic Divergence $D_A(\mathbf{y} \| \mathbf{p}) = \sum y_i^2 / p_i - 1$ has a Riemannian Hessian equal to $2 \nabla^2 D_{\mathrm{KL}}$ at the optimum. Theorem 4.15 proves that the Octo-Algebraic Cross-Entropy $\mathcal{L}_{1/8} = 8(p_k^{-1/8} - 1)$ possesses a strictly bounded gradient $8 p_k^{-1/8} \leq 8 K^{1/8}\rho(\sqrt{K})^2$, completely eliminating the numerical instability of logarithmic loss. Information geometry on the probability simplex is an algebraic Riemannian geometry.

### 15.5 Curvature-Aware Optimization Without Continuous Transcendentals

Does adaptive optimization require continuous exponential integrals? Theorem 12.4 proves that factorized row-column projections $\mathbf{r}_t \otimes \mathbf{c}_t$ approximate the diagonal Fisher information matrix with spectral fidelity under Kronecker covariance. The learning rate schedule ARDS decays as $1/\sqrt{1 + \alpha t^2} \sim 1/t$, matching the optimal asymptotic convergence rate for stochastic non-convex optimization (Theorem 12.7) with zero trigonometric calls.

### 15.6 The Affirmative Answer

We conclude:
$$\textbf{Yes: Algebra and algebra alone can give rise to intelligence.}$$

Intelligence does not reside in the transcendental nature of $e^x$ or $\ln x$. It emerges from:
1. **Multilinear compositional capacity** (dense matrix multiplications).
2. **Continuous, non-polynomial thresholding** (the Algebraic Gate $\beta$ and ALU).
3. **Projective normalization on bounded manifolds** (AVN).
4. **Relational routing and associative memory** (A-Softmax and AA).
5. **Exact rotational group actions** (AGO Cayley rotations).
6. **Curvature-aligned Riemannian preconditioning** (ACO).

Every one of these mechanisms is purely algebraic. By purging transcendentals, we do not compromise expressive power; we gain numerical stability, HBM efficiency, and synchronization-free distributed scaling.

---

## 16 Structural Comparison with State-of-the-Art

Frontier deployments have developed complex engineering heuristics to patch the structural failures of transcendental architectures. Table 1 demonstrates that each heuristic is an ad-hoc fix for a transcendental defect, whereas the Algebraic Stack provides the solution natively.

**Table 1: Structural comparison between frontier LLM engineering patches and native Algebraic Stack primitives.**

| Failure Mode / Concern | Frontier Empirical Patch | Algebraic Primitive | Mathematical Mechanism |
| :--- | :--- | :--- | :--- |
| **Attention Sinks** | Learnable sink logit / token at pos 0 | A-Softmax Kernel $\rho$ | $\rho(\hat{s}) \sim \frac{1}{2\|\hat{s}\|}$ rational tail provides built-in sink |
| **Positional Encoding** | Trigonometric RoPE ($\sin, \cos$) | Algebraic Geometric Ordering (AGO) | Static Cayley $\mathbf{R}_k = (\mathbf{I} + \omega_k\mathbf{J})(\mathbf{I} - \omega_k\mathbf{J})^{-1}$ via 4 FMAs |
| **Ring Attention Sync** | Serial max-reduction across nodes | Algebraic Flash Attention (AFA) | Strictly positive $\rho^8$ allows single-pass lock-free AllReduce |
| **Quantization Outliers** | QAT outlier suppression / carve-outs | 2-Lipschitz A-Softmax | $\operatorname{Var}(\rho(X)) \leq 4\operatorname{Var}(X)$; uniform $n/4$ Jacobian bound |
| **MoE Routing Collapse** | Auxiliary entropy / load-balance loss | A-MoE Router | $w_j = (1 + \hat{r}_j^2)^{-1/2}$ naturally attenuates confident experts |
| **MoE Gumbel Noise** | Transcendental $g = -\ln(-\ln U)$ | Algebraic Noise Transform (ANT) | Inverse-CDF $\eta = (2U-1)/\sqrt{1 - (2U-1)^2 + \epsilon_n}$ |
| **HBM Normalization** | Learnable $\boldsymbol{\gamma}$ vector in HBM | AVN Layer | Zero-parameter projection; Coupling Identity $\beta(x; v) = \beta(\hat{x}; 1)$ |
| **Optimizer Memory** | Full $\mathcal{O}(d_{\mathrm{out}} d_{\mathrm{in}})$ AdamW state | Algebraic Curvature Optimizer (ACO) | Factorized row-column projections in $\mathcal{O}(d_{\mathrm{out}} + d_{\mathrm{in}})$ memory |
| **Learning Rate Schedule**| Cosine Annealing $\frac{1}{2}(1 + \cos(\pi t / T))$ | Algebraic Rational Decay (ARDS) | Rational decay $\eta_t \propto \mathrm{rsqrt}(1 + \alpha t^2)$ via 1 $\mathrm{rsqrt}$ |
| **Loss Gradient Explosion**| Gradient clipping / Logit soft-capping | Octo-Algebraic Cross-Entropy (OACE) | Gradient bounded by $8 p_k^{-1/8} \leq 8 K^{1/8} \rho(\sqrt{K})^2$ |
| **Typo Shatter in Tokens**| Character fallbacks / Subword heuristic | Algebraic Byte Algebra (ABA) | Constant-bounded Lipschitz shatter norm $\mathcal{O}(1)$ |
| **Latent Mode Collapse** | Barlow Twins log / VICReg hinge | Algebraic Information Preservation (AIP) | Off-diagonal norm $\|\mathbf{C} - \mathbf{I}\|_F^2$ and AVN repulsive field |

---

## 17 The Complete Algebraic Stack: Summary

Table 2 presents the twelve unified primitives of the Algebraic Stack, their transcendental targets, and their proved theorems.

**Table 2: Complete specification of the Algebraic Stack.**

| Component | Standard Target Replaced | Algebraic Formulation | Defining Mathematical Guarantee |
| :--- | :--- | :--- | :--- |
| **ALU** | GELU, Swish | $K(x) = \frac{x}{2}(1 + u), u = x \cdot \mathrm{rsqrt}(x^2 + 1)$ | $\mathcal{O}(1)$ backward pass; $L_K \approx 1.0445$; Inflection at $-\sqrt{2}$ (Thm 3.2, 3.4) |
| **A-Softmax** | Softmax | $\mathbf{S}_n(\mathbf{s}) = \rho(\hat{\mathbf{s}})^n / \sum \rho(\hat{\mathbf{s}})^n, n = 8$ | 2-Lipschitz operator; $10^5$ contrast at bounded inputs; INT4/FP4 stable (Thm 4.6, 4.7) |
| **OACE** | Cross-Entropy ($-\ln p$) | $\mathcal{L}_{1/8} = 8(p_k^{-1/8} - 1)$ | 3-rsqrt backward; strictly bounded gradient $8 p_k^{-1/8}$ (Thm 4.15, Prop 4.16) |
| **AD** | KL Divergence | $D_A(\mathbf{y} \| \mathbf{p}) = \sum y_i^2 / p_i - 1$ | Pearson $\chi^2$ equivalence; Riemannian Fisher equivalence; Bounded gradient (Thm 5.2, 5.3) |
| **AVN** | LayerNorm, RMSNorm | $\tau = \mathrm{rsqrt}(m_2(\mathbf{x}) + \epsilon), \hat{\mathbf{x}} = \tau \mathbf{x}$ | Zero parameters; Coupling Identity $\beta(x; v) = \beta(\hat{x}; 1)$ (Def 6.1, Thm 6.2) |
| **AGO** | RoPE, Sinusoidal PE | $\mathbf{R}_k = (\mathbf{I} + \omega_k\mathbf{J})(\mathbf{I} - \omega_k\mathbf{J})^{-1}$ | Exact shift equivariance $\langle\mathbf{Q}_m,\mathbf{K}_n\rangle = f(n - m)$; $\mathcal{O}(1)$ decode (Thm 7.5, 7.6) |
| **AA** | Softmax Attention | Dual-track: local A-Softmax + ALU delta rule | Linear global associative memory; contractive stability $\|\mathbf{S}_t\|_F < \infty$ (Thm 8.2) |
| **AFA** | FlashAttention-2 | Additive tile accumulation without max reduction | Lock-free asynchronous Ring Attention via single AllReduce (Thm 9.1, Cor 9.2) |
| **ALU-GLU** | SwiGLU, GeGLU | $\mathbf{W}_d [(\mathbf{W}_g \mathbf{x}) \odot K(\mathbf{W}_u \mathbf{x})]$ | Polynomial backward in cached $u$; Universal approximation (Thm 10.2, 10.3) |
| **A-MoE** | Softmax + Gumbel MoE | AVN-bounded $\rho^8$ routing + ANT noise | Native FP4 routing; variance-adaptive exploration; anti-collapse (Thm 11.3, Cor 4.10) |
| **ACO** | AdamW Optimizer | Factorized curvature $r_i \otimes c_j$ + ARDS schedule | $\mathcal{O}(d_{\mathrm{out}} + d_{\mathrm{in}})$ memory; rational momentum; $\mathcal{O}(1/\sqrt{T})$ rate (Thm 12.3, 12.7) |
| **ABA** | BPE Tokenizer | Patch-pooled ALU-GLU on raw bytes | Constant-bounded typo shatter $\mathcal{O}(1)$ vs BPE $\Omega(\sqrt{L})$ (Thm 13.3) |
| **AIP** | VICReg, Barlow Twins | Power iteration + $\|\mathbf{C} - \mathbf{I}\|_F^2$ + AVN repulsion | Structural anti-roughness, anti-dimension, and anti-mode collapse (Section 14) |

Every component shares the identical execution profile: dense matrix multiplications, additions, fused multiply-adds, and hardware-pipelined inverse square roots. The backward graph of every component is a polynomial in cached forward state.

---

## 18 Conclusion

This paper has investigated the foundational question: **Can algebra and algebra alone give rise to intelligence?**

Through the construction and formalization of the **Algebraic Stack**, we have established that transcendental operations ($e^x, \ln x, \sin x, \cos x$, exponential moving averages, cosine schedules) are entirely dispensable in deep learning. Their entrenchment was a historical artifact of continuous analytic physics rather than a computational necessity for intelligence.

By remaking the entire architecture—from the Algebraic Linear Unit (ALU) and Algebraic Softmax (A-Softmax) to Algebraic Geometric Ordering (AGO), the Octo-Algebraic Cross-Entropy (OACE), and the Algebraic Curvature Optimizer (ACO)—purely within algebra:
1. We eliminate the max-reduction synchronization barrier in FlashAttention and Ring Attention, enabling lock-free distributed scaling.
2. We eliminate the exponential variance inflation of softmax, enabling native sub-byte (FP4/INT4) inference and training without outlier suppression.
3. We eliminate the $\mathcal{O}(d_{\mathrm{out}} \cdot d_{\mathrm{in}})$ memory bloat of AdamW, replacing it with factorized $\mathcal{O}(d_{\mathrm{out}} + d_{\mathrm{in}})$ algebraic curvature.
4. We preserve exact shift equivariance, universal approximation, and optimal stochastic non-convex convergence rates.

Algebra and algebra alone is sufficient to construct robust, scalable, and memory-frugal artificial intelligence.

---

## References

[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems 30.

[2] Shazeer, N. (2020). GLU Variants Improve Transformer. arXiv:2002.05202.

[3] Ramachandran, P., Zoph, B., Le, Q.V. (2017). Searching for Activation Functions. arXiv:1710.05941.

[4] Hendrycks, D., Gimpel, K. (2016). Gaussian Error Linear Units (GELUs). arXiv:1606.08415.

[5] Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., Liu, Y. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864.

[6] Ba, J.L., Kiros, J.R., Hinton, G.E. (2016). Layer Normalization. arXiv:1607.06450.

[7] Zhang, B., Sennrich, R. (2019). Root Mean Square Layer Normalization. Advances in Neural Information Processing Systems 32.

[8] Dao, T., Fu, D., Ermon, S., Rudra, A., Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. Advances in Neural Information Processing Systems 35.

[9] Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. arXiv:2307.08691.

[10] Liu, H., Zaharia, M., Abbeel, P. (2023). Ring Attention with Blockwise Transformers for Near-Infinite Context. arXiv:2310.01889.

[11] Katharopoulos, A., Vyas, A., Pappas, N., Fleuret, F. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML 2020.

[12] Gu, A., Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752.

[13] Shazeer, N., Mirhoseini, A., Maziarz, K., Davis, A., Le, Q., Hinton, G., Dean, J. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR 2017.

[14] Fedus, W., Zoph, B., Shazeer, N. (2021). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. arXiv:2101.03961.

[15] Jang, E., Gu, S., Poole, B. (2017). Categorical Reparameterization with Gumbel-Softmax. ICLR 2017.

[16] Kingma, D.P., Ba, J. (2015). Adam: A Method for Stochastic Optimization. ICLR 2015.

[17] Loshchilov, I., Hutter, F. (2019). Decoupled Weight Decay Regularization (AdamW). ICLR 2019.

[18] Shazeer, N., Stern, M. (2018). Adafactor: Adaptive Learning Rates with Sublinear Memory Cost. ICML 2018.

[19] Martens, J., Grosse, R. (2015). Optimizing Neural Networks with Kronecker-Factored Approximate Curvature. ICML 2015.

[20] Amari, S. (1998). Natural Gradient Works Efficiently in Learning. Neural Computation 10(2).

[21] Boumal, N. (2023). An Introduction to Optimization on Smooth Manifolds. Cambridge University Press.

[22] Cayley, A. (1846). Sur quelques propriétés des déterminants gauches. Journal für die reine und angewandte Mathematik, 32, 119-123.

[23] Hall, B.C. (2015). Lie Groups, Lie Algebras, and Representations. GTM 222, Springer.

[24] LeCun, Y. (2022). A Path Towards Autonomous Machine Intelligence. Technical Report, Meta AI.

[25] Bardes, A., Ponce, J., LeCun, Y. (2022). VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. ICLR 2022.

[26] Zbontar, J., Jing, L., Misra, I., LeCun, Y., Deny, S. (2021). Barlow Twins: Self-Supervised Learning via Redundancy Reduction. ICML 2021.

[27] Miyato, T., Kataoka, T., Koyama, M., Yoshida, Y. (2018). Spectral Normalization for Generative Adversarial Networks. ICLR 2018.

[28] Leshno, M., Lin, V.Y., Pinkus, A., Schocken, S. (1993). Multilayer Feedforward Networks with a Nonpolynomial Activation Function Can Approximate Any Function. Neural Networks, 6(6), 861-867.

[29] Hornik, K. (1991). Approximation Capabilities of Multilayer Feedforward Networks. Neural Networks, 4(2), 251-257.

[30] Schlag, I., Irie, K., Schmidhuber, J. (2021). Linear Transformers Are Secretly Fast Weight Programmers. ICML 2021.

[31] Blelloch, G.E. (1990). Prefix Sums and Their Applications. Technical Report CMU-CS-90-190, CMU.

[32] Arjovsky, M., Shah, A., Bengio, Y. (2016). Unitary Evolution Recurrent Neural Networks. ICML 2016.

[33] Helfrich, K., Willmott, D., Ye, Q. (2018). Orthogonal Recurrent Neural Networks with Scaled Cayley Transform. ICML 2018.

[34] Widrow, B., Hoff, M.E. (1960). Adaptive Switching Circuits. IRE WESCON Convention Record.

[35] Higham, N.J. (2002). Accuracy and Stability of Numerical Algorithms (2nd ed.). SIAM.

[36] NVIDIA Corporation. (2024). CUDA C++ Programming Guide. Technical reference, v12.x.

[37] Goodfellow, I., Bengio, Y., Courville, A. (2016). Deep Learning. MIT Press.

[38] Lee, J.M. (2018). Introduction to Riemannian Manifolds, 2nd ed. Springer.

[39] Nesterov, Y. (2018). Lectures on Convex Optimization, 2nd ed. Springer.

[40] Jin, C., Ge, R., Netrapalli, P., Kakade, S.M., Jordan, M.I. (2017). How to Escape Saddle Points Efficiently. ICML 2017.

[41] Bronstein, M.M., Bruna, J., Cohen, T., Velickovic, P. (2021). Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges. arXiv:2104.13478.

[42] Milakov, M., Gimelshein, N. (2018). Online Normalizer Calculation for Softmax. arXiv:1805.02867.

[43] Yang, S., Wang, B., Shen, Y., Panda, R., Kim, Y. (2024). Gated Linear Attention Transformers. arXiv:2312.06635.

[44] DeepSeek-AI. (2024). DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model. arXiv:2405.04434.

[45] DeepSeek-AI. (2024). DeepSeek-V3 Technical Report. arXiv:2412.19437.
