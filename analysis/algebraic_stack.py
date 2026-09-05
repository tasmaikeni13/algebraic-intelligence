# -*- coding: utf-8 -*-
"""
The Algebraic Stack: PyTorch Reference Implementation
Every operation is rational or radical (rsqrt). Zero transcendentals.
Author: Tasmai Keni (tas.ken.rt25@dypatil.edu)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# Section 2 & 3: Foundational Primitives (Algebraic Gate, Kernel, ALU)
# ============================================================================

def algebraic_gate(x: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """Algebraic Gate: beta(x; kappa) = 0.5 * (1 + x / sqrt(x^2 + kappa))"""
    denom = torch.rsqrt(x.pow(2) + kappa)
    return 0.5 * (1.0 + x * denom)

def algebraic_kernel(x: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """Algebraic Kernel: rho(x) = x + sqrt(x^2 + kappa)"""
    return x + torch.sqrt(x.pow(2) + kappa)

class ALUFunction(torch.autograd.Function):
    """
    Algebraic Linear Unit with exact polynomial backward pass:
    Forward: K(x) = 0.5 * x * (1 + u), where u = x / sqrt(x^2 + 1)
    Backward: K'(x) = 0.5 * (1 + 2*u - u^3)
    Zero rsqrt or divisions in the backward pass!
    """
    @staticmethod
    def forward(ctx, x):
        inv_sqrt = torch.rsqrt(x.pow(2) + 1.0)
        u = x * inv_sqrt
        ctx.save_for_backward(u)
        return 0.5 * x * (1.0 + u)

    @staticmethod
    def backward(ctx, grad_output):
        u, = ctx.saved_tensors
        # Polynomial backward pass: 0.5 * (1 + 2u - u^3)
        grad_x = 0.5 * (1.0 + 2.0 * u - u.pow(3))
        return grad_output * grad_x

class ALU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ALUFunction.apply(x)

# ============================================================================
# Section 6: Algebraic Variance Normalization (AVN)
# ============================================================================

class AVN(nn.Module):
    """
    Algebraic Variance Normalization:
    tau = rsqrt(m2(x) + eps), x_hat = tau * x
    Zero learnable parameters in HBM!
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor):
        # x: (..., d)
        d = x.shape[-1]
        m2 = torch.sum(x.pow(2), dim=-1, keepdim=True) / d
        tau = torch.rsqrt(m2 + self.eps)
        x_hat = x * tau
        return x_hat, tau

# ============================================================================
# Section 4: Algebraic Softmax (A-Softmax)
# ============================================================================

class AlgebraicSoftmax(nn.Module):
    """
    AVN-bounded A-Softmax with power sharpening n=8 (computed via 3 squarings):
    S_8(s)_i = rho(s_hat_i)^8 / (sum_j rho(s_hat_j)^8 + Omega)
    where Omega >= 0 is a constant algebraic attention sink.
    """
    def __init__(self, eps: float = 1e-6, omega: float = 0.0):
        super().__init__()
        self.eps = eps
        self.omega = omega

    def forward(self, s: torch.Tensor, dim: int = -1, omega: float = None) -> torch.Tensor:
        # Step 1: AVN logit bounding along dim
        d = s.shape[dim]
        m2 = torch.sum(s.pow(2), dim=dim, keepdim=True) / d
        tau = torch.rsqrt(m2 + self.eps)
        s_hat = s * tau

        # Step 2: Algebraic Kernel rho(s_hat) = s_hat + sqrt(s_hat^2 + 1)
        rho = s_hat + torch.sqrt(s_hat.pow(2) + 1.0)

        # Step 3: Power of 8 via 3 squarings
        rho2 = rho * rho
        rho4 = rho2 * rho2
        rho8 = rho4 * rho4

        # Step 4: Normalization (denominator accumulated in float32 with attention sink)
        omega_val = self.omega if omega is None else omega
        denom = torch.sum(rho8, dim=dim, keepdim=True) + omega_val
        return rho8 / (denom + 1e-12)

# ============================================================================
# Section 4 & 5: Losses and Divergences (OACE and AD)
# ============================================================================

class OctoAlgebraicCrossEntropy(nn.Module):
    """
    Octo-Algebraic Cross-Entropy (OACE):
    L_{1/8}(p, k) = 8 * (p_k^{-1/8} - 1)
    Computed via 3 sequential rsqrt steps.
    """
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, p: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # p: (batch_size, num_classes), target: (batch_size,)
        p_k = p.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        p_k = torch.clamp(p_k, min=self.eps, max=1.0)
        # 3 rsqrt evaluations: p^(-1/2) -> p^(-1/4) -> p^(-1/8)
        z1 = torch.rsqrt(p_k)            # p^(-1/2)
        z2 = torch.rsqrt(1.0 / z1)       # p^(-1/4)
        z3 = torch.rsqrt(1.0 / z2)       # p^(-1/8)
        loss = 8.0 * (z3 - 1.0)
        return loss.mean()

class AlgebraicDivergence(nn.Module):
    """
    Algebraic Divergence (Pearson chi^2 divergence):
    D_A(y || p) = sum_i y_i^2 / p_i - 1
    Strictly proper on simplex, Fisher equivalent to KL.
    """
    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, y: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        p = torch.clamp(p, min=self.eps)
        div = torch.sum(y.pow(2) / p, dim=-1) - 1.0
        return div.mean()

# ============================================================================
# Section 7: Algebraic Geometric Ordering (AGO) via Cayley Transform
# ============================================================================

class AlgebraicGeometricOrdering(nn.Module):
    """
    Algebraic Geometric Ordering:
    Static Cayley rotation: R_k = (I + omega_k J)(I - omega_k J)^(-1)
    = 1/(1 + omega_k^2) * [[1 - omega_k^2, -2 omega_k], [2 omega_k, 1 - omega_k^2]]
    Shift-equivariant relative attention with ZERO trigonometry!
    """
    def __init__(self, dim: int, max_seq_len: int = 4096):
        super().__init__()
        assert dim % 2 == 0, "dim must be even for 2D rotations"
        self.dim = dim
        self.num_pairs = dim // 2
        self.max_seq_len = max_seq_len

        # Base frequencies omega_k: algebraic geometric sequence
        # omega_k = (1 / 10000)^(2k / dim)
        indices = torch.arange(self.num_pairs, dtype=torch.float64)
        omega = torch.pow(10000.0, -2.0 * indices / dim)

        w2 = omega.pow(2)
        denom = 1.0 / (1.0 + w2)
        c = (1.0 - w2) * denom
        s = 2.0 * omega * denom
        self.register_buffer("base_c", c.to(torch.float32))
        self.register_buffer("base_s", s.to(torch.float32))

        # Precompute position powers up to max_seq_len via repeated Cayley rotation in float64
        c_list = [torch.ones_like(c)]
        s_list = [torch.zeros_like(s)]
        cur_c = torch.ones_like(c)
        cur_s = torch.zeros_like(s)
        for _ in range(1, max_seq_len):
            next_c = cur_c * c - cur_s * s
            next_s = cur_c * s + cur_s * c
            # Algebraic re-normalization to prevent cumulative numerical drift (Phase 3 playbook)
            inv_norm = torch.rsqrt(next_c.pow(2) + next_s.pow(2))
            next_c = next_c * inv_norm
            next_s = next_s * inv_norm
            c_list.append(next_c)
            s_list.append(next_s)
            cur_c = next_c
            cur_s = next_s

        self.register_buffer("pos_c", torch.stack(c_list, dim=0).to(torch.float32))
        self.register_buffer("pos_s", torch.stack(s_list, dim=0).to(torch.float32))

    def forward(self, x: torch.Tensor, seq_offset: int = 0) -> torch.Tensor:
        # x: (batch, seq_len, num_heads, head_dim)
        seq_len = x.shape[1]
        c = self.pos_c[seq_offset:seq_offset + seq_len].unsqueeze(0).unsqueeze(2) # (1, seq, 1, num_pairs)
        s = self.pos_s[seq_offset:seq_offset + seq_len].unsqueeze(0).unsqueeze(2)

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        rot_x1 = x1 * c - x2 * s
        rot_x2 = x1 * s + x2 * c

        out = torch.empty_like(x)
        out[..., 0::2] = rot_x1
        out[..., 1::2] = rot_x2
        return out

# ============================================================================
# Section 8 & 9: Algebraic Attention & Algebraic Flash Attention
# ============================================================================

class AlgebraicAttention(nn.Module):
    """
    Algebraic Multi-Head Attention:
    Scores normalized via A-Softmax (rho^8 kernel), additive tile accumulation.
    """
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        assert self.head_dim % 2 == 0

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.ago = AlgebraicGeometricOrdering(self.head_dim, max_seq_len=max_seq_len)
        self.a_softmax = AlgebraicSoftmax()

    def forward(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        B, L, D = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim)

        # Apply AGO positional rotation
        q = self.ago(q)
        k = self.ago(k)

        # Transpose to (B, num_heads, L, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        if is_causal:
            mask = torch.triu(torch.ones(L, L, device=x.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, -100.0)

        attn_weights = self.a_softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v) # (B, num_heads, L, head_dim)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.o_proj(out)

# ============================================================================
# Section 10: ALU-GLU Feed-Forward Block
# ============================================================================

class ALUGLU(nn.Module):
    """
    ALU-GLU: W_d [(W_g x) * K(W_u x)]
    SwiGLU-compatible pure algebraic feed-forward block.
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_g = nn.Linear(d_model, d_ff, bias=False)
        self.w_u = nn.Linear(d_model, d_ff, bias=False)
        self.alu = ALU()
        self.w_d = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w_g(x)
        up = self.w_u(x)
        return self.w_d(gate * self.alu(up))

# ============================================================================
# Section 11: Algebraic Mixture of Experts (A-MoE)
# ============================================================================

def algebraic_noise_transform(shape, device=None, eps: float = 1e-4) -> torch.Tensor:
    """ANT: Inverse-CDF sampling of algebraic distribution from Uniform(0,1)"""
    u = torch.rand(shape, device=device)
    v = 2.0 * u - 1.0
    denom = torch.rsqrt(1.0 - v.pow(2) + eps)
    return v * denom

class AlgebraicMoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([ALUGLU(d_model, d_ff) for _ in range(num_experts)])
        self.a_softmax = AlgebraicSoftmax()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        x_flat = x.view(-1, D)
        logits = self.router(x_flat) # (N, E)

        if self.training:
            noise = algebraic_noise_transform(logits.shape, device=logits.device)
            logits = logits + 0.1 * noise

        probs = self.a_softmax(logits, dim=-1) # (N, E)
        topk_probs, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-12)

        out = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (topk_idx == i)
            if mask.any():
                token_indices, k_slots = torch.where(mask)
                expert_inputs = x_flat[token_indices]
                expert_outputs = expert(expert_inputs)
                weights = topk_probs[token_indices, k_slots].unsqueeze(-1)
                out.index_add_(0, token_indices, expert_outputs * weights)

        return out.view(B, L, D)

# ============================================================================
# Section 12: Algebraic Curvature Optimizer (ACO)
# ============================================================================

class AlgebraicCurvatureOptimizer(torch.optim.Optimizer):
    """
    Algebraic Curvature Optimizer (ACO):
    - Factorized O(d_out + d_in) curvature preconditioning
    - Rational moment accumulation
    - Algebraic Rational Decay Schedule (ARDS) support
    - Zero transcendentals!
    """
    def __init__(self, params, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8, weight_decay: float = 0.01):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, weight_decay=weight_decay, step=0)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1 = group['beta1']
            beta2 = group['beta2']
            lr = group['lr']
            eps = group['eps']
            wd = group['weight_decay']
            group['step'] += 1
            step = group['step']

            bias_corr1 = 1.0 - (beta1 ** step)
            bias_corr2 = 1.0 - (beta2 ** step)

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['m_avg'] = torch.zeros_like(p)
                    if p.dim() >= 2:
                        state['row_avg'] = torch.zeros(p.shape[0], device=p.device, dtype=p.dtype)
                        state['col_avg'] = torch.zeros(p.shape[1], device=p.device, dtype=p.dtype)
                    else:
                        state['v_avg'] = torch.zeros_like(p)

                m_avg = state['m_avg']
                m_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                debiased_m = m_avg / bias_corr1

                if p.dim() >= 2:
                    row_avg = state['row_avg']
                    col_avg = state['col_avg']

                    grad_sq = grad.pow(2)
                    mean_row = grad_sq.mean(dim=1)
                    mean_col = grad_sq.mean(dim=0)

                    row_avg.mul_(beta2).add_(mean_row, alpha=1.0 - beta2)
                    col_avg.mul_(beta2).add_(mean_col, alpha=1.0 - beta2)

                    debiased_r = row_avg / bias_corr2
                    debiased_c = col_avg / bias_corr2

                    r_bar = debiased_r.mean()
                    scale = torch.sqrt(r_bar + eps)
                    r_col = debiased_r.unsqueeze(1)
                    c_row = debiased_c.unsqueeze(0)
                    inv_curv = scale * torch.rsqrt(r_col * c_row + (eps * eps))
                    update = debiased_m * inv_curv
                else:
                    v_avg = state['v_avg']
                    v_avg.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    debiased_v = v_avg / bias_corr2
                    update = debiased_m * torch.rsqrt(debiased_v + (eps * eps))

                if wd > 0:
                    p.mul_(1.0 - lr * wd)

                p.add_(update, alpha=-lr)

        return loss

def ards_learning_rate(step: int, lr_max: float, t_warmup: int = 1000, t_decay: int = 10000, alpha: float = 1.0) -> float:
    """Algebraic Rational Decay Schedule: lr(t) = lr_max * warmup * rsqrt(1 + alpha * ((t - warmup)/decay)^2)"""
    if step <= t_warmup:
        return lr_max * (float(step) / float(max(1, t_warmup)))
    decay_steps = float(step - t_warmup) / float(t_decay)
    return lr_max * (1.0 / math.sqrt(1.0 + alpha * (decay_steps ** 2)))

# ============================================================================
# Complete Algebraic Transformer
# ============================================================================

class AlgebraicTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int = 4096):
        super().__init__()
        self.avn1 = AVN()
        self.attn = AlgebraicAttention(d_model, num_heads, max_seq_len=max_seq_len)
        self.avn2 = AVN()
        self.ffn = ALUGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x, _ = self.avn1(x)
        x = x + self.attn(norm_x)
        norm_x2, _ = self.avn2(x)
        x = x + self.ffn(norm_x2)
        return x

class AlgebraicTransformerLM(nn.Module):
    """
    A Complete 100% Pure Algebraic Transformer Language Model.
    Contains zero transcendental functions in embedding, attention, normalization, feed-forward, or heads.
    """
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, d_ff: int, max_seq_len: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([
            AlgebraicTransformerBlock(d_model, num_heads, d_ff, max_seq_len=max_seq_len)
            for _ in range(num_layers)
        ])
        self.final_avn = AVN()
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.a_softmax = AlgebraicSoftmax()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.token_emb(idx)
        for block in self.blocks:
            x = block(x)
        norm_x, _ = self.final_avn(x)
        logits = self.lm_head(norm_x)
        return logits

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = self.a_softmax(logits, dim=-1)
        oace = OctoAlgebraicCrossEntropy()
        return oace(probs.view(-1, probs.shape[-1]), targets.view(-1))
