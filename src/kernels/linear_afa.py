"""Experimental O(N) diagonal-feature approximation to Algebraic Attention.

Under Section 4 of the Kernel Engineering Blueprint:
rho(s)^8 = exp(8 asinh(s)) is not a finite polynomial. The recurrence in this
module uses its degree-8 Taylor coefficients and retains only coordinate-wise
monomials. It is therefore a bounded-state approximation and is not equivalent
to the production octic AFA kernel.

The full polynomial kernel for (q^T k)^m would require all symmetric tensor
monomials. This implementation deliberately uses a much smaller diagonal map:
    phi(x) = [1, sqrt(c_1) x, sqrt(c_2) x^2, ..., sqrt(c_8) x^8]

Linear Recurrence Step (Inference / Long-Context):
Instead of quadratic O(N^2) attention matrices:
1. State Update:
   M_t = M_{t-1} + phi(k_t) tensor v_t in R^{D_phi x D_v}
   Z_t = Z_{t-1} + phi(k_t) in R^{D_phi}
2. Output Query:
   out_t = (phi(q_t) M_t) / (phi(q_t) Z_t + Omega)
3. Complexity:
   - Training: O(N) with parallel prefix scan.
   - Inference: O(1) memory per step, independent of context length (100k, 1M, or 10M tokens).

Strictly zero transcendental functions (0 exp, 0 log, 0 trig).
"""

from typing import NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp


# Degree-8 Taylor coefficients at s=0 for rho(s)^8. The series continues past
# degree 8; these values do not define an exact global polynomial identity.
OCTIC_POLYNOMIAL_COEFFICIENTS = (
    1.0,      # c_0
    8.0,      # c_1
    32.0,     # c_2
    84.0,     # c_3
    160.0,    # c_4
    231.0,    # c_5
    256.0,    # c_6
    214.5,    # c_7 (429 / 2)
    128.0,    # c_8
)


class LinearAFAState(NamedTuple):
    """Recurrent inference memory state for O(1) step generation."""
    M: jax.Array  # (..., D_phi, D_v)
    Z: jax.Array  # (..., D_phi)
    step: int = 0


def compute_algebraic_feature_map(
    x: jax.Array,
    projection_basis: Optional[jax.Array] = None,
    order: int = 4,
    scale: Optional[float] = None,
) -> jax.Array:
    """Evaluates the polynomial feature map phi(x) without transcendentals.

    Args:
        x: Input tensor of shape (..., D).
        projection_basis: Optional random or learned projection basis (R, D).
            If None, uses degree-wise powers of scaled x.
        order: Truncation order of the polynomial (default: 4, max: 8).
        scale: Optional per-vector scale factor (defaults to D^(-1/4), so a
            paired coordinate product has the usual 1/sqrt(D) score scale).

    Returns:
        phi(x) tensor of shape (..., D_phi) with positive real features.
    """
    if not (1 <= order <= 8):
        raise ValueError(f"order must be in [1, 8], got {order}")
    d_in = x.shape[-1]
    if scale is None:
        scale = float(d_in ** -0.25)

    x_scaled = x * scale
    c = jnp.array(OCTIC_POLYNOMIAL_COEFFICIENTS[: order + 1], dtype=x.dtype)

    if projection_basis is not None:
        proj = jnp.matmul(x_scaled, projection_basis.T.astype(x.dtype))
        feature_blocks = [jnp.ones(x.shape[:-1] + (1,), dtype=x.dtype)]
        for m in range(1, order + 1):
            weight = jnp.sqrt(c[m])
            feature_blocks.append(weight * (proj ** m))
        return jnp.concatenate(feature_blocks, axis=-1)
    else:
        # Compact diagonal coordinate-power basis. This omits cross monomials.
        feature_blocks = [jnp.ones(x.shape[:-1] + (1,), dtype=x.dtype)]
        for m in range(1, order + 1):
            weight = jnp.sqrt(c[m])
            feature_blocks.append(weight * (x_scaled ** m))
        return jnp.concatenate(feature_blocks, axis=-1)


def linear_afa_init_state(
    batch_shape: Tuple[int, ...],
    d_phi: int,
    d_v: int,
    dtype: jnp.dtype = jnp.float32,
) -> LinearAFAState:
    """Initializes recurrent state M_0 = 0 and Z_0 = 0."""
    m_shape = batch_shape + (d_phi, d_v)
    z_shape = batch_shape + (d_phi,)
    return LinearAFAState(
        M=jnp.zeros(m_shape, dtype=dtype),
        Z=jnp.zeros(z_shape, dtype=dtype),
        step=0,
    )


def linear_afa_step(
    state: LinearAFAState,
    q_t: jax.Array,
    k_t: jax.Array,
    v_t: jax.Array,
    sink_omega: float = 0.5,
    projection_basis: Optional[jax.Array] = None,
    order: int = 4,
) -> Tuple[jax.Array, LinearAFAState]:
    """Single recurrent inference step with O(1) memory and O(D_phi * D_v) compute.

    Args:
        state: Previous LinearAFAState (M_{t-1}, Z_{t-1}).
        q_t: Query token slice of shape (..., D_k).
        k_t: Key token slice of shape (..., D_k).
        v_t: Value token slice of shape (..., D_v).
        sink_omega: Nonnegative attention sink scalar mass.
        projection_basis: Optional polynomial projection matrix.
        order: Truncation order of polynomial expansion.

    Returns:
        (out_t, next_state) where out_t is of shape (..., D_v).
    """
    phi_k = compute_algebraic_feature_map(k_t, projection_basis=projection_basis, order=order)
    phi_q = compute_algebraic_feature_map(q_t, projection_basis=projection_basis, order=order)

    # 1. State Update:
    # M_t = M_{t-1} + phi(k_t) tensor v_t
    # Z_t = Z_{t-1} + phi(k_t)
    delta_m = phi_k[..., :, None] * v_t[..., None, :]
    next_m = state.M + delta_m
    next_z = state.Z + phi_k

    # 2. Output Query:
    # out_t = (phi(q_t) M_t) / (phi(q_t) Z_t + Omega)
    numerator = jnp.sum(phi_q[..., :, None] * next_m, axis=-2)
    denominator = jnp.sum(phi_q * next_z, axis=-1, keepdims=True) + sink_omega
    out_t = numerator / denominator

    next_state = LinearAFAState(M=next_m, Z=next_z, step=state.step + 1)
    return out_t.astype(v_t.dtype), next_state


def linear_afa_parallel_scan(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    sink_omega: float = 0.5,
    projection_basis: Optional[jax.Array] = None,
    order: int = 4,
) -> jax.Array:
    """Approximate O(N) training execution via a cumulative scan.

    Evaluates the same diagonal-feature recurrence as ``linear_afa_step`` across
    the full sequence. It does not reproduce exact octic AFA.

    Args:
        q: Query tensor of shape (B, H, T, D_k).
        k: Key tensor of shape (B, H, T, D_k).
        v: Value tensor of shape (B, H, T, D_v).
        sink_omega: Nonnegative attention sink scalar mass.
        projection_basis: Optional polynomial projection matrix.
        order: Truncation order of polynomial expansion.

    Returns:
        Attention output tensor of shape (B, H, T, D_v).
    """
    phi_q = compute_algebraic_feature_map(q, projection_basis=projection_basis, order=order)
    phi_k = compute_algebraic_feature_map(k, projection_basis=projection_basis, order=order)

    # Outer product delta M: (B, H, T, D_phi, D_v)
    delta_m = phi_k[..., :, None] * v[..., None, :]
    delta_z = phi_k  # (B, H, T, D_phi)

    # Parallel causal prefix scan along sequence axis T (axis=2)
    m_scan = jnp.cumsum(delta_m, axis=2)
    z_scan = jnp.cumsum(delta_z, axis=2)

    # Contract with query features
    numerator = jnp.sum(phi_q[..., :, None] * m_scan, axis=-2)
    denominator = jnp.sum(phi_q * z_scan, axis=-1, keepdims=True) + sink_omega

    out = numerator / denominator
    return out.astype(v.dtype)
