"""Hardware-fused kernels for algebraic intelligence."""

from src.kernels.pallas_afa import (
    afa_kernel,
    pallas_afa_forward,
    tiled_afa_forward,
    tiled_afa_backward,
    exact_afa_reference,
    exact_afa_with_denominator,
    pallas_afa,
    distributed_ring_afa,
    sharded_pallas_afa,
    algebraic_flash_attention,
)
from src.kernels.pallas_oace import (
    fused_linear_oace_forward,
    fused_linear_oace_backward,
    fused_linear_oace,
)
from src.kernels.linear_afa import (
    OCTIC_POLYNOMIAL_COEFFICIENTS,
    LinearAFAState,
    compute_algebraic_feature_map,
    linear_afa_init_state,
    linear_afa_step,
    linear_afa_parallel_scan,
)
from src.kernels.triton_afa import (
    _triton_afa_fwd_kernel,
    _triton_afa_bwd_kernel,
    triton_algebraic_flash_attention,
)
from src.kernels.triton_oace import (
    _triton_linear_oace_fwd_kernel,
    triton_fused_linear_oace,
)

__all__ = [
    "afa_kernel",
    "pallas_afa_forward",
    "tiled_afa_forward",
    "tiled_afa_backward",
    "exact_afa_reference",
    "exact_afa_with_denominator",
    "pallas_afa",
    "distributed_ring_afa",
    "sharded_pallas_afa",
    "algebraic_flash_attention",
    "fused_linear_oace_forward",
    "fused_linear_oace_backward",
    "fused_linear_oace",
    "OCTIC_POLYNOMIAL_COEFFICIENTS",
    "LinearAFAState",
    "compute_algebraic_feature_map",
    "linear_afa_init_state",
    "linear_afa_step",
    "linear_afa_parallel_scan",
    "_triton_afa_fwd_kernel",
    "_triton_afa_bwd_kernel",
    "triton_algebraic_flash_attention",
    "_triton_linear_oace_fwd_kernel",
    "triton_fused_linear_oace",
]
