# Phase 2 Dependency Audit: A-Softmax

Phase 1 and Phase 2 are fully implemented and verified on the 16-chip TPU v4 slice.

| Consumer | Repository State | Contract Carried Forward |
| :--- | :--- | :--- |
| **Phase 3 `src/position.py`** | Specifications only | Rotated keys and queries via AGO Cayley transform preserve the Euclidean norm ($\|\mathbf{R}_k \mathbf{x}\| = \|\mathbf{x}\|$), ensuring invariant input scaling into AVN and A-Softmax. |
| **Phase 6 `src/kernels/pallas_afa.py`** | Specifications only | Tile accumulation is purely additive without requiring online maximum subtraction $\max(s)$. Intermediates use FP32 accumulation; output matches input dtype (FP32/BF16). 3-squaring circuit depth must be evaluated in VMEM without register spilling. |
| **Phases 7–9 `src/model.py`** | Specifications only | Attention head outputs must use static rational sink $\Omega = 0.5$, default AVN $\epsilon = 10^{-5}$, and last-axis normalization. Backward pass must include the full AVN quotient-rule cotangent. |

---

### Contract Guarantees
1. **Public Interface:** `algebraic_softmax(scores, sink_omega=0.5, eps=1e-5)` in [`src/attention.py`](../../src/attention.py) retains exact static arguments and shapes.
2. **2-Lipschitz Invariant:** Upstream gradient propagation through attention is guaranteed to have bounded entrywise amplification $\le 2.0$.
3. **Purity Contract:** Zero transcendental operations (`exp`, `log`, `softmax`) permitted in forward or backward graphs.
