# Phase 3 Dependency Audit: Algebraic Geometric Oscillators (AGO)

Phases 1, 2, and 3 are fully implemented, formally certified, and verified on the 16-chip TPU v4 slice.

| Consumer | Repository State | Contract Carried Forward |
| :--- | :--- | :--- |
| **Phase 4 `src/embeddings.py`** | Planned | Algebraic embeddings must map discrete vocabulary tokens to rational manifold coordinates compatible with Cayley rotations. |
| **Phase 6 `src/kernels/pallas_afa.py`** | Planned | Fused attention kernels evaluate Cayley rotations via 4 FMAs per channel pair inline before octic polynomial scoring in VMEM. |
| **Phases 7–9 `src/model.py`** | Planned | Attention query and key projections are rotated via `apply_ago_rotations(q, k, rotary_params)` prior to score computation in `algebraic_softmax`. Input norms are invariant under rotation ($\|\mathbf{R}\mathbf{x}\| = \|\mathbf{x}\|$). |

---

### Contract Guarantees
1. **Zero-Transcendental Axiom:** Rotation tables are generated via rational recurrence using square radical `rsqrt` without trigonometric (`sin`, `cos`) or non-integer power calls.
2. **Exact Lie Group Closure:** Rotations are strictly orthogonal ($\mathbf{c}_1 \cdot \mathbf{c}_2 = 0$) and unimodular ($\det(\mathbf{R}) = 1.0$), ensuring norm conservation without channel explosion or decay.
3. **Shift Equivariance:** Relative query-key position encoding satisfies $\langle \mathbf{R}(m)\mathbf{q}, \mathbf{R}(n)\mathbf{k} \rangle = \mathbf{q}^\top \mathbf{R}(n - m) \mathbf{k}$ over arbitrarily long contexts.
4. **Public Interface:**
   - `build_cayley_rotary_matrix(dim, max_seq_len, freqs=None, base=10000.0, dtype=jnp.float32)`
   - `apply_ago_rotations(q, k, rotary_params=None, seq_axis=None)`
