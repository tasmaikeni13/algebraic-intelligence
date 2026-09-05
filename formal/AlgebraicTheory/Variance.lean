import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp

namespace AlgebraicTheory

theorem avn_bounded_norm (S d eps : ℝ) (_hS : 0 ≤ S) (hd : 0 < d) (heps : 0 ≤ eps) :
    S ≤ d * (S / d + eps) := by
  have hd_ne : d ≠ 0 := ne_of_gt hd
  have h1 : d * (S / d + eps) = S + d * eps := by
    calc
      d * (S / d + eps) = d * (S / d) + d * eps := by ring
      _ = S + d * eps := by rw [mul_div_cancel₀ S hd_ne]
  rw [h1]
  have h2 : 0 ≤ d * eps := mul_nonneg (le_of_lt hd) heps
  linarith

theorem avn_coupling_identity (x v tau : ℝ) (htau : tau^2 * v = 1) :
    (tau * x)^2 + 1 = tau^2 * (x^2 + v) := by
  calc
    (tau * x)^2 + 1 = tau^2 * x^2 + 1 := by ring
    _ = tau^2 * x^2 + tau^2 * v := by rw [htau.symm]
    _ = tau^2 * (x^2 + v) := by ring

end AlgebraicTheory
