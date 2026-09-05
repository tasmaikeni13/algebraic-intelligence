import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp

namespace AlgebraicTheory

theorem pearson_chi_sq_expansion (y p : ℝ) (hp : p ≠ 0) :
    (y - p)^2 / p = y^2 / p - 2 * y + p := by
  field_simp [hp]
  ring

theorem pearson_divergence_expansion (y p : ℝ) (hp : p ≠ 0) :
    (y - p)^2 / p = y^2 / p - 2 * y + p :=
  pearson_chi_sq_expansion y p hp

theorem pearson_divergence_nonneg (y p : ℝ) (hp : 0 < p) :
    0 ≤ (y - p)^2 / p :=
  div_nonneg (sq_nonneg (y - p)) (le_of_lt hp)

theorem pearson_zero_iff_equal (y p : ℝ) (hp : 0 < p) :
    (y - p)^2 / p = 0 ↔ y = p := by
  have hp_ne : p ≠ 0 := ne_of_gt hp
  rw [div_eq_zero_iff]
  simp [hp_ne, sub_eq_zero]

theorem oace_power_chain (z : ℝ) :
    (((z^2)^2)^2) = z^8 := by
  ring

end AlgebraicTheory
