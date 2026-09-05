import Mathlib.Basic.Real.Basic
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp

namespace AlgebraicTheory

theorem cayley_pythagorean_identity (w : ℝ) :
    (1 - w^2)^2 + (2 * w)^2 = (1 + w^2)^2 := by
  ring

theorem cayley_column_norm_one (w : ℝ) (hw : 1 + w^2 ≠ 0) :
    ((1 - w^2) / (1 + w^2))^2 + ((2 * w) / (1 + w^2))^2 = 1 := by
  field_simp [hw]
  ring

theorem cayley_columns_orthogonal (w : ℝ) :
    (1 - w^2) * (-2 * w) + (2 * w) * (1 - w^2) = 0 := by
  ring

theorem cayley_determinant_one (w : ℝ) (hw : 1 + w^2 ≠ 0) :
    ((1 - w^2) * (1 - w^2) - (-2 * w) * (2 * w)) / (1 + w^2)^2 = 1 := by
  field_simp [hw]
  ring

theorem cayley_composition_norm (c1 s1 c2 s2 : ℝ)
    (h1 : c1^2 + s1^2 = 1) (h2 : c2^2 + s2^2 = 1) :
    (c1 * c2 - s1 * s2)^2 + (c1 * s2 + s1 * c2)^2 = 1 := by
  calc
    (c1 * c2 - s1 * s2)^2 + (c1 * s2 + s1 * c2)^2 =
      (c1^2 + s1^2) * (c2^2 + s2^2) := by ring
    _ = 1 * 1 := by rw [h1, h2]
    _ = 1 := by ring

end AlgebraicTheory
