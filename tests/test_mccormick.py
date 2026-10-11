"""Independent tests for the exact McCormick product semantics."""

import unittest

from llm_resource_allocation.math.mccormick import (
    compute_binary_continuous_product,
    compute_mccormick_residuals,
    satisfies_mccormick_envelope,
)


REPRESENTATIVE_X_VALUES = (0.0, 0.1, 0.5, 0.999999, 1.0)


class McCormickExactnessTests(unittest.TestCase):
    """Translate the binary-case exactness proof into deterministic tests."""

    def test_w_zero_forces_unique_zero_product(self) -> None:
        """Show that ``v <= w=0`` and ``v >= 0`` force ``v=0``."""

        for x in REPRESENTATIVE_X_VALUES:
            with self.subTest(x=x):
                expected = compute_binary_continuous_product(x, 0)
                self.assertEqual(0.0, expected)
                self.assertTrue(satisfies_mccormick_envelope(x, 0, expected))
                self.assertFalse(satisfies_mccormick_envelope(x, 0, 0.001))
                self.assertFalse(satisfies_mccormick_envelope(x, 0, -0.001))

    def test_w_one_forces_unique_x_product(self) -> None:
        """Show that ``v <= x`` and ``v >= x`` force ``v=x``."""

        for x in REPRESENTATIVE_X_VALUES:
            with self.subTest(x=x):
                expected = compute_binary_continuous_product(x, 1)
                self.assertEqual(x, expected)
                self.assertTrue(satisfies_mccormick_envelope(x, 1, expected))
                self.assertFalse(satisfies_mccormick_envelope(x, 1, x + 0.001))
                self.assertFalse(satisfies_mccormick_envelope(x, 1, x - 0.001))

    def test_all_four_binary_boundaries(self) -> None:
        """Check the products at every corner of the continuous-binary domain."""

        expected_products = {
            (0.0, 0): 0.0,
            (0.0, 1): 0.0,
            (1.0, 0): 0.0,
            (1.0, 1): 1.0,
        }
        for (x, w), expected in expected_products.items():
            with self.subTest(x=x, w=w):
                actual = compute_binary_continuous_product(x, w)
                self.assertEqual(expected, actual)
                self.assertTrue(satisfies_mccormick_envelope(x, w, actual))

    def test_exact_product_satisfies_every_residual_on_grid(self) -> None:
        """Evaluate both binary states on a deterministic 101-point x grid."""

        for grid_index in range(101):
            x = grid_index / 100.0
            for w in (0, 1):
                with self.subTest(x=x, w=w):
                    v = compute_binary_continuous_product(x, w)
                    residuals = compute_mccormick_residuals(x, w, v)
                    self.assertGreaterEqual(residuals.x_minus_v, 0.0)
                    self.assertGreaterEqual(residuals.w_minus_v, 0.0)
                    self.assertGreaterEqual(
                        residuals.v_minus_x_minus_w_plus_one, 0.0
                    )
                    self.assertGreaterEqual(residuals.v_nonnegative, 0.0)
                    self.assertTrue(satisfies_mccormick_envelope(x, w, v))

    def test_named_residuals_expose_sign_and_violated_inequality(self) -> None:
        """Verify residual arithmetic independently of the boolean checker."""

        residuals = compute_mccormick_residuals(0.4, 1, 0.3)
        self.assertAlmostEqual(0.1, residuals.x_minus_v)
        self.assertAlmostEqual(0.7, residuals.w_minus_v)
        self.assertAlmostEqual(-0.1, residuals.v_minus_x_minus_w_plus_one)
        self.assertAlmostEqual(0.3, residuals.v_nonnegative)
        self.assertAlmostEqual(-0.1, residuals.minimum_residual)

    def test_values_above_upper_and_below_lower_bounds_fail(self) -> None:
        """Detect deliberately non-boundary violations in both directions."""

        self.assertFalse(satisfies_mccormick_envelope(0.4, 0, 0.2))
        self.assertFalse(satisfies_mccormick_envelope(0.4, 0, -0.2))
        self.assertFalse(satisfies_mccormick_envelope(0.4, 1, 0.6))
        self.assertFalse(satisfies_mccormick_envelope(0.4, 1, 0.2))


class McCormickValidationTests(unittest.TestCase):
    """Keep mathematical-domain validation separate from numeric tolerance."""

    def test_invalid_x_is_rejected_without_clipping(self) -> None:
        """Reject values outside ``[0,1]`` and non-finite pseudo-values."""

        for invalid_x in (-0.001, 1.001, float("nan"), float("inf"), True):
            with self.subTest(x=invalid_x):
                with self.assertRaises(ValueError):
                    compute_binary_continuous_product(invalid_x, 1)

    def test_non_binary_w_is_rejected_without_rounding(self) -> None:
        """Reject non-binary integers, fractional values, and booleans."""

        for invalid_w in (-1, 2, 0.5, 0.9999999999, True):
            with self.subTest(w=invalid_w):
                with self.assertRaises(ValueError):
                    compute_binary_continuous_product(0.5, invalid_w)

    def test_exact_float_binary_values_are_accepted(self) -> None:
        """Accept exact numeric solver-style values without rounding them."""

        self.assertEqual(0.0, compute_binary_continuous_product(0.5, 0.0))
        self.assertEqual(0.5, compute_binary_continuous_product(0.5, 1.0))

    def test_invalid_v_and_tolerance_are_rejected(self) -> None:
        """Require finite ``v`` and a finite non-negative tolerance."""

        for invalid_v in (float("nan"), float("inf"), True):
            with self.subTest(v=invalid_v):
                with self.assertRaises(ValueError):
                    compute_mccormick_residuals(0.5, 1, invalid_v)
        for invalid_tolerance in (-1e-9, float("nan"), float("inf"), True):
            with self.subTest(tolerance=invalid_tolerance):
                with self.assertRaises(ValueError):
                    satisfies_mccormick_envelope(
                        0.5, 1, 0.5, tolerance=invalid_tolerance
                    )

    def test_tolerance_affects_only_inequality_feasibility(self) -> None:
        """Allow tiny residual error without changing products or domains."""

        x = 0.4
        slightly_high_v = x + 5e-10
        self.assertFalse(satisfies_mccormick_envelope(x, 1, slightly_high_v))
        self.assertTrue(
            satisfies_mccormick_envelope(
                x, 1, slightly_high_v, tolerance=1e-9
            )
        )
        self.assertEqual(x, compute_binary_continuous_product(x, 1))
        with self.assertRaises(ValueError):
            satisfies_mccormick_envelope(1.0 + 5e-10, 1, 1.0, tolerance=1e-9)
        with self.assertRaises(ValueError):
            satisfies_mccormick_envelope(x, 0.9999999999, x, tolerance=1e-9)


if __name__ == "__main__":
    unittest.main()
