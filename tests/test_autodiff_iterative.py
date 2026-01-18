"""Tests for iterative automatic differentiation.

This module tests the iterative gradient computation engine introduced in
v1.2.1, which enables gradient computation for deep expression trees
(depth > 1000) without hitting Python's recursion limit.

Issue #63: Add iterative autodiff test suite
"""

import sys
import time

import numpy as np
import pytest

from optyx import Constant, Variable, VectorVariable, sin, cos, exp, log, sqrt, tanh
from optyx.core.autodiff import (
    _estimate_tree_depth,
    _gradient_cached,
    _gradient_iterative,
    _RECURSION_THRESHOLD,
    gradient,
    increased_recursion_limit,
)


# =============================================================================
# Test Basic Operators - Iterative vs Recursive Equivalence
# =============================================================================


class TestBasicOperatorsIterative:
    """Test that iterative gradient matches recursive for basic operators."""

    def test_addition_iterative(self):
        """d/dx(x + y) = 1"""
        x = Variable("x")
        y = Variable("y")
        expr = x + y

        grad_recursive = _gradient_cached(expr, x)
        grad_iterative = _gradient_iterative(expr, x)

        point = {"x": 3.0, "y": 4.0}
        assert grad_recursive.evaluate(point) == grad_iterative.evaluate(point)
        assert grad_iterative.evaluate(point) == 1.0

    def test_subtraction_iterative(self):
        """d/dx(x - y) = 1, d/dy(x - y) = -1"""
        x = Variable("x")
        y = Variable("y")
        expr = x - y

        grad_x_iter = _gradient_iterative(expr, x)
        grad_y_iter = _gradient_iterative(expr, y)

        point = {"x": 3.0, "y": 4.0}
        assert grad_x_iter.evaluate(point) == 1.0
        assert grad_y_iter.evaluate(point) == -1.0

    def test_multiplication_iterative(self):
        """d/dx(x * y) = y (product rule)"""
        x = Variable("x")
        y = Variable("y")
        expr = x * y

        grad_x_rec = _gradient_cached(expr, x)
        grad_x_iter = _gradient_iterative(expr, x)

        point = {"x": 3.0, "y": 4.0}
        assert grad_x_rec.evaluate(point) == grad_x_iter.evaluate(point)
        assert grad_x_iter.evaluate(point) == 4.0  # d/dx(x*y) = y

    def test_division_iterative(self):
        """d/dx(x / y) = 1/y (quotient rule)"""
        x = Variable("x")
        y = Variable("y")
        expr = x / y

        grad_x_iter = _gradient_iterative(expr, x)
        grad_y_iter = _gradient_iterative(expr, y)

        point = {"x": 6.0, "y": 2.0}
        np.testing.assert_almost_equal(grad_x_iter.evaluate(point), 0.5)  # 1/2
        np.testing.assert_almost_equal(grad_y_iter.evaluate(point), -1.5)  # -6/4

    def test_power_constant_exponent_iterative(self):
        """d/dx(x^n) = n * x^(n-1)"""
        x = Variable("x")
        expr = x**3

        grad_rec = _gradient_cached(expr, x)
        grad_iter = _gradient_iterative(expr, x)

        point = {"x": 2.0}
        assert grad_rec.evaluate(point) == grad_iter.evaluate(point)
        assert grad_iter.evaluate(point) == 12.0  # 3 * 2^2 = 12

    def test_power_zero_exponent(self):
        """d/dx(x^0) = 0"""
        x = Variable("x")
        expr = x**0

        grad_iter = _gradient_iterative(expr, x)
        assert grad_iter.evaluate({"x": 5.0}) == 0.0

    def test_power_one_exponent(self):
        """d/dx(x^1) = 1"""
        x = Variable("x")
        expr = x**1

        grad_iter = _gradient_iterative(expr, x)
        assert grad_iter.evaluate({"x": 5.0}) == 1.0

    def test_negation_iterative(self):
        """d/dx(-x) = -1"""
        x = Variable("x")
        expr = -x

        grad_iter = _gradient_iterative(expr, x)
        assert grad_iter.evaluate({"x": 5.0}) == -1.0


# =============================================================================
# Test Transcendental Functions - Iterative
# =============================================================================


class TestTranscendentalsIterative:
    """Test iterative gradients for transcendental functions."""

    def test_sin_iterative(self):
        """d/dx(sin(x)) = cos(x)"""
        x = Variable("x")
        expr = sin(x)

        grad_iter = _gradient_iterative(expr, x)

        # At x=0: cos(0) = 1
        np.testing.assert_almost_equal(grad_iter.evaluate({"x": 0.0}), 1.0)
        # At x=π/2: cos(π/2) ≈ 0
        np.testing.assert_almost_equal(
            grad_iter.evaluate({"x": np.pi / 2}), 0.0, decimal=10
        )

    def test_cos_iterative(self):
        """d/dx(cos(x)) = -sin(x)"""
        x = Variable("x")
        expr = cos(x)

        grad_iter = _gradient_iterative(expr, x)

        # At x=0: -sin(0) = 0
        np.testing.assert_almost_equal(grad_iter.evaluate({"x": 0.0}), 0.0)
        # At x=π/2: -sin(π/2) = -1
        np.testing.assert_almost_equal(grad_iter.evaluate({"x": np.pi / 2}), -1.0)

    def test_exp_iterative(self):
        """d/dx(exp(x)) = exp(x)"""
        x = Variable("x")
        expr = exp(x)

        grad_iter = _gradient_iterative(expr, x)

        np.testing.assert_almost_equal(grad_iter.evaluate({"x": 0.0}), 1.0)
        np.testing.assert_almost_equal(grad_iter.evaluate({"x": 1.0}), np.e)

    def test_log_iterative(self):
        """d/dx(log(x)) = 1/x"""
        x = Variable("x")
        expr = log(x)

        grad_iter = _gradient_iterative(expr, x)

        np.testing.assert_almost_equal(grad_iter.evaluate({"x": 1.0}), 1.0)
        np.testing.assert_almost_equal(grad_iter.evaluate({"x": 2.0}), 0.5)

    def test_sqrt_iterative(self):
        """d/dx(sqrt(x)) = 1/(2*sqrt(x))"""
        x = Variable("x")
        expr = sqrt(x)

        grad_iter = _gradient_iterative(expr, x)

        # At x=4: 1/(2*2) = 0.25
        np.testing.assert_almost_equal(grad_iter.evaluate({"x": 4.0}), 0.25)

    def test_tanh_iterative(self):
        """d/dx(tanh(x)) = 1 - tanh^2(x) = sech^2(x)"""
        x = Variable("x")
        expr = tanh(x)

        grad_iter = _gradient_iterative(expr, x)

        # At x=0: 1 - tanh^2(0) = 1 - 0 = 1
        np.testing.assert_almost_equal(grad_iter.evaluate({"x": 0.0}), 1.0)


# =============================================================================
# Test Chain Rule Combinations
# =============================================================================


class TestChainRuleIterative:
    """Test chain rule combinations with iterative gradient."""

    def test_sin_of_quadratic(self):
        """d/dx(sin(x^2)) = cos(x^2) * 2x"""
        x = Variable("x")
        expr = sin(x**2)

        grad_rec = _gradient_cached(expr, x)
        grad_iter = _gradient_iterative(expr, x)

        point = {"x": 1.0}
        np.testing.assert_almost_equal(
            grad_rec.evaluate(point), grad_iter.evaluate(point)
        )
        # At x=1: cos(1) * 2*1 = cos(1) * 2
        expected = np.cos(1.0) * 2.0
        np.testing.assert_almost_equal(grad_iter.evaluate(point), expected)

    def test_exp_of_product(self):
        """d/dx(exp(x*y)) = y * exp(x*y)"""
        x = Variable("x")
        y = Variable("y")
        expr = exp(x * y)

        grad_iter = _gradient_iterative(expr, x)

        point = {"x": 1.0, "y": 2.0}
        # d/dx(exp(x*y)) = exp(x*y) * y = exp(2) * 2
        expected = np.exp(2.0) * 2.0
        np.testing.assert_almost_equal(grad_iter.evaluate(point), expected)

    def test_log_of_sqrt(self):
        """d/dx(log(sqrt(x))) = 1/(2x)"""
        x = Variable("x")
        expr = log(sqrt(x))

        grad_iter = _gradient_iterative(expr, x)

        # At x=4: 1/(2*4) = 0.125
        np.testing.assert_almost_equal(grad_iter.evaluate({"x": 4.0}), 0.125)

    def test_nested_operations(self):
        """Test deeply nested operations match between recursive and iterative."""
        x = Variable("x")
        # sin(cos(exp(x^2)))
        expr = sin(cos(exp(x**2)))

        grad_rec = _gradient_cached(expr, x)
        grad_iter = _gradient_iterative(expr, x)

        point = {"x": 0.5}
        np.testing.assert_almost_equal(
            grad_rec.evaluate(point), grad_iter.evaluate(point), decimal=10
        )


# =============================================================================
# Test Auto-Switching Behavior
# =============================================================================


class TestAutoSwitching:
    """Test automatic switching between recursive and iterative at depth threshold."""

    def test_recursion_threshold_value(self):
        """Verify threshold is set to a safe value (< Python's recursion limit)."""
        assert _RECURSION_THRESHOLD < sys.getrecursionlimit()
        assert _RECURSION_THRESHOLD == 400  # Current documented value

    def test_shallow_tree_uses_recursive(self):
        """Shallow trees should use cached recursive gradient."""
        x = Variable("x")
        # Depth ~10
        expr = x
        for _ in range(10):
            expr = expr + x

        depth = _estimate_tree_depth(expr)
        assert depth < _RECURSION_THRESHOLD

        # gradient() should work without issues (uses recursive)
        grad = gradient(expr, x)
        assert grad.evaluate({"x": 1.0}) == 11.0  # Sum of 11 ones

    def test_deep_tree_switches_to_iterative(self):
        """Trees at/above threshold should switch to iterative."""
        x = Variable("x")

        # Build tree just above threshold
        expr = x
        for _ in range(_RECURSION_THRESHOLD + 50):
            expr = expr + x

        depth = _estimate_tree_depth(expr)
        assert depth >= _RECURSION_THRESHOLD

        # gradient() should switch to iterative automatically
        grad = gradient(expr, x)
        expected = _RECURSION_THRESHOLD + 51  # n+1 terms
        assert grad.evaluate({"x": 1.0}) == expected

    def test_estimate_depth_left_skewed(self):
        """Left-skewed trees (common from loops) are detected correctly."""
        x = Variable("x")

        # obj = obj + x pattern creates left-skewed tree
        expr = x
        for _ in range(100):
            expr = expr + x

        depth = _estimate_tree_depth(expr)
        assert depth == 100  # Exact for left-skewed

    def test_estimate_depth_full_traversal(self):
        """full_traversal=True finds exact depth for any tree shape."""
        x = Variable("x")
        y = Variable("y")

        # Build a balanced-ish tree
        expr = (x + y) * (x - y) + (x * y) / (x + Constant(1.0))

        depth_heuristic = _estimate_tree_depth(expr)
        depth_full = _estimate_tree_depth(expr, full_traversal=True)

        # Full traversal should find actual depth
        assert depth_full >= depth_heuristic


# =============================================================================
# Test Deep Trees - No RecursionError
# =============================================================================


class TestDeepTrees:
    """Test gradient computation on deep expression trees.

    The iterative gradient engine enables gradient *computation* for deep trees
    without RecursionError. However, *evaluating* the resulting gradient
    expression (which is also deep) may still hit recursion limits.

    For truly deep problems (n>1000), users should use VectorVariable with
    its O(1) gradient rules, which avoids deep expression trees entirely.
    """

    def test_depth_500_no_recursion_error(self):
        """n=500 depth should work without RecursionError."""
        x = Variable("x")

        expr = x
        for _ in range(499):
            expr = expr + x

        # Should not raise RecursionError
        grad = gradient(expr, x)
        assert grad.evaluate({"x": 1.0}) == 500

    def test_depth_1000_gradient_computation_succeeds(self):
        """n=1000: gradient computation succeeds without RecursionError.

        Note: The gradient expression itself is deep (sum of 1000 ones),
        which means evaluation may hit limits. This test verifies gradient
        *computation* is safe.
        """
        x = Variable("x")

        expr = x
        for _ in range(999):
            expr = expr + x

        # Gradient computation should NOT raise RecursionError
        # (uses iterative algorithm)
        grad = gradient(expr, x)

        # Verify we got a gradient expression
        assert grad is not None

    def test_depth_10000_gradient_computation_succeeds(self):
        """n=10000: gradient computation succeeds without RecursionError.

        Demonstrates the iterative gradient engine handles very deep trees.
        """
        x = Variable("x")

        expr = x
        for _ in range(9999):
            expr = expr + x

        # Gradient computation should NOT raise RecursionError
        grad = gradient(expr, x)

        # Verify we got a gradient expression
        assert grad is not None

    def test_constant_scalar_multiplication_gradient(self):
        """Simple scalar multiplication: d/dx(c * x) = c.

        Unlike deep trees built with loops, single scalar multiplication
        produces a simple constant gradient.
        """
        x = Variable("x")

        # c * x has gradient c
        expr = Constant(2.0**10) * x

        grad = gradient(expr, x)
        assert grad.evaluate({"x": 1.0}) == 2.0**10

    def test_deep_mixed_operations_gradient_computation(self):
        """Deep tree with mixed ops: gradient computation succeeds."""
        x = Variable("x")

        expr = x
        for i in range(600):
            if i % 3 == 0:
                expr = expr + x
            elif i % 3 == 1:
                expr = expr - x
            else:
                expr = expr * Constant(1.0)

        # Gradient computation should succeed
        grad = gradient(expr, x)
        assert grad is not None

    def test_vectorvariable_avoids_deep_trees(self):
        """VectorVariable.sum() produces O(1) gradient - no deep trees.

        This is the recommended approach for large-scale problems.
        """
        n = 10000
        x = VectorVariable("x", n)
        expr = x.sum()  # sum(x) has O(1) gradient

        # Gradient is constant (VectorSum registered rule)
        grad = gradient(expr, x[0])

        # Evaluate without issues - gradient is just Constant(1.0)
        assert grad.evaluate({f"x[{i}]": 1.0 for i in range(n)}) == 1.0


# =============================================================================
# Test Results Match Between Recursive and Iterative
# =============================================================================


class TestResultsMatch:
    """Verify recursive and iterative produce identical results."""

    @pytest.mark.parametrize(
        "expr_builder",
        [
            lambda x: x + x,
            lambda x: x * x,
            lambda x: x**2,
            lambda x: x**3,
            lambda x: sin(x),
            lambda x: cos(x),
            lambda x: exp(x),
            lambda x: log(x),
            lambda x: sqrt(x),
            lambda x: tanh(x),
            lambda x: sin(x) * cos(x),
            lambda x: exp(x**2),
            lambda x: log(sqrt(x)),
            lambda x: (x + Constant(1.0)) * (x - Constant(1.0)),
        ],
    )
    def test_single_variable_expressions(self, expr_builder):
        """Test various single-variable expressions."""
        x = Variable("x")
        expr = expr_builder(x)

        grad_rec = _gradient_cached(expr, x)
        grad_iter = _gradient_iterative(expr, x)

        # Test at several points
        for val in [0.5, 1.0, 2.0, 3.0]:
            point = {"x": val}
            np.testing.assert_almost_equal(
                grad_rec.evaluate(point),
                grad_iter.evaluate(point),
                decimal=10,
                err_msg=f"Mismatch at x={val}",
            )

    def test_two_variable_expressions(self):
        """Test expressions with two variables."""
        x = Variable("x")
        y = Variable("y")

        expressions = [
            x + y,
            x * y,
            x / y,
            x**2 + y**2,
            sin(x) * cos(y),
            exp(x * y),
        ]

        point = {"x": 1.5, "y": 2.0}

        for expr in expressions:
            for wrt in [x, y]:
                grad_rec = _gradient_cached(expr, wrt)
                grad_iter = _gradient_iterative(expr, wrt)
                np.testing.assert_almost_equal(
                    grad_rec.evaluate(point), grad_iter.evaluate(point), decimal=10
                )


# =============================================================================
# Test increased_recursion_limit Context Manager
# =============================================================================


class TestIncreasedRecursionLimit:
    """Test the increased_recursion_limit context manager."""

    def test_limit_increases_inside_context(self):
        """Recursion limit should increase inside the context."""
        original_limit = sys.getrecursionlimit()

        with increased_recursion_limit(5000):
            assert sys.getrecursionlimit() == 5000

        assert sys.getrecursionlimit() == original_limit

    def test_limit_restores_on_exception(self):
        """Recursion limit should restore even if exception occurs."""
        original_limit = sys.getrecursionlimit()

        try:
            with increased_recursion_limit(6000):
                raise ValueError("test exception")
        except ValueError:
            pass

        assert sys.getrecursionlimit() == original_limit

    def test_default_limit_value(self):
        """Default limit should be 5000."""
        with increased_recursion_limit():
            assert sys.getrecursionlimit() == 5000


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance tests for iterative gradient computation."""

    @pytest.mark.slow
    def test_n10000_gradient_computation_under_500ms(self):
        """n=10000 gradient computation should complete in < 500ms.

        Tests only gradient computation, not evaluation.
        """
        x = Variable("x")

        # Build expression
        expr = x
        for _ in range(9999):
            expr = expr + x

        # Time the gradient computation only
        start = time.perf_counter()
        grad = gradient(expr, x)
        elapsed = time.perf_counter() - start

        # Verify gradient was computed
        assert grad is not None
        assert elapsed < 0.5, f"Took {elapsed * 1000:.1f}ms, expected < 500ms"

    def test_vector_sum_o1_gradient(self):
        """VectorVariable.sum() should have O(1) gradient time."""
        # O(1) because VectorSum has registered gradient rule
        for n in [100, 1000, 10000]:
            x = VectorVariable("x", n)
            expr = x.sum()

            start = time.perf_counter()
            grad = gradient(expr, x[0])
            _ = grad.evaluate({f"x[{i}]": float(i) for i in range(n)})
            elapsed = time.perf_counter() - start

            # Should be very fast regardless of n
            assert elapsed < 0.1, f"n={n} took {elapsed * 1000:.1f}ms"


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge cases for iterative gradient computation."""

    def test_constant_expression(self):
        """Gradient of constant is zero."""
        x = Variable("x")
        expr = Constant(42.0)

        grad = _gradient_iterative(expr, x)
        assert grad.evaluate({"x": 1.0}) == 0.0

    def test_variable_wrt_itself(self):
        """d/dx(x) = 1"""
        x = Variable("x")
        grad = _gradient_iterative(x, x)
        assert grad.evaluate({"x": 5.0}) == 1.0

    def test_variable_wrt_other(self):
        """d/dy(x) = 0"""
        x = Variable("x")
        y = Variable("y")
        grad = _gradient_iterative(x, y)
        assert grad.evaluate({"x": 5.0, "y": 3.0}) == 0.0

    def test_abs_gradient(self):
        """d/dx(|x|) = sign(x)"""
        from optyx import abs_

        x = Variable("x")
        expr = abs_(x)

        grad = _gradient_iterative(expr, x)

        # At x=2: sign(2) = 1
        assert grad.evaluate({"x": 2.0}) == 1.0
        # At x=-3: sign(-3) = -1
        assert grad.evaluate({"x": -3.0}) == -1.0

    def test_deeply_nested_unary(self):
        """Deep chain of unary operations."""
        x = Variable("x")

        # sin(sin(sin(...(x)...))) - 100 deep
        expr = x
        for _ in range(100):
            expr = sin(expr)

        # Should not crash
        grad = gradient(expr, x)
        assert isinstance(grad.evaluate({"x": 0.1}), float)
