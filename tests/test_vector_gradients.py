"""Tests for native vector gradient rules.

These tests verify that gradient rules for vector expressions (VectorSum,
DotProduct, LinearCombination, L2Norm, L1Norm, QuadraticForm) work correctly
and have O(1) complexity for VectorVariable inputs.
"""

import numpy as np
import pytest
import time
from numpy.testing import assert_allclose, assert_array_almost_equal

from optyx import VectorVariable, Variable, Constant
from optyx.core.autodiff import (
    gradient,
    has_gradient_rule,
    _gradient_registry,
)
from optyx.core.vectors import (
    VectorSum,
    DotProduct,
    LinearCombination,
    L2Norm,
    L1Norm,
)
from optyx.core.matrices import QuadraticForm
from optyx.core.compiler import compile_expression


class TestGradientRegistry:
    """Tests for the gradient registry system."""

    def test_vector_types_registered(self):
        """All vector expression types should have registered gradient rules."""
        assert VectorSum in _gradient_registry
        assert DotProduct in _gradient_registry
        assert LinearCombination in _gradient_registry
        assert L2Norm in _gradient_registry
        assert L1Norm in _gradient_registry
        assert QuadraticForm in _gradient_registry

    def test_has_gradient_rule(self):
        """has_gradient_rule should return True for vector expressions."""
        x = VectorVariable("x", 5)

        assert has_gradient_rule(VectorSum(x))
        assert has_gradient_rule(x.dot(x))
        assert has_gradient_rule(np.ones(5) @ x)
        assert has_gradient_rule(L2Norm(x))
        assert has_gradient_rule(L1Norm(x))

        Q = np.eye(5)
        assert has_gradient_rule(QuadraticForm(x, Q))


class TestLinearCombinationGradient:
    """Tests for LinearCombination gradient: ∂(c·x)/∂x_i = c_i."""

    def test_basic_gradient(self):
        """Gradient of c @ x with respect to x_i should be c_i."""
        x = VectorVariable("x", 5)
        c = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        expr = c @ x  # LinearCombination

        for i in range(5):
            grad = gradient(expr, x[i])
            # Gradient should be constant c_i
            assert isinstance(grad, Constant)
            assert grad.value == c[i]

    def test_gradient_not_in_vector(self):
        """Gradient with respect to variable not in vector should be 0."""
        x = VectorVariable("x", 3)
        y = Variable("y")
        c = np.array([1.0, 2.0, 3.0])
        expr = c @ x

        grad = gradient(expr, y)
        assert isinstance(grad, Constant)
        assert grad.value == 0.0

    def test_compiled_gradient_correctness(self):
        """Compiled gradient should match numerical gradient."""
        n = 100
        x = VectorVariable("x", n)
        c = np.random.rand(n)
        expr = c @ x

        # Compile gradients
        variables = list(x)
        grad_fns = [compile_expression(gradient(expr, v), variables) for v in variables]

        # Evaluate at random point
        x_val = np.random.rand(n)
        computed = np.array([fn(x_val) for fn in grad_fns])

        # Gradient of c @ x is just c
        assert_array_almost_equal(computed, c)


class TestVectorSumGradient:
    """Tests for VectorSum gradient: ∂(Σx_i)/∂x_j = 1 if j in x else 0."""

    def test_basic_gradient(self):
        """Gradient of sum(x) with respect to any x_i should be 1."""
        x = VectorVariable("x", 5)
        expr = x.sum()

        for i in range(5):
            grad = gradient(expr, x[i])
            assert isinstance(grad, Constant)
            assert grad.value == 1.0

    def test_gradient_not_in_vector(self):
        """Gradient with respect to variable not in vector should be 0."""
        x = VectorVariable("x", 3)
        y = Variable("y")
        expr = x.sum()

        grad = gradient(expr, y)
        assert isinstance(grad, Constant)
        assert grad.value == 0.0

    def test_compiled_gradient_correctness(self):
        """Compiled gradient should be all ones."""
        n = 100
        x = VectorVariable("x", n)
        expr = x.sum()

        variables = list(x)
        grad_fns = [compile_expression(gradient(expr, v), variables) for v in variables]

        x_val = np.random.rand(n)
        computed = np.array([fn(x_val) for fn in grad_fns])

        # Gradient of sum(x) is all ones
        assert_array_almost_equal(computed, np.ones(n))


class TestDotProductGradient:
    """Tests for DotProduct gradient."""

    def test_self_dot_product(self):
        """Gradient of x · x with respect to x_i should be 2*x_i."""
        x = VectorVariable("x", 5)
        expr = x.dot(x)

        # Test at a specific point
        point = {"x[0]": 1.0, "x[1]": 2.0, "x[2]": 3.0, "x[3]": 4.0, "x[4]": 5.0}

        for i in range(5):
            grad = gradient(expr, x[i])
            # Should be 2 * x[i]
            val = grad.evaluate(point)
            expected = 2.0 * point[f"x[{i}]"]
            assert_allclose(val, expected)

    def test_different_vectors(self):
        """Gradient of x · y with respect to x_i should be y_i."""
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        expr = x.dot(y)

        point = {
            "x[0]": 1.0,
            "x[1]": 2.0,
            "x[2]": 3.0,
            "y[0]": 4.0,
            "y[1]": 5.0,
            "y[2]": 6.0,
        }

        # ∂(x·y)/∂x_i = y_i
        for i in range(3):
            grad_x = gradient(expr, x[i])
            val = grad_x.evaluate(point)
            assert_allclose(val, point[f"y[{i}]"])

        # ∂(x·y)/∂y_i = x_i
        for i in range(3):
            grad_y = gradient(expr, y[i])
            val = grad_y.evaluate(point)
            assert_allclose(val, point[f"x[{i}]"])

    def test_gradient_not_in_vectors(self):
        """Gradient with respect to unrelated variable should be 0."""
        x = VectorVariable("x", 3)
        z = Variable("z")
        expr = x.dot(x)

        grad = gradient(expr, z)
        assert isinstance(grad, Constant)
        assert grad.value == 0.0

    def test_compiled_self_dot_gradient(self):
        """Compiled gradient of x·x should be 2x."""
        n = 100
        x = VectorVariable("x", n)
        expr = x.dot(x)

        variables = list(x)
        grad_fns = [compile_expression(gradient(expr, v), variables) for v in variables]

        x_val = np.random.rand(n)
        computed = np.array([fn(x_val) for fn in grad_fns])

        # Gradient of x·x is 2x
        assert_array_almost_equal(computed, 2 * x_val)


class TestL2NormGradient:
    """Tests for L2Norm gradient: ∂||x||/∂x_i = x_i / ||x||."""

    def test_basic_gradient(self):
        """Gradient of ||x|| with respect to x_i should be x_i / ||x||."""
        x = VectorVariable("x", 3)
        expr = L2Norm(x)

        # Test at point (3, 4, 0) - norm is 5
        point = {"x[0]": 3.0, "x[1]": 4.0, "x[2]": 0.0}
        norm_val = 5.0

        grad_0 = gradient(expr, x[0])
        grad_1 = gradient(expr, x[1])
        grad_2 = gradient(expr, x[2])

        assert_allclose(grad_0.evaluate(point), 3.0 / norm_val)
        assert_allclose(grad_1.evaluate(point), 4.0 / norm_val)
        assert_allclose(grad_2.evaluate(point), 0.0 / norm_val)

    def test_gradient_not_in_vector(self):
        """Gradient with respect to variable not in vector should be 0."""
        x = VectorVariable("x", 3)
        y = Variable("y")
        expr = L2Norm(x)

        grad = gradient(expr, y)
        assert isinstance(grad, Constant)
        assert grad.value == 0.0

    def test_compiled_gradient_correctness(self):
        """Compiled gradient should match x / ||x||."""
        n = 50
        x = VectorVariable("x", n)
        expr = L2Norm(x)

        variables = list(x)
        grad_fns = [compile_expression(gradient(expr, v), variables) for v in variables]

        x_val = np.random.rand(n) + 0.1  # Avoid zero
        computed = np.array([fn(x_val) for fn in grad_fns])

        # Gradient of ||x|| is x / ||x||
        expected = x_val / np.linalg.norm(x_val)
        assert_array_almost_equal(computed, expected)

    def test_numerical_gradient_check(self):
        """Verify gradient numerically with finite differences."""
        n = 10
        x = VectorVariable("x", n)
        expr = L2Norm(x)

        variables = list(x)
        grad_fns = [compile_expression(gradient(expr, v), variables) for v in variables]
        obj_fn = compile_expression(expr, variables)

        x_val = np.random.rand(n) + 0.5
        eps = 1e-7

        for i in range(n):
            # Numerical gradient
            x_plus = x_val.copy()
            x_plus[i] += eps
            x_minus = x_val.copy()
            x_minus[i] -= eps
            numerical = (obj_fn(x_plus) - obj_fn(x_minus)) / (2 * eps)

            # Symbolic gradient
            symbolic = grad_fns[i](x_val)

            assert_allclose(symbolic, numerical, rtol=1e-5)


class TestL1NormGradient:
    """Tests for L1Norm gradient: ∂||x||₁/∂x_i = sign(x_i)."""

    def test_basic_gradient_positive(self):
        """Gradient of ||x||₁ with respect to positive x_i should be 1."""
        x = VectorVariable("x", 3)
        expr = L1Norm(x)

        point = {"x[0]": 2.0, "x[1]": 3.0, "x[2]": 1.0}

        for i in range(3):
            grad = gradient(expr, x[i])
            val = grad.evaluate(point)
            assert_allclose(val, 1.0)

    def test_basic_gradient_negative(self):
        """Gradient of ||x||₁ with respect to negative x_i should be -1."""
        x = VectorVariable("x", 3)
        expr = L1Norm(x)

        point = {"x[0]": -2.0, "x[1]": -3.0, "x[2]": -1.0}

        for i in range(3):
            grad = gradient(expr, x[i])
            val = grad.evaluate(point)
            assert_allclose(val, -1.0)

    def test_gradient_not_in_vector(self):
        """Gradient with respect to variable not in vector should be 0."""
        x = VectorVariable("x", 3)
        y = Variable("y")
        expr = L1Norm(x)

        grad = gradient(expr, y)
        assert isinstance(grad, Constant)
        assert grad.value == 0.0

    def test_compiled_gradient_correctness(self):
        """Compiled gradient should match sign(x)."""
        n = 50
        x = VectorVariable("x", n)
        expr = L1Norm(x)

        variables = list(x)
        grad_fns = [compile_expression(gradient(expr, v), variables) for v in variables]

        # Mix of positive and negative values (avoid zero)
        x_val = np.random.rand(n) * 2 - 1
        x_val[x_val == 0] = 0.1  # Avoid zero

        computed = np.array([fn(x_val) for fn in grad_fns])

        # Gradient of ||x||₁ is sign(x)
        expected = np.sign(x_val)
        assert_array_almost_equal(computed, expected)


class TestQuadraticFormGradient:
    """Tests for QuadraticForm gradient: ∂(x'Qx)/∂x_i = [(Q + Q')x]_i."""

    def test_identity_matrix(self):
        """Gradient of x'Ix with respect to x_i should be 2x_i."""
        n = 5
        x = VectorVariable("x", n)
        Q = np.eye(n)
        expr = QuadraticForm(x, Q)

        point = {f"x[{i}]": float(i + 1) for i in range(n)}

        for i in range(n):
            grad = gradient(expr, x[i])
            val = grad.evaluate(point)
            expected = 2.0 * point[f"x[{i}]"]
            assert_allclose(val, expected)

    def test_symmetric_matrix(self):
        """Gradient of x'Qx for symmetric Q should be 2Qx."""
        n = 4
        x = VectorVariable("x", n)
        # Random symmetric matrix
        A = np.random.rand(n, n)
        Q = (A + A.T) / 2  # Symmetric
        expr = QuadraticForm(x, Q)

        variables = list(x)
        grad_fns = [compile_expression(gradient(expr, v), variables) for v in variables]

        x_val = np.random.rand(n)
        computed = np.array([fn(x_val) for fn in grad_fns])

        # For symmetric Q: gradient is 2Qx
        expected = 2 * Q @ x_val
        assert_array_almost_equal(computed, expected)

    def test_asymmetric_matrix(self):
        """Gradient of x'Qx for asymmetric Q should be (Q + Q')x."""
        n = 4
        x = VectorVariable("x", n)
        Q = np.random.rand(n, n)  # Not symmetric
        expr = QuadraticForm(x, Q)

        variables = list(x)
        grad_fns = [compile_expression(gradient(expr, v), variables) for v in variables]

        x_val = np.random.rand(n)
        computed = np.array([fn(x_val) for fn in grad_fns])

        # General case: gradient is (Q + Q')x
        expected = (Q + Q.T) @ x_val
        assert_array_almost_equal(computed, expected)

    def test_gradient_not_in_vector(self):
        """Gradient with respect to variable not in vector should be 0."""
        x = VectorVariable("x", 3)
        y = Variable("y")
        Q = np.eye(3)
        expr = QuadraticForm(x, Q)

        grad = gradient(expr, y)
        assert isinstance(grad, Constant)
        assert grad.value == 0.0

    def test_numerical_gradient_check(self):
        """Verify gradient numerically with finite differences."""
        n = 5
        x = VectorVariable("x", n)
        Q = np.random.rand(n, n)
        expr = QuadraticForm(x, Q)

        variables = list(x)
        grad_fns = [compile_expression(gradient(expr, v), variables) for v in variables]
        obj_fn = compile_expression(expr, variables)

        x_val = np.random.rand(n)
        eps = 1e-7

        for i in range(n):
            # Numerical gradient
            x_plus = x_val.copy()
            x_plus[i] += eps
            x_minus = x_val.copy()
            x_minus[i] -= eps
            numerical = (obj_fn(x_plus) - obj_fn(x_minus)) / (2 * eps)

            # Symbolic gradient
            symbolic = grad_fns[i](x_val)

            assert_allclose(symbolic, numerical, rtol=1e-5)


class TestGradientComplexity:
    """Tests to verify O(1) gradient complexity for VectorVariable."""

    @pytest.mark.parametrize("n", [100, 500, 1000])
    def test_linear_combination_constant_time(self, n):
        """LinearCombination gradient should be O(1) regardless of vector size."""
        x = VectorVariable("x", n)
        c = np.random.rand(n)
        expr = c @ x

        # Time gradient computation for first variable
        start = time.perf_counter()
        for _ in range(100):
            grad = gradient(expr, x[0])
        elapsed_ms = (time.perf_counter() - start) * 10  # ms per iteration

        # Should be very fast (< 1ms per gradient)
        assert (
            elapsed_ms < 1.0
        ), f"LinearCombination gradient too slow: {elapsed_ms:.3f}ms"

        # Result should be a constant
        assert isinstance(grad, Constant)

    @pytest.mark.parametrize("n", [100, 500, 1000])
    def test_vector_sum_constant_time(self, n):
        """VectorSum gradient should be O(1) for first element."""
        x = VectorVariable("x", n)
        expr = x.sum()

        start = time.perf_counter()
        for _ in range(100):
            grad = gradient(expr, x[0])
        elapsed_ms = (time.perf_counter() - start) * 10

        assert elapsed_ms < 1.0, f"VectorSum gradient too slow: {elapsed_ms:.3f}ms"
        assert isinstance(grad, Constant)
        assert float(grad.value) == 1.0

    @pytest.mark.parametrize("n", [100, 500, 1000])
    def test_dot_product_constant_time(self, n):
        """DotProduct gradient should be O(1) for VectorVariable."""
        x = VectorVariable("x", n)
        expr = x.dot(x)

        start = time.perf_counter()
        for _ in range(100):
            grad = gradient(expr, x[0])  # noqa: F841
        elapsed_ms = (time.perf_counter() - start) * 10

        assert elapsed_ms < 1.0, f"DotProduct gradient too slow: {elapsed_ms:.3f}ms"

    @pytest.mark.parametrize("n", [100, 500, 1000])
    def test_l2_norm_constant_time(self, n):
        """L2Norm gradient should be O(1) for VectorVariable."""
        x = VectorVariable("x", n)
        expr = L2Norm(x)

        start = time.perf_counter()
        for _ in range(100):
            grad = gradient(expr, x[0])  # noqa: F841
        elapsed_ms = (time.perf_counter() - start) * 10

        assert elapsed_ms < 1.0, f"L2Norm gradient too slow: {elapsed_ms:.3f}ms"

    @pytest.mark.parametrize("n", [100, 500, 1000])
    def test_quadratic_form_constant_time(self, n):
        """QuadraticForm gradient should be O(1) for VectorVariable."""
        x = VectorVariable("x", n)
        Q = np.eye(n)  # Use identity for fast matrix ops
        expr = QuadraticForm(x, Q)

        start = time.perf_counter()
        for _ in range(100):
            grad = gradient(expr, x[0])  # noqa: F841
        elapsed_ms = (time.perf_counter() - start) * 10

        # QuadraticForm gradient is O(n) due to row extraction, but still fast
        # Allow 5ms for n=1000 (vs minutes for full tree traversal)
        assert elapsed_ms < 5.0, f"QuadraticForm gradient too slow: {elapsed_ms:.3f}ms"


class TestCombinedExpressions:
    """Tests for gradients of combined vector expressions."""

    def test_dot_minus_sum(self):
        """Gradient of x·x - sum(x) should be 2x - 1."""
        n = 10
        x = VectorVariable("x", n)
        expr = x.dot(x) - x.sum()

        variables = list(x)
        grad_fns = [compile_expression(gradient(expr, v), variables) for v in variables]

        x_val = np.random.rand(n)
        computed = np.array([fn(x_val) for fn in grad_fns])

        expected = 2 * x_val - 1
        assert_array_almost_equal(computed, expected)

    def test_linear_plus_quadratic(self):
        """Gradient of c@x + x'Qx should be c + (Q + Q')x."""
        n = 5
        x = VectorVariable("x", n)
        c = np.random.rand(n)
        Q = np.random.rand(n, n)

        expr = (c @ x) + QuadraticForm(x, Q)

        variables = list(x)
        grad_fns = [compile_expression(gradient(expr, v), variables) for v in variables]

        x_val = np.random.rand(n)
        computed = np.array([fn(x_val) for fn in grad_fns])

        expected = c + (Q + Q.T) @ x_val
        assert_array_almost_equal(computed, expected)

    def test_scaled_norm(self):
        """Gradient of 0.5 * ||x||² = 0.5 * x·x should be x."""
        n = 10
        x = VectorVariable("x", n)
        expr = Constant(0.5) * x.dot(x)

        variables = list(x)
        grad_fns = [compile_expression(gradient(expr, v), variables) for v in variables]

        x_val = np.random.rand(n)
        computed = np.array([fn(x_val) for fn in grad_fns])

        # ∂(0.5 * x·x)/∂x = x
        expected = x_val
        assert_array_almost_equal(computed, expected)
