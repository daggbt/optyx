"""Tests for automatic differentiation."""

import numpy as np
import pytest

from optyx import Variable, Constant, sin, cos, exp, log, sqrt, tanh, abs_
from optyx.core.autodiff import (
    gradient,
    compute_jacobian,
    compute_hessian,
    compile_jacobian,
    compile_hessian,
)
from optyx.core.compiler import compile_gradient
from optyx.core.verification import (
    verify_gradient,
    gradient_check,
)

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:divide by zero encountered in scalar divide:RuntimeWarning:optyx.core.compiler"
    ),
    pytest.mark.filterwarnings(
        "ignore:invalid value encountered in scalar divide:RuntimeWarning:optyx.core.compiler"
    ),
]


class TestBasicGradients:
    """Tests for basic differentiation rules."""

    def test_constant_gradient(self):
        x = Variable("x")
        expr = Constant(5.0)
        grad = gradient(expr, x)
        assert grad.evaluate({"x": 1.0}) == 0.0

    def test_variable_gradient_same(self):
        x = Variable("x")
        grad = gradient(x, x)
        assert grad.evaluate({"x": 1.0}) == 1.0

    def test_variable_gradient_different(self):
        x = Variable("x")
        y = Variable("y")
        grad = gradient(x, y)
        assert grad.evaluate({"x": 1.0, "y": 2.0}) == 0.0

    def test_addition_gradient(self):
        x = Variable("x")
        y = Variable("y")
        expr = x + y

        grad_x = gradient(expr, x)
        grad_y = gradient(expr, y)

        assert grad_x.evaluate({"x": 1.0, "y": 2.0}) == 1.0
        assert grad_y.evaluate({"x": 1.0, "y": 2.0}) == 1.0

    def test_subtraction_gradient(self):
        x = Variable("x")
        y = Variable("y")
        expr = x - y

        grad_x = gradient(expr, x)
        grad_y = gradient(expr, y)

        assert grad_x.evaluate({"x": 1.0, "y": 2.0}) == 1.0
        assert grad_y.evaluate({"x": 1.0, "y": 2.0}) == -1.0

    def test_multiplication_gradient(self):
        x = Variable("x")
        y = Variable("y")
        expr = x * y

        grad_x = gradient(expr, x)
        grad_y = gradient(expr, y)

        point = {"x": 3.0, "y": 4.0}
        # d/dx(x*y) = y
        assert grad_x.evaluate(point) == 4.0
        # d/dy(x*y) = x
        assert grad_y.evaluate(point) == 3.0

    def test_division_gradient(self):
        x = Variable("x")
        y = Variable("y")
        expr = x / y

        grad_x = gradient(expr, x)
        grad_y = gradient(expr, y)

        point = {"x": 6.0, "y": 2.0}
        # d/dx(x/y) = 1/y = 0.5
        np.testing.assert_almost_equal(grad_x.evaluate(point), 0.5)
        # d/dy(x/y) = -x/y^2 = -6/4 = -1.5
        np.testing.assert_almost_equal(grad_y.evaluate(point), -1.5)

    def test_power_gradient_constant_exponent(self):
        x = Variable("x")
        expr = x**2

        grad = gradient(expr, x)
        # d/dx(x^2) = 2x
        assert grad.evaluate({"x": 3.0}) == 6.0

    def test_power_gradient_cubic(self):
        x = Variable("x")
        expr = x**3

        grad = gradient(expr, x)
        # d/dx(x^3) = 3x^2
        assert grad.evaluate({"x": 2.0}) == 12.0

    def test_negation_gradient(self):
        x = Variable("x")
        expr = -x

        grad = gradient(expr, x)
        assert grad.evaluate({"x": 5.0}) == -1.0

    def test_linear_combination_gradient(self):
        x = Variable("x")
        y = Variable("y")
        expr = 2 * x + 3 * y

        grad_x = gradient(expr, x)
        grad_y = gradient(expr, y)

        point = {"x": 1.0, "y": 1.0}
        assert grad_x.evaluate(point) == 2.0
        assert grad_y.evaluate(point) == 3.0


class TestTranscendentalGradients:
    """Tests for transcendental function derivatives."""

    def test_sin_gradient(self):
        x = Variable("x")
        expr = sin(x)
        grad = gradient(expr, x)

        # d/dx(sin(x)) = cos(x)
        # At x=0: cos(0) = 1
        np.testing.assert_almost_equal(grad.evaluate({"x": 0.0}), 1.0)
        # At x=π/2: cos(π/2) = 0
        np.testing.assert_almost_equal(grad.evaluate({"x": np.pi / 2}), 0.0, decimal=10)

    def test_cos_gradient(self):
        x = Variable("x")
        expr = cos(x)
        grad = gradient(expr, x)

        # d/dx(cos(x)) = -sin(x)
        # At x=0: -sin(0) = 0
        np.testing.assert_almost_equal(grad.evaluate({"x": 0.0}), 0.0)
        # At x=π/2: -sin(π/2) = -1
        np.testing.assert_almost_equal(grad.evaluate({"x": np.pi / 2}), -1.0)

    def test_exp_gradient(self):
        x = Variable("x")
        expr = exp(x)
        grad = gradient(expr, x)

        # d/dx(exp(x)) = exp(x)
        np.testing.assert_almost_equal(grad.evaluate({"x": 0.0}), 1.0)
        np.testing.assert_almost_equal(grad.evaluate({"x": 1.0}), np.e)

    def test_log_gradient(self):
        x = Variable("x")
        expr = log(x)
        grad = gradient(expr, x)

        # d/dx(log(x)) = 1/x
        np.testing.assert_almost_equal(grad.evaluate({"x": 1.0}), 1.0)
        np.testing.assert_almost_equal(grad.evaluate({"x": 2.0}), 0.5)

    def test_sqrt_gradient(self):
        x = Variable("x")
        expr = sqrt(x)
        grad = gradient(expr, x)

        # d/dx(sqrt(x)) = 1/(2*sqrt(x))
        # At x=4: 1/(2*2) = 0.25
        np.testing.assert_almost_equal(grad.evaluate({"x": 4.0}), 0.25)

    def test_tanh_gradient(self):
        x = Variable("x")
        expr = tanh(x)
        grad = gradient(expr, x)

        # d/dx(tanh(x)) = 1 - tanh^2(x) = sech^2(x)
        # At x=0: 1 - 0 = 1
        np.testing.assert_almost_equal(grad.evaluate({"x": 0.0}), 1.0)


class TestChainRule:
    """Tests for chain rule application."""

    def test_sin_of_squared(self):
        x = Variable("x")
        expr = sin(x**2)
        grad = gradient(expr, x)

        # d/dx(sin(x^2)) = cos(x^2) * 2x
        point = {"x": 1.0}
        expected = np.cos(1.0) * 2.0
        np.testing.assert_almost_equal(grad.evaluate(point), expected)

    def test_exp_of_negative(self):
        x = Variable("x")
        expr = exp(-x)
        grad = gradient(expr, x)

        # d/dx(exp(-x)) = -exp(-x)
        point = {"x": 1.0}
        expected = -np.exp(-1.0)
        np.testing.assert_almost_equal(grad.evaluate(point), expected)

    def test_log_of_sum(self):
        x = Variable("x")
        y = Variable("y")
        expr = log(x + y)

        grad_x = gradient(expr, x)
        grad_y = gradient(expr, y)

        point = {"x": 2.0, "y": 3.0}
        # d/dx(log(x+y)) = 1/(x+y) = 1/5 = 0.2
        np.testing.assert_almost_equal(grad_x.evaluate(point), 0.2)
        np.testing.assert_almost_equal(grad_y.evaluate(point), 0.2)

    def test_composed_functions(self):
        x = Variable("x")
        expr = exp(sin(x))
        grad = gradient(expr, x)

        # d/dx(exp(sin(x))) = exp(sin(x)) * cos(x)
        point = {"x": 0.5}
        expected = np.exp(np.sin(0.5)) * np.cos(0.5)
        np.testing.assert_almost_equal(grad.evaluate(point), expected)

    def test_product_chain_rule(self):
        x = Variable("x")
        expr = x * sin(x)
        grad = gradient(expr, x)

        # d/dx(x*sin(x)) = sin(x) + x*cos(x)
        point = {"x": 1.0}
        expected = np.sin(1.0) + 1.0 * np.cos(1.0)
        np.testing.assert_almost_equal(grad.evaluate(point), expected)


class TestComplexExpressions:
    """Tests for complex multi-variable expressions."""

    def test_rosenbrock_gradient(self):
        """Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2"""
        x = Variable("x")
        y = Variable("y")
        expr = (1 - x) ** 2 + 100 * (y - x**2) ** 2

        grad_x = gradient(expr, x)
        grad_y = gradient(expr, y)

        # At minimum (1, 1), gradients should be 0
        point = {"x": 1.0, "y": 1.0}
        np.testing.assert_almost_equal(grad_x.evaluate(point), 0.0)
        np.testing.assert_almost_equal(grad_y.evaluate(point), 0.0)

        # At (0, 0):
        # df/dx = -2(1-x) - 400*x*(y-x^2) = -2*1 - 0 = -2
        # df/dy = 200*(y-x^2) = 0
        point = {"x": 0.0, "y": 0.0}
        np.testing.assert_almost_equal(grad_x.evaluate(point), -2.0)
        np.testing.assert_almost_equal(grad_y.evaluate(point), 0.0)

    def test_quadratic_form_gradient(self):
        """f(x,y) = x^2 + 2*x*y + 3*y^2"""
        x = Variable("x")
        y = Variable("y")
        expr = x**2 + 2 * x * y + 3 * y**2

        grad_x = gradient(expr, x)
        grad_y = gradient(expr, y)

        point = {"x": 1.0, "y": 2.0}
        # df/dx = 2x + 2y = 2 + 4 = 6
        np.testing.assert_almost_equal(grad_x.evaluate(point), 6.0)
        # df/dy = 2x + 6y = 2 + 12 = 14
        np.testing.assert_almost_equal(grad_y.evaluate(point), 14.0)


class TestNumericalVerification:
    """Tests comparing symbolic gradients to numerical gradients."""

    def test_verify_polynomial(self):
        x = Variable("x")
        expr = x**3 - 2 * x**2 + x - 1

        assert verify_gradient(expr, x, {"x": 2.0})
        assert verify_gradient(expr, x, {"x": -1.5})

    def test_verify_transcendental(self):
        x = Variable("x")
        expr = sin(x) * exp(x)

        assert verify_gradient(expr, x, {"x": 0.5})
        assert verify_gradient(expr, x, {"x": 1.0})

    def test_verify_multivariate(self):
        x = Variable("x")
        y = Variable("y")
        expr = x**2 * y + sin(x * y)

        point = {"x": 1.0, "y": 2.0}
        assert verify_gradient(expr, x, point)
        assert verify_gradient(expr, y, point)

    def test_gradient_check_simple(self):
        x = Variable("x")
        expr = x**2 + 3 * x + 1

        result = gradient_check(expr, [x], n_samples=50, seed=42)
        assert result.all_passed
        assert result.max_error < 1e-5

    def test_gradient_check_multivariate(self):
        x = Variable("x")
        y = Variable("y")
        expr = x**2 + x * y + y**2

        result = gradient_check(expr, [x, y], n_samples=50, seed=42)
        assert result.all_passed


class TestJacobian:
    """Tests for Jacobian computation."""

    def test_jacobian_linear(self):
        x = Variable("x")
        y = Variable("y")

        exprs = [2 * x + 3 * y, x - y]
        jac = compute_jacobian(exprs, [x, y])

        point = {"x": 1.0, "y": 1.0}

        # J = [[2, 3], [1, -1]]
        assert jac[0][0].evaluate(point) == 2.0
        assert jac[0][1].evaluate(point) == 3.0
        assert jac[1][0].evaluate(point) == 1.0
        assert jac[1][1].evaluate(point) == -1.0

    def test_jacobian_nonlinear(self):
        x = Variable("x")
        y = Variable("y")

        exprs = [x**2 + y, x * y]
        jac = compute_jacobian(exprs, [x, y])

        point = {"x": 2.0, "y": 3.0}

        # J = [[2x, 1], [y, x]] at (2,3) = [[4, 1], [3, 2]]
        assert jac[0][0].evaluate(point) == 4.0
        assert jac[0][1].evaluate(point) == 1.0
        assert jac[1][0].evaluate(point) == 3.0
        assert jac[1][1].evaluate(point) == 2.0

    def test_compiled_jacobian(self):
        x = Variable("x")
        y = Variable("y")

        exprs = [x**2 + y, x * y]
        jac_fn = compile_jacobian(exprs, [x, y])

        result = jac_fn(np.array([2.0, 3.0]))

        expected = np.array([[4.0, 1.0], [3.0, 2.0]])
        np.testing.assert_array_almost_equal(result, expected)


class TestHessian:
    """Tests for Hessian computation."""

    def test_hessian_quadratic(self):
        x = Variable("x")
        y = Variable("y")

        # f(x,y) = x^2 + 2*x*y + 3*y^2
        expr = x**2 + 2 * x * y + 3 * y**2
        hess = compute_hessian(expr, [x, y])

        point = {"x": 1.0, "y": 1.0}

        # H = [[2, 2], [2, 6]]
        np.testing.assert_almost_equal(hess[0][0].evaluate(point), 2.0)
        np.testing.assert_almost_equal(hess[0][1].evaluate(point), 2.0)
        np.testing.assert_almost_equal(hess[1][0].evaluate(point), 2.0)
        np.testing.assert_almost_equal(hess[1][1].evaluate(point), 6.0)

    def test_hessian_symmetry(self):
        x = Variable("x")
        y = Variable("y")

        expr = sin(x * y) + x**2 * y
        hess = compute_hessian(expr, [x, y])

        point = {"x": 1.0, "y": 2.0}

        # Hessian should be symmetric
        h01 = hess[0][1].evaluate(point)
        h10 = hess[1][0].evaluate(point)
        np.testing.assert_almost_equal(h01, h10)

    def test_compiled_hessian(self):
        x = Variable("x")
        y = Variable("y")

        expr = x**2 + 2 * x * y + 3 * y**2
        hess_fn = compile_hessian(expr, [x, y])

        result = hess_fn(np.array([1.0, 1.0]))

        expected = np.array([[2.0, 2.0], [2.0, 6.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_compiled_hessian_symmetry(self):
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")

        expr = x * y + y * z + x * z + x**2 + y**2 + z**2
        hess_fn = compile_hessian(expr, [x, y, z])

        result = hess_fn(np.array([1.0, 2.0, 3.0]))

        # Check symmetry
        np.testing.assert_array_almost_equal(result, result.T)


class TestSimplification:
    """Tests for expression simplification in gradients."""

    def test_zero_simplification(self):
        x = Variable("x")
        y = Variable("y")

        # Gradient of y w.r.t. x should be 0, not "0 + 0 + 0..."
        grad = gradient(y, x)
        # It should evaluate to 0
        assert grad.evaluate({"x": 1.0, "y": 2.0}) == 0.0

    def test_one_simplification(self):
        x = Variable("x")

        # d/dx(x) = 1, should be a constant
        grad = gradient(x, x)
        assert grad.evaluate({"x": 100.0}) == 1.0

    def test_zero_times_elimination(self):
        x = Variable("x")
        y = Variable("y")

        # d/dx(y^2) should simplify nicely
        expr = y**2
        grad = gradient(expr, x)
        assert grad.evaluate({"x": 1.0, "y": 5.0}) == 0.0


class TestGradientCaching:
    """Tests for gradient caching."""

    def test_same_gradient_cached(self):
        x = Variable("x")
        expr = x**2 + sin(x)

        # Call gradient twice
        grad1 = gradient(expr, x)
        grad2 = gradient(expr, x)

        # Both should work and give same results
        point = {"x": 1.0}
        assert grad1.evaluate(point) == grad2.evaluate(point)


class TestSingularityHandling:
    """Tests for handling derivative singularities at x=0."""

    def test_abs_gradient_at_zero_returns_finite(self):
        """d/dx(|x|) = x/|x| which is 0/0 at x=0. Should return 0 (subgradient)."""
        x = Variable("x")
        expr = abs_(x)
        grad_fn = compile_gradient(expr, [x])

        result = grad_fn(np.array([0.0]))

        # Should be finite (0.0), not NaN
        assert np.isfinite(result[0])
        assert result[0] == 0.0

    def test_sqrt_gradient_at_zero_returns_large_finite(self):
        """d/dx(sqrt(x)) = 1/(2*sqrt(x)) which is +Inf at x=0."""
        x = Variable("x")
        expr = sqrt(x)
        grad_fn = compile_gradient(expr, [x])

        result = grad_fn(np.array([0.0]))

        # Should be finite (large positive), not Inf
        assert np.isfinite(result[0])
        assert result[0] > 0  # Preserves direction

    def test_log_gradient_at_zero_returns_large_finite(self):
        """d/dx(log(x)) = 1/x which is +Inf at x=0."""
        x = Variable("x")
        expr = log(x)
        grad_fn = compile_gradient(expr, [x])

        result = grad_fn(np.array([0.0]))

        # Should be finite (large positive), not Inf
        assert np.isfinite(result[0])
        assert result[0] > 0  # Preserves direction

    def test_jacobian_at_singularity_returns_finite(self):
        """Jacobian should also handle singularities."""
        x = Variable("x")
        exprs = [sqrt(x), log(x)]
        jac_fn = compile_jacobian(exprs, [x])

        result = jac_fn(np.array([0.0]))

        # All values should be finite
        assert np.all(np.isfinite(result))

    def test_hessian_at_singularity_returns_finite(self):
        """Hessian should also handle singularities."""
        x = Variable("x")
        expr = sqrt(x)  # Second derivative also has singularity at 0
        hess_fn = compile_hessian(expr, [x])

        result = hess_fn(np.array([0.0]))

        # All values should be finite
        assert np.all(np.isfinite(result))

    def test_optimization_through_singularity_does_not_crash(self):
        """Solver should handle starting near singularities."""
        from optyx import Problem

        x = Variable("x", lb=0.01)  # Avoid exact zero
        y = Variable("y")

        # Objective with sqrt - could hit near-singularity
        prob = Problem().minimize((sqrt(x) - 1) ** 2 + y**2)

        # Should not crash
        sol = prob.solve()

        assert sol.is_optimal
        # sqrt(x) = 1 means x = 1
        assert abs(sol["x"] - 1.0) < 0.1
        assert abs(sol["y"]) < 0.1
