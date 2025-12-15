"""Tests for expression compilation."""

import time

import numpy as np

from optyx import Variable, sin, cos, exp, log, sqrt
from optyx.core.compiler import (
    compile_expression,
    compile_to_dict_function,
    compile_gradient,
    CompiledExpression,
)


class TestCompileExpression:
    """Tests for compile_expression function."""

    def test_simple_expression(self):
        x = Variable("x")
        y = Variable("y")
        expr = x + y
        
        f = compile_expression(expr, [x, y])
        result = f(np.array([3.0, 4.0]))
        assert result == 7.0

    def test_complex_expression(self):
        x = Variable("x")
        y = Variable("y")
        expr = 2*x + 3*y**2 - x*y
        
        f = compile_expression(expr, [x, y])
        result = f(np.array([1.0, 2.0]))
        # 2*1 + 3*4 - 1*2 = 2 + 12 - 2 = 12
        assert result == 12.0

    def test_transcendental_expression(self):
        x = Variable("x")
        expr = sin(x)**2 + cos(x)**2
        
        f = compile_expression(expr, [x])
        result = f(np.array([1.234]))
        np.testing.assert_almost_equal(result, 1.0)

    def test_nested_expression(self):
        x = Variable("x")
        expr = exp(log(x))
        
        f = compile_expression(expr, [x])
        result = f(np.array([5.0]))
        np.testing.assert_almost_equal(result, 5.0)

    def test_variable_order_matters(self):
        x = Variable("x")
        y = Variable("y")
        expr = x - y
        
        f_xy = compile_expression(expr, [x, y])
        f_yx = compile_expression(expr, [y, x])
        
        # [3, 4] means x=3, y=4 for f_xy -> 3 - 4 = -1
        assert f_xy(np.array([3.0, 4.0])) == -1.0
        # [3, 4] means y=3, x=4 for f_yx -> 4 - 3 = 1
        assert f_yx(np.array([3.0, 4.0])) == 1.0

    def test_constant_expression(self):
        x = Variable("x")
        expr = x + 5
        
        f = compile_expression(expr, [x])
        assert f(np.array([3.0])) == 8.0

    def test_single_variable(self):
        x = Variable("x")
        expr = x**2
        
        f = compile_expression(expr, [x])
        assert f(np.array([4.0])) == 16.0


class TestCompileToDictFunction:
    """Tests for compile_to_dict_function."""

    def test_dict_function(self):
        x = Variable("x")
        y = Variable("y")
        expr = x * y
        
        f = compile_to_dict_function(expr, [x, y])
        result = f({"x": 3.0, "y": 4.0})
        assert result == 12.0

    def test_matches_evaluate(self):
        x = Variable("x")
        y = Variable("y")
        expr = 2*x**2 + 3*y - x*y
        
        f = compile_to_dict_function(expr, [x, y])
        values = {"x": 2.5, "y": 1.5}
        
        compiled_result = f(values)
        direct_result = expr.evaluate(values)
        
        np.testing.assert_almost_equal(compiled_result, direct_result)


class TestCompileGradient:
    """Tests for compile_gradient (numerical gradient)."""

    def test_linear_gradient(self):
        x = Variable("x")
        y = Variable("y")
        expr = 2*x + 3*y
        
        grad_fn = compile_gradient(expr, [x, y])
        grad = grad_fn(np.array([1.0, 1.0]))
        
        np.testing.assert_array_almost_equal(grad, [2.0, 3.0])

    def test_quadratic_gradient(self):
        x = Variable("x")
        y = Variable("y")
        expr = x**2 + y**2
        
        grad_fn = compile_gradient(expr, [x, y])
        grad = grad_fn(np.array([3.0, 4.0]))
        
        # d/dx = 2x = 6, d/dy = 2y = 8
        np.testing.assert_array_almost_equal(grad, [6.0, 8.0])

    def test_transcendental_gradient(self):
        x = Variable("x")
        expr = sin(x)
        
        grad_fn = compile_gradient(expr, [x])
        grad = grad_fn(np.array([0.0]))
        
        # d/dx sin(x) at x=0 is cos(0) = 1
        np.testing.assert_array_almost_equal(grad, [1.0], decimal=5)

    def test_product_rule_gradient(self):
        x = Variable("x")
        y = Variable("y")
        expr = x * y
        
        grad_fn = compile_gradient(expr, [x, y])
        grad = grad_fn(np.array([3.0, 4.0]))
        
        # d/dx = y = 4, d/dy = x = 3
        np.testing.assert_array_almost_equal(grad, [4.0, 3.0])


class TestCompiledExpression:
    """Tests for CompiledExpression class."""

    def test_value(self):
        x = Variable("x")
        y = Variable("y")
        expr = x**2 + y**2
        
        compiled = CompiledExpression(expr, [x, y])
        value = compiled.value(np.array([3.0, 4.0]))
        
        assert value == 25.0

    def test_gradient(self):
        x = Variable("x")
        y = Variable("y")
        expr = x**2 + y**2
        
        compiled = CompiledExpression(expr, [x, y])
        grad = compiled.gradient(np.array([3.0, 4.0]))
        
        np.testing.assert_array_almost_equal(grad, [6.0, 8.0])

    def test_value_and_gradient(self):
        x = Variable("x")
        y = Variable("y")
        expr = x**2 + y**2
        
        compiled = CompiledExpression(expr, [x, y])
        value, grad = compiled.value_and_gradient(np.array([3.0, 4.0]))
        
        assert value == 25.0
        np.testing.assert_array_almost_equal(grad, [6.0, 8.0])

    def test_n_variables(self):
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")
        expr = x + y + z
        
        compiled = CompiledExpression(expr, [x, y, z])
        assert compiled.n_variables == 3

    def test_variable_names(self):
        x = Variable("x")
        y = Variable("y")
        expr = x + y
        
        compiled = CompiledExpression(expr, [x, y])
        assert compiled.variable_names == ["x", "y"]


class TestCompilationCaching:
    """Tests for LRU caching of compiled functions."""

    def test_same_expression_cached(self):
        x = Variable("x")
        expr = x**2
        
        f1 = compile_expression(expr, [x])
        f2 = compile_expression(expr, [x])
        
        # Both should work
        assert f1(np.array([3.0])) == 9.0
        assert f2(np.array([3.0])) == 9.0


class TestPerformance:
    """Performance benchmarks for compiled vs tree-walk evaluation."""

    def test_compiled_faster_than_evaluate(self):
        """Compiled evaluation should be faster for repeated calls."""
        x = Variable("x")
        y = Variable("y")
        expr = 2*x**2 + 3*y**2 + sin(x*y) + exp(-x) * log(y + 1)
        
        f = compile_expression(expr, [x, y])
        n_iterations = 10000
        
        # Benchmark compiled
        start = time.perf_counter()
        for _ in range(n_iterations):
            f(np.array([1.5, 2.5]))
        compiled_time = time.perf_counter() - start
        
        # Benchmark tree-walk evaluate
        start = time.perf_counter()
        for _ in range(n_iterations):
            expr.evaluate({"x": 1.5, "y": 2.5})
        evaluate_time = time.perf_counter() - start
        
        # Compiled should be faster (or at least not much slower)
        # We're lenient here since the test environment may vary
        assert compiled_time < evaluate_time * 2, (
            f"Compiled ({compiled_time:.4f}s) should be faster than "
            f"evaluate ({evaluate_time:.4f}s)"
        )

    def test_compiled_correctness_matches_evaluate(self):
        """Compiled and evaluate should give identical results."""
        x = Variable("x")
        y = Variable("y")
        expr = 2*x**2 + 3*y**2 + sin(x*y) + exp(-x) * log(y + 1)
        
        f = compile_expression(expr, [x, y])
        
        # Test at multiple points
        test_points = [
            (1.0, 2.0),
            (0.5, 0.5),
            (3.0, 1.0),
            (-1.0, 2.0),
        ]
        
        for x_val, y_val in test_points:
            compiled_result = f(np.array([x_val, y_val]))
            evaluate_result = expr.evaluate({"x": x_val, "y": y_val})
            np.testing.assert_almost_equal(
                compiled_result, evaluate_result,
                err_msg=f"Mismatch at x={x_val}, y={y_val}"
            )


class TestSymbolicGradient:
    """Tests verifying compile_gradient uses exact symbolic differentiation."""

    def test_symbolic_gradient_exact_polynomial(self):
        """Symbolic gradient should match analytical derivative exactly."""
        x = Variable("x")
        # f(x) = x^3 + 2x^2 - 5x + 3
        # f'(x) = 3x^2 + 4x - 5
        expr = x**3 + 2*x**2 - 5*x + 3
        
        grad_fn = compile_gradient(expr, [x])
        
        # Test at multiple points - should be exact (not numerical approximation)
        test_points = [0.0, 1.0, 2.0, -1.0, 0.5, 10.0]
        for x_val in test_points:
            grad = grad_fn(np.array([x_val]))
            expected = 3*x_val**2 + 4*x_val - 5
            # Use high precision - symbolic should be exact
            np.testing.assert_almost_equal(
                grad[0], expected, decimal=12,
                err_msg=f"Gradient mismatch at x={x_val}"
            )

    def test_symbolic_gradient_multivariate(self):
        """Symbolic gradient for multivariate expression."""
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")
        # f(x,y,z) = x^2*y + y^2*z + z^2*x
        # df/dx = 2xy + z^2
        # df/dy = x^2 + 2yz
        # df/dz = y^2 + 2zx
        expr = x**2 * y + y**2 * z + z**2 * x
        
        grad_fn = compile_gradient(expr, [x, y, z])
        
        x_val, y_val, z_val = 2.0, 3.0, 4.0
        grad = grad_fn(np.array([x_val, y_val, z_val]))
        
        expected_dx = 2*x_val*y_val + z_val**2  # 2*2*3 + 16 = 28
        expected_dy = x_val**2 + 2*y_val*z_val  # 4 + 24 = 28
        expected_dz = y_val**2 + 2*z_val*x_val  # 9 + 16 = 25
        
        np.testing.assert_almost_equal(grad[0], expected_dx, decimal=12)
        np.testing.assert_almost_equal(grad[1], expected_dy, decimal=12)
        np.testing.assert_almost_equal(grad[2], expected_dz, decimal=12)

    def test_symbolic_gradient_transcendental(self):
        """Symbolic gradient for transcendental functions."""
        x = Variable("x")
        # f(x) = sin(x) * exp(x)
        # f'(x) = cos(x)*exp(x) + sin(x)*exp(x) = exp(x)*(cos(x) + sin(x))
        expr = sin(x) * exp(x)
        
        grad_fn = compile_gradient(expr, [x])
        
        test_points = [0.0, 0.5, 1.0, np.pi/4]
        for x_val in test_points:
            grad = grad_fn(np.array([x_val]))
            expected = np.exp(x_val) * (np.cos(x_val) + np.sin(x_val))
            np.testing.assert_almost_equal(
                grad[0], expected, decimal=10,
                err_msg=f"Gradient mismatch at x={x_val}"
            )

    def test_symbolic_gradient_log_sqrt(self):
        """Symbolic gradient for log and sqrt functions."""
        x = Variable("x")
        # f(x) = log(x) + sqrt(x)
        # f'(x) = 1/x + 1/(2*sqrt(x))
        expr = log(x) + sqrt(x)
        
        grad_fn = compile_gradient(expr, [x])
        
        test_points = [0.5, 1.0, 2.0, 4.0, 10.0]
        for x_val in test_points:
            grad = grad_fn(np.array([x_val]))
            expected = 1/x_val + 1/(2*np.sqrt(x_val))
            np.testing.assert_almost_equal(
                grad[0], expected, decimal=10,
                err_msg=f"Gradient mismatch at x={x_val}"
            )

    def test_symbolic_gradient_chain_rule(self):
        """Symbolic gradient correctly applies chain rule."""
        x = Variable("x")
        # f(x) = sin(x^2)
        # f'(x) = cos(x^2) * 2x
        expr = sin(x**2)
        
        grad_fn = compile_gradient(expr, [x])
        
        test_points = [0.5, 1.0, 1.5, 2.0]
        for x_val in test_points:
            grad = grad_fn(np.array([x_val]))
            expected = np.cos(x_val**2) * 2 * x_val
            np.testing.assert_almost_equal(
                grad[0], expected, decimal=10,
                err_msg=f"Gradient mismatch at x={x_val}"
            )

    def test_symbolic_gradient_quotient_rule(self):
        """Symbolic gradient correctly applies quotient rule."""
        x = Variable("x")
        y = Variable("y")
        # f(x,y) = x / y
        # df/dx = 1/y
        # df/dy = -x/y^2
        expr = x / y
        
        grad_fn = compile_gradient(expr, [x, y])
        
        x_val, y_val = 3.0, 2.0
        grad = grad_fn(np.array([x_val, y_val]))
        
        expected_dx = 1 / y_val  # 0.5
        expected_dy = -x_val / y_val**2  # -0.75
        
        np.testing.assert_almost_equal(grad[0], expected_dx, decimal=12)
        np.testing.assert_almost_equal(grad[1], expected_dy, decimal=12)

    def test_symbolic_vs_autodiff_consistency(self):
        """compile_gradient should match autodiff.gradient evaluation."""
        from optyx.core.autodiff import gradient
        
        x = Variable("x")
        y = Variable("y")
        expr = x**2 * sin(y) + exp(x*y)
        
        # Get symbolic gradient expressions
        grad_x_expr = gradient(expr, x)
        grad_y_expr = gradient(expr, y)
        
        # Get compiled gradient function
        grad_fn = compile_gradient(expr, [x, y])
        
        # Compare at several points
        test_points = [(1.0, 2.0), (0.5, 0.5), (2.0, 1.0)]
        for x_val, y_val in test_points:
            values = {"x": x_val, "y": y_val}
            
            # Evaluate symbolic gradient expressions directly
            expected_dx = grad_x_expr.evaluate(values)
            expected_dy = grad_y_expr.evaluate(values)
            
            # Get compiled gradient
            grad = grad_fn(np.array([x_val, y_val]))
            
            np.testing.assert_almost_equal(grad[0], expected_dx, decimal=12)
            np.testing.assert_almost_equal(grad[1], expected_dy, decimal=12)

    def test_gradient_of_constant(self):
        """Gradient of a constant expression should be zero."""
        x = Variable("x")
        from optyx.core.expressions import Constant
        expr = Constant(42.0)
        
        grad_fn = compile_gradient(expr, [x])
        grad = grad_fn(np.array([5.0]))
        
        np.testing.assert_almost_equal(grad[0], 0.0, decimal=12)

    def test_gradient_of_unrelated_variable(self):
        """Gradient w.r.t. unrelated variable should be zero."""
        x = Variable("x")
        y = Variable("y")
        expr = x**2  # Only depends on x
        
        grad_fn = compile_gradient(expr, [x, y])
        grad = grad_fn(np.array([3.0, 5.0]))
        
        # df/dx = 2x = 6
        np.testing.assert_almost_equal(grad[0], 6.0, decimal=12)
        # df/dy = 0
        np.testing.assert_almost_equal(grad[1], 0.0, decimal=12)
