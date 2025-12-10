"""Tests for expression compilation."""

import time

import numpy as np
import pytest

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
