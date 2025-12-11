"""Tests for the core expression system."""

import numpy as np
import pytest

from optyx import Variable, Constant, sin, cos, exp, log, sqrt, abs_, tanh


class TestConstant:
    """Tests for Constant expressions."""

    def test_scalar_constant(self):
        c = Constant(5.0)
        assert c.evaluate({}) == 5.0

    def test_integer_constant(self):
        c = Constant(3)
        assert c.evaluate({}) == 3

    def test_array_constant(self):
        arr = np.array([1.0, 2.0, 3.0])
        c = Constant(arr)
        result = c.evaluate({})
        np.testing.assert_array_equal(result, arr)

    def test_constant_has_no_variables(self):
        c = Constant(5.0)
        assert c.get_variables() == set()


class TestVariable:
    """Tests for Variable expressions."""

    def test_variable_evaluation(self):
        x = Variable("x")
        assert x.evaluate({"x": 3.0}) == 3.0

    def test_variable_with_bounds(self):
        x = Variable("x", lb=0, ub=10)
        assert x.lb == 0
        assert x.ub == 10

    def test_binary_variable(self):
        b = Variable("b", domain="binary")
        assert b.domain == "binary"
        assert b.lb == 0.0
        assert b.ub == 1.0

    def test_integer_variable(self):
        i = Variable("i", domain="integer")
        assert i.domain == "integer"

    def test_missing_variable_raises(self):
        x = Variable("x")
        with pytest.raises(KeyError):
            x.evaluate({"y": 1.0})

    def test_variable_get_variables(self):
        x = Variable("x")
        assert x.get_variables() == {x}

    def test_variable_equality(self):
        x1 = Variable("x")
        x2 = Variable("x")
        x3 = Variable("y")
        assert x1 == x2
        assert x1 != x3

    def test_variable_hash(self):
        x1 = Variable("x")
        x2 = Variable("x")
        assert hash(x1) == hash(x2)


class TestBinaryOperations:
    """Tests for binary operations."""

    def test_addition(self):
        x = Variable("x")
        expr = x + 2
        assert expr.evaluate({"x": 3.0}) == 5.0

    def test_reverse_addition(self):
        x = Variable("x")
        expr = 2 + x
        assert expr.evaluate({"x": 3.0}) == 5.0

    def test_subtraction(self):
        x = Variable("x")
        expr = x - 2
        assert expr.evaluate({"x": 5.0}) == 3.0

    def test_reverse_subtraction(self):
        x = Variable("x")
        expr = 10 - x
        assert expr.evaluate({"x": 3.0}) == 7.0

    def test_multiplication(self):
        x = Variable("x")
        expr = x * 3
        assert expr.evaluate({"x": 4.0}) == 12.0

    def test_reverse_multiplication(self):
        x = Variable("x")
        expr = 3 * x
        assert expr.evaluate({"x": 4.0}) == 12.0

    def test_division(self):
        x = Variable("x")
        expr = x / 2
        assert expr.evaluate({"x": 10.0}) == 5.0

    def test_reverse_division(self):
        x = Variable("x")
        expr = 12 / x
        assert expr.evaluate({"x": 4.0}) == 3.0

    def test_power(self):
        x = Variable("x")
        expr = x ** 2
        assert expr.evaluate({"x": 3.0}) == 9.0

    def test_reverse_power(self):
        x = Variable("x")
        expr = 2 ** x
        assert expr.evaluate({"x": 3.0}) == 8.0

    def test_negation(self):
        x = Variable("x")
        expr = -x
        assert expr.evaluate({"x": 5.0}) == -5.0

    def test_complex_expression(self):
        x = Variable("x")
        y = Variable("y")
        expr = 2*x + 3*y**2 - x*y
        # 2*1 + 3*4 - 1*2 = 2 + 12 - 2 = 12
        assert expr.evaluate({"x": 1.0, "y": 2.0}) == 12.0

    def test_nested_expression(self):
        x = Variable("x")
        y = Variable("y")
        expr = (x + y) * (x - y)  # x² - y²
        assert expr.evaluate({"x": 5.0, "y": 3.0}) == 16.0  # 25 - 9

    def test_get_variables_from_expression(self):
        x = Variable("x")
        y = Variable("y")
        expr = x + y * 2
        variables = expr.get_variables()
        assert x in variables
        assert y in variables
        assert len(variables) == 2


class TestTranscendentalFunctions:
    """Tests for transcendental functions."""

    def test_sin(self):
        x = Variable("x")
        expr = sin(x)
        result = expr.evaluate({"x": np.pi / 2})
        np.testing.assert_almost_equal(result, 1.0)

    def test_cos(self):
        x = Variable("x")
        expr = cos(x)
        result = expr.evaluate({"x": 0.0})
        np.testing.assert_almost_equal(result, 1.0)

    def test_exp(self):
        x = Variable("x")
        expr = exp(x)
        result = expr.evaluate({"x": 0.0})
        np.testing.assert_almost_equal(result, 1.0)

    def test_log(self):
        x = Variable("x")
        expr = log(x)
        result = expr.evaluate({"x": np.e})
        np.testing.assert_almost_equal(result, 1.0)

    def test_sqrt(self):
        x = Variable("x")
        expr = sqrt(x)
        result = expr.evaluate({"x": 4.0})
        np.testing.assert_almost_equal(result, 2.0)

    def test_abs(self):
        x = Variable("x")
        expr = abs_(x)
        assert expr.evaluate({"x": -5.0}) == 5.0
        assert expr.evaluate({"x": 5.0}) == 5.0

    def test_tanh(self):
        x = Variable("x")
        expr = tanh(x)
        result = expr.evaluate({"x": 0.0})
        np.testing.assert_almost_equal(result, 0.0)

    def test_composed_functions(self):
        x = Variable("x")
        expr = sin(x) ** 2 + cos(x) ** 2
        # sin²(x) + cos²(x) = 1 for all x
        result = expr.evaluate({"x": 1.234})
        np.testing.assert_almost_equal(result, 1.0)

    def test_function_of_expression(self):
        x = Variable("x")
        y = Variable("y")
        expr = exp(-x**2 - y**2)
        result = expr.evaluate({"x": 0.0, "y": 0.0})
        np.testing.assert_almost_equal(result, 1.0)


class TestVectorizedEvaluation:
    """Tests for vectorized evaluation with numpy arrays."""

    def test_vector_variable(self):
        x = Variable("x")
        expr = x ** 2
        values = np.array([1.0, 2.0, 3.0, 4.0])
        result = expr.evaluate({"x": values})
        expected = np.array([1.0, 4.0, 9.0, 16.0])
        np.testing.assert_array_equal(result, expected)

    def test_vector_binary_op(self):
        x = Variable("x")
        y = Variable("y")
        expr = x + y
        x_vals = np.array([1.0, 2.0, 3.0])
        y_vals = np.array([10.0, 20.0, 30.0])
        result = expr.evaluate({"x": x_vals, "y": y_vals})
        expected = np.array([11.0, 22.0, 33.0])
        np.testing.assert_array_equal(result, expected)

    def test_vector_transcendental(self):
        x = Variable("x")
        expr = sin(x)
        values = np.array([0.0, np.pi/2, np.pi])
        result = expr.evaluate({"x": values})
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)


class TestEdgeCases:
    """Tests for edge cases and special values."""

    @pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
    def test_division_by_zero(self):
        x = Variable("x")
        expr = 1 / x
        result = expr.evaluate({"x": 0.0})
        assert np.isinf(result)

    @pytest.mark.filterwarnings("ignore:divide by zero:RuntimeWarning")
    def test_log_of_zero(self):
        x = Variable("x")
        expr = log(x)
        result = expr.evaluate({"x": 0.0})
        assert np.isneginf(result)

    @pytest.mark.filterwarnings("ignore:invalid value:RuntimeWarning")
    def test_sqrt_of_negative(self):
        x = Variable("x")
        expr = sqrt(x)
        result = expr.evaluate({"x": -1.0})
        assert np.isnan(result)

    def test_zero_times_expression(self):
        x = Variable("x")
        expr = 0 * x
        assert expr.evaluate({"x": 100.0}) == 0.0

    def test_expression_plus_zero(self):
        x = Variable("x")
        expr = x + 0
        assert expr.evaluate({"x": 5.0}) == 5.0

    def test_expression_times_one(self):
        x = Variable("x")
        expr = x * 1
        assert expr.evaluate({"x": 7.0}) == 7.0

    def test_expression_to_power_zero(self):
        x = Variable("x")
        expr = x ** 0
        assert expr.evaluate({"x": 5.0}) == 1.0

    def test_expression_to_power_one(self):
        x = Variable("x")
        expr = x ** 1
        assert expr.evaluate({"x": 5.0}) == 5.0
