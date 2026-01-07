"""Tests for Parameter and VectorParameter."""

import numpy as np
import pytest

from optyx.core.errors import InvalidSizeError, ShapeMismatchError
from optyx.core.parameters import Parameter, VectorParameter
from optyx.core.expressions import Variable, Expression
from optyx.core.vectors import VectorVariable


# =============================================================================
# Parameter Creation Tests
# =============================================================================


class TestParameterCreation:
    """Tests for Parameter creation."""

    def test_basic_creation(self):
        """Parameter creates with name and value."""
        p = Parameter("price", value=100)
        assert p.name == "price"
        assert p.value == 100.0

    def test_default_value(self):
        """Default value is 0.0."""
        p = Parameter("p")
        assert p.value == 0.0

    def test_integer_value(self):
        """Integer value is converted to float."""
        p = Parameter("p", value=42)
        assert p.value == 42.0
        assert isinstance(p.value, float)

    def test_array_value(self):
        """Array values are supported."""
        p = Parameter("p", value=[1, 2, 3])
        np.testing.assert_array_equal(p.value, [1, 2, 3])

    def test_is_expression(self):
        """Parameter is an Expression."""
        p = Parameter("p", value=10)
        assert isinstance(p, Expression)


class TestParameterSet:
    """Tests for Parameter.set() method."""

    def test_set_updates_value(self):
        """set() updates the parameter value."""
        p = Parameter("price", value=100)
        p.set(120)
        assert p.value == 120.0

    def test_set_integer(self):
        """set() converts integer to float."""
        p = Parameter("p", value=10)
        p.set(20)
        assert p.value == 20.0

    def test_set_array(self):
        """set() works with arrays."""
        p = Parameter("p", value=[1, 2, 3])
        p.set([4, 5, 6])
        np.testing.assert_array_equal(p.value, [4, 5, 6])

    def test_set_array_shape_mismatch(self):
        """set() raises on shape mismatch."""
        p = Parameter("p", value=[1, 2, 3])
        with pytest.raises(ValueError, match="(?i)shape mismatch"):
            p.set([1, 2])

    def test_set_scalar_to_array_raises(self):
        """Cannot change scalar to array."""
        p = Parameter("p", value=10)
        with pytest.raises(ValueError, match="(?i)cannot change scalar"):
            p.set([1, 2, 3])


class TestParameterEvaluate:
    """Tests for Parameter.evaluate()."""

    def test_evaluate_returns_value(self):
        """evaluate() returns current value."""
        p = Parameter("p", value=42)
        result = p.evaluate({})
        assert result == 42.0

    def test_evaluate_ignores_values_dict(self):
        """evaluate() ignores the values dict."""
        p = Parameter("p", value=42)
        result = p.evaluate({"p": 100, "other": 200})
        assert result == 42.0

    def test_evaluate_after_set(self):
        """evaluate() returns updated value after set()."""
        p = Parameter("p", value=10)
        p.set(20)
        result = p.evaluate({})
        assert result == 20.0


class TestParameterGetVariables:
    """Tests for Parameter.get_variables()."""

    def test_get_variables_empty(self):
        """Parameters have no variables."""
        p = Parameter("p", value=10)
        assert p.get_variables() == set()


class TestParameterEquality:
    """Tests for Parameter equality and hashing."""

    def test_equality_by_name(self):
        """Parameters with same name are equal."""
        p1 = Parameter("price", value=100)
        p2 = Parameter("price", value=200)  # Different value
        assert p1 == p2

    def test_inequality_different_names(self):
        """Parameters with different names are not equal."""
        p1 = Parameter("price", value=100)
        p2 = Parameter("cost", value=100)  # Same value
        assert p1 != p2

    def test_hash_by_name(self):
        """Parameters hash by name."""
        p1 = Parameter("price", value=100)
        p2 = Parameter("price", value=200)
        assert hash(p1) == hash(p2)

    def test_usable_in_set(self):
        """Parameters can be used in sets."""
        p1 = Parameter("a", value=1)
        p2 = Parameter("b", value=2)
        p3 = Parameter("a", value=3)  # Same name as p1
        s = {p1, p2, p3}
        assert len(s) == 2  # p1 and p3 are same


class TestParameterRepr:
    """Tests for Parameter repr."""

    def test_repr_basic(self):
        """Parameter has readable repr."""
        p = Parameter("price", value=100)
        r = repr(p)
        assert "Parameter" in r
        assert "price" in r
        assert "100" in r


# =============================================================================
# Parameter Arithmetic Tests
# =============================================================================


class TestParameterArithmetic:
    """Tests for Parameter in expressions."""

    def test_add_parameter_constant(self):
        """Parameter + constant."""
        p = Parameter("p", value=10)
        expr = p + 5
        assert expr.evaluate({}) == 15.0

    def test_add_constant_parameter(self):
        """constant + Parameter."""
        p = Parameter("p", value=10)
        expr = 5 + p
        assert expr.evaluate({}) == 15.0

    def test_multiply_parameter_variable(self):
        """Parameter * Variable."""
        p = Parameter("price", value=100)
        x = Variable("x")
        expr = p * x
        result = expr.evaluate({"x": 5})
        assert result == 500.0

    def test_subtract_parameter(self):
        """Parameter subtraction."""
        p = Parameter("p", value=10)
        expr = p - 3
        assert expr.evaluate({}) == 7.0

    def test_divide_by_parameter(self):
        """Division by Parameter."""
        p = Parameter("p", value=2)
        x = Variable("x")
        expr = x / p
        result = expr.evaluate({"x": 10})
        assert result == 5.0

    def test_power_parameter(self):
        """Parameter power."""
        p = Parameter("p", value=2)
        expr = p**3
        assert expr.evaluate({}) == 8.0

    def test_negative_parameter(self):
        """Negation of Parameter."""
        p = Parameter("p", value=10)
        expr = -p
        assert expr.evaluate({}) == -10.0

    def test_complex_expression(self):
        """Parameter in complex expression."""
        price = Parameter("price", value=100)
        cost = Parameter("cost", value=30)
        x = Variable("x")

        profit = (price - cost) * x
        result = profit.evaluate({"x": 10})
        assert result == 700.0  # (100 - 30) * 10

    def test_parameter_updated_before_evaluate(self):
        """Updated parameter value used in evaluate."""
        price = Parameter("price", value=100)
        x = Variable("x")
        expr = price * x

        # First evaluation
        assert expr.evaluate({"x": 5}) == 500.0

        # Update parameter
        price.set(120)

        # Second evaluation uses new value
        assert expr.evaluate({"x": 5}) == 600.0


class TestParameterConstraints:
    """Tests for Parameter in constraints."""

    def test_parameter_upper_bound(self):
        """x <= Parameter constraint."""
        x = Variable("x", lb=0)
        limit = Parameter("limit", value=10)

        constraint = x <= limit
        assert constraint is not None

    def test_parameter_lower_bound(self):
        """Parameter <= x constraint."""
        x = Variable("x")
        min_val = Parameter("min", value=5)

        constraint = min_val <= x
        assert constraint is not None

    def test_parameter_in_constraint_expression(self):
        """Parameter in constraint expression."""
        x = Variable("x")
        rate = Parameter("rate", value=2)

        constraint = rate * x <= 100
        assert constraint is not None


# =============================================================================
# VectorParameter Tests
# =============================================================================


class TestVectorParameterCreation:
    """Tests for VectorParameter creation."""

    def test_basic_creation(self):
        """VectorParameter creates with size."""
        vp = VectorParameter("prices", 5)
        assert vp.name == "prices"
        assert vp.size == 5
        assert len(vp) == 5

    def test_with_scalar_value(self):
        """VectorParameter with scalar fills all elements."""
        vp = VectorParameter("p", 3, values=10)
        np.testing.assert_array_equal(vp.get_values(), [10, 10, 10])

    def test_with_array_values(self):
        """VectorParameter with array values."""
        vp = VectorParameter("p", 3, values=[1, 2, 3])
        np.testing.assert_array_equal(vp.get_values(), [1, 2, 3])

    def test_default_values_zero(self):
        """Default values are zeros."""
        vp = VectorParameter("p", 3)
        np.testing.assert_array_equal(vp.get_values(), [0, 0, 0])

    def test_zero_size_raises(self):
        """Zero size raises InvalidSizeError."""
        with pytest.raises(InvalidSizeError, match="positive"):
            VectorParameter("p", 0)

    def test_negative_size_raises(self):
        """Negative size raises InvalidSizeError."""
        with pytest.raises(InvalidSizeError, match="positive"):
            VectorParameter("p", -1)

    def test_values_shape_mismatch_raises(self):
        """Values shape mismatch raises ShapeMismatchError."""
        with pytest.raises(ShapeMismatchError, match="(?i)shape mismatch"):
            VectorParameter("p", 3, values=[1, 2])


class TestVectorParameterIndexing:
    """Tests for VectorParameter indexing."""

    def test_getitem_returns_parameter(self):
        """Indexing returns Parameter."""
        vp = VectorParameter("p", 3, values=[10, 20, 30])
        assert isinstance(vp[0], Parameter)
        assert vp[0].value == 10

    def test_getitem_all_indices(self):
        """All indices work correctly."""
        vp = VectorParameter("p", 3, values=[10, 20, 30])
        assert vp[0].value == 10
        assert vp[1].value == 20
        assert vp[2].value == 30

    def test_getitem_negative_index(self):
        """Negative indexing works."""
        vp = VectorParameter("p", 3, values=[10, 20, 30])
        assert vp[-1].value == 30
        assert vp[-2].value == 20

    def test_getitem_out_of_range_raises(self):
        """Out of range index raises IndexError."""
        vp = VectorParameter("p", 3)
        with pytest.raises(IndexError):
            _ = vp[5]

    def test_element_naming(self):
        """Elements are named correctly."""
        vp = VectorParameter("prices", 3)
        assert vp[0].name == "prices[0]"
        assert vp[1].name == "prices[1]"
        assert vp[2].name == "prices[2]"


class TestVectorParameterSet:
    """Tests for VectorParameter.set() method."""

    def test_set_updates_all(self):
        """set() updates all values."""
        vp = VectorParameter("p", 3, values=[1, 2, 3])
        vp.set([10, 20, 30])
        np.testing.assert_array_equal(vp.get_values(), [10, 20, 30])

    def test_set_wrong_size_raises(self):
        """set() with wrong size raises ShapeMismatchError."""
        vp = VectorParameter("p", 3)
        with pytest.raises(ShapeMismatchError, match="(?i)shape mismatch"):
            vp.set([1, 2])

    def test_individual_element_set(self):
        """Can set individual elements."""
        vp = VectorParameter("p", 3, values=[1, 2, 3])
        vp[1].set(100)
        np.testing.assert_array_equal(vp.get_values(), [1, 100, 3])


class TestVectorParameterIteration:
    """Tests for VectorParameter iteration."""

    def test_iterate_yields_parameters(self):
        """Iteration yields Parameter objects."""
        vp = VectorParameter("p", 3, values=[10, 20, 30])
        params = list(vp)
        assert len(params) == 3
        assert all(isinstance(p, Parameter) for p in params)

    def test_iterate_correct_values(self):
        """Iteration yields correct values."""
        vp = VectorParameter("p", 3, values=[10, 20, 30])
        values = [p.value for p in vp]
        assert values == [10.0, 20.0, 30.0]


class TestVectorParameterToNumpy:
    """Tests for VectorParameter.to_numpy()."""

    def test_to_numpy_returns_array(self):
        """to_numpy() returns numpy array."""
        vp = VectorParameter("p", 3, values=[1, 2, 3])
        arr = vp.to_numpy()
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [1, 2, 3])

    def test_get_values_alias(self):
        """get_values() is same as to_numpy()."""
        vp = VectorParameter("p", 3, values=[1, 2, 3])
        np.testing.assert_array_equal(vp.get_values(), vp.to_numpy())


class TestVectorParameterRepr:
    """Tests for VectorParameter repr."""

    def test_repr_basic(self):
        """VectorParameter has readable repr."""
        vp = VectorParameter("prices", 5)
        r = repr(vp)
        assert "VectorParameter" in r
        assert "prices" in r
        assert "5" in r


# =============================================================================
# Integration Tests
# =============================================================================


class TestParameterIntegration:
    """Integration tests for Parameter with other optyx components."""

    def test_parameter_with_vector_variable(self):
        """Parameter works with VectorVariable."""
        price = Parameter("price", value=10)
        x = VectorVariable("x", 3)

        # price * x[i] for each element
        exprs = [price * x[i] for i in range(3)]
        values = {"x[0]": 1, "x[1]": 2, "x[2]": 3}

        results = [e.evaluate(values) for e in exprs]
        assert results == [10.0, 20.0, 30.0]

        # Update price
        price.set(20)
        results = [e.evaluate(values) for e in exprs]
        assert results == [20.0, 40.0, 60.0]

    def test_vector_parameter_dot_product(self):
        """VectorParameter in dot product expression."""
        prices = VectorParameter("p", 3, values=[10, 20, 30])
        x = VectorVariable("x", 3)

        # Manual dot product (prices are Parameters, not VectorVariable)
        revenue = sum(prices[i] * x[i] for i in range(3))

        values = {"x[0]": 1, "x[1]": 2, "x[2]": 3}
        result = revenue.evaluate(values)
        # 10*1 + 20*2 + 30*3 = 10 + 40 + 90 = 140
        assert result == 140.0

        # Update prices
        prices.set([5, 10, 15])
        result = revenue.evaluate(values)
        # 5*1 + 10*2 + 15*3 = 5 + 20 + 45 = 70
        assert result == 70.0

    def test_sensitivity_analysis_pattern(self):
        """Simulate sensitivity analysis workflow."""
        # Problem: maximize profit = price * x - cost
        price = Parameter("price", value=100)
        x = Variable("x", lb=0, ub=10)
        cost = 50

        profit = price * x - cost

        # Evaluate at optimal x
        result1 = profit.evaluate({"x": 10})
        assert result1 == 950.0  # 100*10 - 50

        # Sensitivity: what if price increases?
        price.set(120)
        result2 = profit.evaluate({"x": 10})
        assert result2 == 1150.0  # 120*10 - 50

        # What if price drops?
        price.set(80)
        result3 = profit.evaluate({"x": 10})
        assert result3 == 750.0  # 80*10 - 50

    def test_rolling_horizon_pattern(self):
        """Simulate rolling horizon optimization pattern."""
        # Time-varying demand parameter
        demand = VectorParameter("demand", 3, values=[100, 150, 200])
        x = VectorVariable("x", 3, lb=0)

        # Objective: minimize sum of |x - demand| (simplified)
        # Here we just check expression building
        expressions = [(x[i] - demand[i]) for i in range(3)]

        values = {"x[0]": 100, "x[1]": 150, "x[2]": 200}
        deviations = [e.evaluate(values) for e in expressions]
        assert deviations == [0.0, 0.0, 0.0]

        # Next period: demand shifts
        demand.set([150, 200, 250])
        deviations = [e.evaluate(values) for e in expressions]
        assert deviations == [-50.0, -50.0, -50.0]

    def test_parameter_with_problem_solve(self):
        """Parameter works with Problem.solve() for fast re-solves."""
        from optyx import Problem

        # maximize price * x - x^2 → optimal x = price / 2
        x = Variable("x", lb=0)
        price = Parameter("price", value=100)

        prob = Problem().maximize(price * x - x**2)

        # First solve
        sol1 = prob.solve()
        assert abs(sol1.values["x"] - 50.0) < 0.1  # x = 100/2 = 50

        # Update parameter and re-solve
        price.set(120)
        sol2 = prob.solve()
        assert abs(sol2.values["x"] - 60.0) < 0.1  # x = 120/2 = 60

        # Different solutions confirm parameter was picked up
        assert sol1.values["x"] != sol2.values["x"]
