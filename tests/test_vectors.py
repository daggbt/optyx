"""Tests for VectorVariable."""

import numpy as np
import pytest

from optyx.core.vectors import (
    VectorVariable,
    VectorExpression,
    VectorSum,
    DotProduct,
    L2Norm,
    L1Norm,
    LinearCombination,
    vector_sum,
    norm,
)
from optyx.core.expressions import Variable


class TestVectorVariableCreation:
    """Tests for VectorVariable creation."""

    def test_basic_creation(self):
        """VectorVariable creates the specified number of variables."""
        x = VectorVariable("x", 5)
        assert len(x) == 5
        assert x.size == 5
        assert x.name == "x"

    def test_large_vector(self):
        """VectorVariable can handle large sizes."""
        x = VectorVariable("x", 100)
        assert len(x) == 100

    def test_element_naming(self):
        """Elements are named with bracket notation."""
        x = VectorVariable("x", 5)
        assert x[0].name == "x[0]"
        assert x[1].name == "x[1]"
        assert x[4].name == "x[4]"

    def test_zero_size_raises(self):
        """Size must be positive."""
        with pytest.raises(ValueError, match="positive"):
            VectorVariable("x", 0)

    def test_negative_size_raises(self):
        """Negative size raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            VectorVariable("x", -5)


class TestVectorVariableBounds:
    """Tests for bounds propagation."""

    def test_lower_bound_propagates(self):
        """Lower bound applies to all elements."""
        x = VectorVariable("x", 5, lb=0)
        for v in x:
            assert v.lb == 0

    def test_upper_bound_propagates(self):
        """Upper bound applies to all elements."""
        x = VectorVariable("x", 5, ub=10)
        for v in x:
            assert v.ub == 10

    def test_both_bounds_propagate(self):
        """Both bounds apply to all elements."""
        x = VectorVariable("x", 5, lb=-1, ub=1)
        for v in x:
            assert v.lb == -1
            assert v.ub == 1

    def test_no_bounds_by_default(self):
        """No bounds by default."""
        x = VectorVariable("x", 5)
        assert x[0].lb is None
        assert x[0].ub is None


class TestVectorVariableDomain:
    """Tests for domain propagation."""

    def test_continuous_by_default(self):
        """Domain is continuous by default."""
        x = VectorVariable("x", 5)
        assert x.domain == "continuous"
        for v in x:
            assert v.domain == "continuous"

    def test_integer_domain(self):
        """Integer domain propagates to all elements."""
        x = VectorVariable("x", 5, domain="integer")
        for v in x:
            assert v.domain == "integer"

    def test_binary_domain(self):
        """Binary domain propagates and sets bounds."""
        x = VectorVariable("x", 5, domain="binary")
        for v in x:
            assert v.domain == "binary"
            assert v.lb == 0.0
            assert v.ub == 1.0


class TestVectorVariableIndexing:
    """Tests for indexing operations."""

    def test_positive_index(self):
        """Positive index returns correct element."""
        x = VectorVariable("x", 10)
        assert x[0].name == "x[0]"
        assert x[5].name == "x[5]"
        assert x[9].name == "x[9]"

    def test_negative_index(self):
        """Negative index works like Python lists."""
        x = VectorVariable("x", 10)
        assert x[-1].name == "x[9]"
        assert x[-2].name == "x[8]"
        assert x[-10].name == "x[0]"

    def test_index_returns_variable(self):
        """Indexing returns a Variable instance."""
        x = VectorVariable("x", 5)
        assert isinstance(x[0], Variable)

    def test_index_out_of_range(self):
        """Out of range index raises IndexError."""
        x = VectorVariable("x", 5)
        with pytest.raises(IndexError):
            x[5]
        with pytest.raises(IndexError):
            x[10]
        with pytest.raises(IndexError):
            x[-6]


class TestVectorVariableSlicing:
    """Tests for slicing operations."""

    def test_basic_slice(self):
        """Basic slicing returns new VectorVariable."""
        x = VectorVariable("x", 10)
        y = x[2:5]
        assert isinstance(y, VectorVariable)
        assert len(y) == 3

    def test_slice_preserves_variables(self):
        """Sliced VectorVariable references same Variable objects."""
        x = VectorVariable("x", 10)
        y = x[2:5]
        assert y[0].name == "x[2]"
        assert y[1].name == "x[3]"
        assert y[2].name == "x[4]"

    def test_slice_from_start(self):
        """Slice from start works."""
        x = VectorVariable("x", 10)
        y = x[:3]
        assert len(y) == 3
        assert y[0].name == "x[0]"

    def test_slice_to_end(self):
        """Slice to end works."""
        x = VectorVariable("x", 10)
        y = x[7:]
        assert len(y) == 3
        assert y[0].name == "x[7]"

    def test_slice_with_step(self):
        """Slice with step works."""
        x = VectorVariable("x", 10)
        y = x[::2]  # Every other element
        assert len(y) == 5
        assert y[0].name == "x[0]"
        assert y[1].name == "x[2]"

    def test_negative_slice(self):
        """Negative indices in slice work."""
        x = VectorVariable("x", 10)
        y = x[-3:]
        assert len(y) == 3
        assert y[0].name == "x[7]"

    def test_empty_slice_raises(self):
        """Empty slice raises IndexError."""
        x = VectorVariable("x", 10)
        with pytest.raises(IndexError, match="empty"):
            x[5:5]


class TestVectorVariableIteration:
    """Tests for iteration."""

    def test_iter(self):
        """Can iterate over VectorVariable."""
        x = VectorVariable("x", 5)
        names = [v.name for v in x]
        assert names == ["x[0]", "x[1]", "x[2]", "x[3]", "x[4]"]

    def test_list_conversion(self):
        """Can convert to list."""
        x = VectorVariable("x", 3)
        vars_list = list(x)
        assert len(vars_list) == 3
        assert all(isinstance(v, Variable) for v in vars_list)


class TestVectorVariableGetVariables:
    """Tests for get_variables method."""

    def test_get_variables(self):
        """get_variables returns list of all variables."""
        x = VectorVariable("x", 5)
        vars_list = x.get_variables()
        assert len(vars_list) == 5
        assert all(isinstance(v, Variable) for v in vars_list)
        assert [v.name for v in vars_list] == ["x[0]", "x[1]", "x[2]", "x[3]", "x[4]"]


class TestVectorVariableRepr:
    """Tests for string representation."""

    def test_basic_repr(self):
        """Basic repr shows name and size."""
        x = VectorVariable("x", 5)
        assert "x" in repr(x)
        assert "5" in repr(x)

    def test_repr_with_bounds(self):
        """Repr shows bounds when set."""
        x = VectorVariable("x", 5, lb=0, ub=10)
        r = repr(x)
        assert "lb=0" in r
        assert "ub=10" in r

    def test_repr_with_domain(self):
        """Repr shows non-continuous domain."""
        x = VectorVariable("x", 5, domain="binary")
        assert "binary" in repr(x)


class TestVectorSum:
    """Tests for VectorSum expression."""

    def test_sum_creation(self):
        """vector_sum returns VectorSum for VectorVariable."""
        x = VectorVariable("x", 5)
        s = vector_sum(x)
        assert isinstance(s, VectorSum)

    def test_sum_evaluate(self):
        """VectorSum evaluates correctly."""
        x = VectorVariable("x", 3)
        s = vector_sum(x)
        result = s.evaluate({"x[0]": 1, "x[1]": 2, "x[2]": 3})
        assert result == 6.0

    def test_sum_evaluate_floats(self):
        """VectorSum works with float values."""
        x = VectorVariable("x", 3)
        s = vector_sum(x)
        result = s.evaluate({"x[0]": 1.5, "x[1]": 2.5, "x[2]": 3.0})
        assert result == 7.0

    def test_sum_get_variables(self):
        """VectorSum returns all vector variables."""
        x = VectorVariable("x", 3)
        s = vector_sum(x)
        variables = s.get_variables()
        assert len(variables) == 3
        assert all(v in variables for v in x)

    def test_sum_repr(self):
        """VectorSum has useful repr."""
        x = VectorVariable("x", 5)
        s = vector_sum(x)
        assert "x" in repr(s)


class TestVectorExpression:
    """Tests for VectorExpression."""

    def test_vector_expression_creation(self):
        """VectorExpression can be created from expressions."""
        x = VectorVariable("x", 3)
        ve = VectorExpression(list(x))
        assert len(ve) == 3

    def test_vector_expression_indexing(self):
        """VectorExpression supports indexing."""
        x = VectorVariable("x", 3)
        ve = VectorExpression(list(x))
        assert ve[0] == x[0]
        assert ve[-1] == x[2]

    def test_vector_expression_evaluate(self):
        """VectorExpression can evaluate all elements."""
        x = VectorVariable("x", 3)
        ve = VectorExpression(list(x))
        result = ve.evaluate({"x[0]": 1, "x[1]": 2, "x[2]": 3})
        assert result == [1, 2, 3]

    def test_empty_vector_expression_raises(self):
        """Empty VectorExpression raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            VectorExpression([])


class TestVectorArithmetic:
    """Tests for element-wise arithmetic operations."""

    def test_add_two_vectors(self):
        """x + y creates element-wise sum."""
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        z = x + y
        assert isinstance(z, VectorExpression)
        assert len(z) == 3
        # Evaluate
        values = {"x[0]": 1, "x[1]": 2, "x[2]": 3, "y[0]": 10, "y[1]": 20, "y[2]": 30}
        result = z.evaluate(values)
        assert result == [11, 22, 33]

    def test_add_scalar_right(self):
        """x + 5 broadcasts scalar to all elements."""
        x = VectorVariable("x", 3)
        z = x + 5
        assert isinstance(z, VectorExpression)
        result = z.evaluate({"x[0]": 1, "x[1]": 2, "x[2]": 3})
        assert result == [6, 7, 8]

    def test_add_scalar_left(self):
        """5 + x broadcasts scalar to all elements."""
        x = VectorVariable("x", 3)
        z = 5 + x
        assert isinstance(z, VectorExpression)
        result = z.evaluate({"x[0]": 1, "x[1]": 2, "x[2]": 3})
        assert result == [6, 7, 8]

    def test_sub_two_vectors(self):
        """x - y creates element-wise difference."""
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        z = x - y
        values = {"x[0]": 10, "x[1]": 20, "x[2]": 30, "y[0]": 1, "y[1]": 2, "y[2]": 3}
        result = z.evaluate(values)
        assert result == [9, 18, 27]

    def test_sub_scalar(self):
        """x - 5 subtracts scalar from all elements."""
        x = VectorVariable("x", 3)
        z = x - 5
        result = z.evaluate({"x[0]": 10, "x[1]": 20, "x[2]": 30})
        assert result == [5, 15, 25]

    def test_rsub_scalar(self):
        """5 - x subtracts vector from scalar."""
        x = VectorVariable("x", 3)
        z = 5 - x
        result = z.evaluate({"x[0]": 1, "x[1]": 2, "x[2]": 3})
        assert result == [4, 3, 2]

    def test_mul_scalar_right(self):
        """x * 2 multiplies all elements by scalar."""
        x = VectorVariable("x", 3)
        z = x * 2
        result = z.evaluate({"x[0]": 1, "x[1]": 2, "x[2]": 3})
        assert result == [2, 4, 6]

    def test_mul_scalar_left(self):
        """2 * x multiplies all elements by scalar."""
        x = VectorVariable("x", 3)
        z = 2 * x
        result = z.evaluate({"x[0]": 1, "x[1]": 2, "x[2]": 3})
        assert result == [2, 4, 6]

    def test_div_scalar(self):
        """x / 2 divides all elements by scalar."""
        x = VectorVariable("x", 3)
        z = x / 2
        result = z.evaluate({"x[0]": 2, "x[1]": 4, "x[2]": 6})
        assert result == [1, 2, 3]

    def test_neg(self):
        """-x negates all elements."""
        x = VectorVariable("x", 3)
        z = -x
        result = z.evaluate({"x[0]": 1, "x[1]": -2, "x[2]": 3})
        assert result == [-1, 2, -3]

    def test_size_mismatch_raises(self):
        """Adding vectors of different sizes raises ValueError."""
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 5)
        with pytest.raises(ValueError, match="size mismatch"):
            x + y

    def test_chained_operations(self):
        """Chained operations work correctly."""
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        z = 2 * x + y - 1
        values = {"x[0]": 1, "x[1]": 2, "x[2]": 3, "y[0]": 10, "y[1]": 20, "y[2]": 30}
        result = z.evaluate(values)
        assert result == [11, 23, 35]  # 2*1+10-1, 2*2+20-1, 2*3+30-1

    def test_vector_expression_arithmetic(self):
        """VectorExpression supports further arithmetic."""
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        z = (x + y) * 2
        values = {"x[0]": 1, "x[1]": 2, "x[2]": 3, "y[0]": 1, "y[1]": 2, "y[2]": 3}
        result = z.evaluate(values)
        assert result == [4, 8, 12]


class TestVectorConstraints:
    """Tests for vectorized constraints."""

    def test_le_scalar(self):
        """x <= scalar returns list of constraints."""
        x = VectorVariable("x", 3)
        constraints = x <= 10
        assert len(constraints) == 3
        # Each constraint should reference the corresponding variable
        for i, c in enumerate(constraints):
            assert c.sense == "<="
            vars_in_constraint = c.get_variables()
            assert len(vars_in_constraint) == 1
            var = list(vars_in_constraint)[0]
            assert var.name == f"x[{i}]"

    def test_ge_scalar(self):
        """x >= scalar returns list of constraints."""
        x = VectorVariable("x", 5)
        constraints = x >= 0
        assert len(constraints) == 5
        for c in constraints:
            assert c.sense == ">="

    def test_le_vector(self):
        """x <= y creates element-wise constraints."""
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        constraints = x <= y
        assert len(constraints) == 3
        for i, c in enumerate(constraints):
            assert c.sense == "<="
            vars_in_constraint = c.get_variables()
            assert len(vars_in_constraint) == 2
            names = {v.name for v in vars_in_constraint}
            assert names == {f"x[{i}]", f"y[{i}]"}

    def test_ge_vector(self):
        """x >= y creates element-wise constraints."""
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        constraints = x >= y
        assert len(constraints) == 3
        for c in constraints:
            assert c.sense == ">="

    def test_eq_scalar(self):
        """x.eq(scalar) returns list of equality constraints."""
        x = VectorVariable("x", 3)
        constraints = x.eq(5)
        assert len(constraints) == 3
        for c in constraints:
            assert c.sense == "=="

    def test_eq_vector(self):
        """x.eq(y) creates element-wise equality constraints."""
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        constraints = x.eq(y)
        assert len(constraints) == 3
        for i, c in enumerate(constraints):
            assert c.sense == "=="
            vars_in_constraint = c.get_variables()
            assert len(vars_in_constraint) == 2

    def test_constraint_satisfaction(self):
        """Vectorized constraints evaluate correctly."""
        x = VectorVariable("x", 3)
        constraints = x >= 0
        # All satisfied
        point_satisfied = {"x[0]": 1, "x[1]": 0, "x[2]": 5}
        for c in constraints:
            assert c.is_satisfied(point_satisfied)
        # One violated
        point_violated = {"x[0]": 1, "x[1]": -1, "x[2]": 5}
        assert constraints[0].is_satisfied(point_violated)
        assert not constraints[1].is_satisfied(point_violated)
        assert constraints[2].is_satisfied(point_violated)

    def test_large_vector_constraints(self):
        """100-element vector produces 100 constraints."""
        x = VectorVariable("x", 100)
        constraints = x >= 0
        assert len(constraints) == 100

    def test_vector_constraint_size_mismatch(self):
        """Constraint between differently sized vectors raises ValueError."""
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 5)
        with pytest.raises(ValueError, match="size mismatch"):
            x <= y

    def test_vector_expression_constraints(self):
        """VectorExpression supports constraints too."""
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        z = x + y
        constraints = z <= 10
        assert len(constraints) == 3
        for c in constraints:
            assert c.sense == "<="
            assert len(c.get_variables()) == 2

    def test_vector_expression_ge(self):
        """VectorExpression >= scalar works."""
        x = VectorVariable("x", 3)
        z = 2 * x
        constraints = z >= 0
        assert len(constraints) == 3
        for c in constraints:
            assert c.sense == ">="

    def test_vector_expression_eq(self):
        """VectorExpression.eq() works."""
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        z = x - y
        constraints = z.eq(0)
        assert len(constraints) == 3
        for c in constraints:
            assert c.sense == "=="


class TestProblemWithVectorConstraints:
    """Tests for Problem.subject_to with vector constraints."""

    def test_subject_to_accepts_list(self):
        """Problem.subject_to accepts list of constraints."""
        from optyx import Problem

        x = VectorVariable("x", 5)
        prob = Problem()
        prob.subject_to(x >= 0)
        assert len(prob.constraints) == 5

    def test_subject_to_multiple_vector_constraints(self):
        """Can add multiple vectorized constraints."""
        from optyx import Problem

        x = VectorVariable("x", 3)
        prob = Problem()
        prob.subject_to(x >= 0)
        prob.subject_to(x <= 10)
        assert len(prob.constraints) == 6

    def test_mixed_scalar_and_vector_constraints(self):
        """Can mix scalar and vector constraints."""
        from optyx import Problem
        from optyx.core.expressions import Variable

        x = VectorVariable("x", 3)
        y = Variable("y")
        prob = Problem()
        prob.subject_to(x >= 0)  # 3 constraints
        prob.subject_to(y <= 100)  # 1 constraint
        assert len(prob.constraints) == 4

    def test_chained_subject_to(self):
        """subject_to returns self for chaining."""
        from optyx import Problem

        x = VectorVariable("x", 3)
        prob = Problem()
        result = prob.subject_to(x >= 0).subject_to(x <= 10)
        assert result is prob
        assert len(prob.constraints) == 6

    def test_problem_variables_from_vector(self):
        """Problem extracts variables from VectorVariable constraints."""
        from optyx import Problem

        x = VectorVariable("x", 3)
        prob = Problem()
        prob.minimize(x[0] + x[1] + x[2])
        prob.subject_to(x >= 0)
        # Should have all 3 variables
        assert len(prob.variables) == 3
        names = {v.name for v in prob.variables}
        assert names == {"x[0]", "x[1]", "x[2]"}


class TestDotProduct:
    """Tests for dot product operations."""

    def test_dot_product_basic(self):
        """x.dot(y) computes dot product."""
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        d = x.dot(y)
        assert isinstance(d, DotProduct)

    def test_dot_product_evaluates_correctly(self):
        """Dot product [1,2,3] · [4,5,6] = 32."""
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        d = x.dot(y)
        values = {
            "x[0]": 1,
            "x[1]": 2,
            "x[2]": 3,
            "y[0]": 4,
            "y[1]": 5,
            "y[2]": 6,
        }
        result = d.evaluate(values)
        assert result == 32  # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

    def test_self_dot_product(self):
        """x.dot(x) works (sum of squares)."""
        x = VectorVariable("x", 3)
        d = x.dot(x)
        values = {"x[0]": 1, "x[1]": 2, "x[2]": 3}
        result = d.evaluate(values)
        assert result == 14  # 1 + 4 + 9 = 14

    def test_dot_product_variables(self):
        """Dot product tracks all variables."""
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        d = x.dot(y)
        vars_set = d.get_variables()
        assert len(vars_set) == 6
        names = {v.name for v in vars_set}
        assert names == {"x[0]", "x[1]", "x[2]", "y[0]", "y[1]", "y[2]"}

    def test_dot_product_size_mismatch(self):
        """Dot product with different sizes raises ValueError."""
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 5)
        with pytest.raises(ValueError, match="size mismatch"):
            x.dot(y)

    def test_dot_product_repr(self):
        """DotProduct has readable repr."""
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        d = x.dot(y)
        assert "DotProduct" in repr(d)
        assert "x" in repr(d)
        assert "y" in repr(d)

    def test_vector_expression_dot(self):
        """VectorExpression.dot() works."""
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        z = x + 1  # VectorExpression
        d = z.dot(y)
        values = {
            "x[0]": 1,
            "x[1]": 2,
            "x[2]": 3,
            "y[0]": 1,
            "y[1]": 1,
            "y[2]": 1,
        }
        result = d.evaluate(values)
        assert result == 9  # (1+1)*1 + (2+1)*1 + (3+1)*1 = 2 + 3 + 4 = 9


class TestL2Norm:
    """Tests for L2 (Euclidean) norm."""

    def test_norm_basic(self):
        """norm(x) returns L2Norm expression."""
        x = VectorVariable("x", 2)
        n = norm(x)
        assert isinstance(n, L2Norm)

    def test_norm_evaluates_correctly(self):
        """||[3, 4]|| = 5."""
        x = VectorVariable("x", 2)
        n = norm(x)
        values = {"x[0]": 3, "x[1]": 4}
        result = n.evaluate(values)
        assert result == 5.0

    def test_norm_3d(self):
        """3D norm evaluates correctly."""
        x = VectorVariable("x", 3)
        n = norm(x)
        values = {"x[0]": 1, "x[1]": 2, "x[2]": 2}
        result = n.evaluate(values)
        assert result == 3.0  # sqrt(1 + 4 + 4) = 3

    def test_norm_variables(self):
        """Norm tracks all variables."""
        x = VectorVariable("x", 3)
        n = norm(x)
        vars_set = n.get_variables()
        assert len(vars_set) == 3

    def test_norm_repr(self):
        """L2Norm has readable repr."""
        x = VectorVariable("x", 3)
        n = norm(x)
        assert "L2Norm" in repr(n)


class TestL1Norm:
    """Tests for L1 (Manhattan) norm."""

    def test_l1_norm_basic(self):
        """norm(x, ord=1) returns L1Norm expression."""
        x = VectorVariable("x", 3)
        n = norm(x, ord=1)
        assert isinstance(n, L1Norm)

    def test_l1_norm_evaluates_correctly(self):
        """L1 norm with negative values: |1| + |-2| + |3| = 6."""
        x = VectorVariable("x", 3)
        n = norm(x, ord=1)
        values = {"x[0]": 1, "x[1]": -2, "x[2]": 3}
        result = n.evaluate(values)
        assert result == 6.0

    def test_l1_norm_positive(self):
        """L1 norm with all positive values."""
        x = VectorVariable("x", 3)
        n = norm(x, ord=1)
        values = {"x[0]": 1, "x[1]": 2, "x[2]": 3}
        result = n.evaluate(values)
        assert result == 6.0

    def test_l1_norm_repr(self):
        """L1Norm has readable repr."""
        x = VectorVariable("x", 3)
        n = norm(x, ord=1)
        assert "L1Norm" in repr(n)


class TestNormFunction:
    """Tests for the norm() function."""

    def test_norm_unsupported_order(self):
        """norm with unsupported order raises ValueError."""
        x = VectorVariable("x", 3)
        with pytest.raises(ValueError, match="Unsupported norm order"):
            norm(x, ord=3)

    def test_norm_default_is_l2(self):
        """norm() defaults to L2."""
        x = VectorVariable("x", 3)
        n = norm(x)
        assert isinstance(n, L2Norm)

    def test_norm_on_vector_expression(self):
        """norm works on VectorExpression."""
        x = VectorVariable("x", 2)
        z = 2 * x  # VectorExpression
        n = norm(z)
        values = {"x[0]": 1.5, "x[1]": 2}
        # ||[3, 4]|| = 5
        result = n.evaluate(values)
        assert result == 5.0


class TestLinearCombination:
    """Tests for LinearCombination expression."""

    def test_linear_combination_basic(self):
        """LinearCombination creates weighted sum."""
        coeffs = np.array([1.0, 2.0, 3.0])
        x = VectorVariable("x", 3)
        lc = LinearCombination(coeffs, x)
        assert isinstance(lc, LinearCombination)

    def test_linear_combination_evaluate(self):
        """LinearCombination evaluates correctly."""
        coeffs = np.array([0.12, 0.08, 0.10])
        w = VectorVariable("w", 3)
        lc = LinearCombination(coeffs, w)
        values = {"w[0]": 0.5, "w[1]": 0.3, "w[2]": 0.2}
        result = lc.evaluate(values)
        expected = 0.12 * 0.5 + 0.08 * 0.3 + 0.10 * 0.2
        assert abs(result - expected) < 1e-10

    def test_linear_combination_variables(self):
        """LinearCombination tracks all variables."""
        coeffs = np.array([1.0, 2.0, 3.0])
        x = VectorVariable("x", 3)
        lc = LinearCombination(coeffs, x)
        vars_set = lc.get_variables()
        assert len(vars_set) == 3

    def test_linear_combination_size_mismatch(self):
        """LinearCombination raises on size mismatch."""
        coeffs = np.array([1.0, 2.0])
        x = VectorVariable("x", 3)
        with pytest.raises(ValueError, match="Coefficient length"):
            LinearCombination(coeffs, x)

    def test_linear_combination_repr(self):
        """LinearCombination has readable repr."""
        coeffs = np.array([1.0, 2.0, 3.0])
        x = VectorVariable("x", 3)
        lc = LinearCombination(coeffs, x)
        assert "LinearCombination" in repr(lc)
        assert "3 coeffs" in repr(lc)


class TestNumpyIntegration:
    """Tests for NumPy integration with VectorVariable."""

    def test_matmul_operator(self):
        """numpy_array @ vector creates LinearCombination."""
        returns = np.array([0.12, 0.08, 0.10])
        weights = VectorVariable("w", 3)
        portfolio_return = returns @ weights
        assert isinstance(portfolio_return, LinearCombination)

    def test_matmul_evaluate(self):
        """matmul expression evaluates correctly."""
        coeffs = np.array([1.0, 2.0, 3.0])
        x = VectorVariable("x", 3)
        expr = coeffs @ x
        values = {"x[0]": 1.0, "x[1]": 2.0, "x[2]": 3.0}
        result = expr.evaluate(values)
        assert result == 14.0  # 1*1 + 2*2 + 3*3

    def test_to_numpy(self):
        """to_numpy extracts solution as numpy array."""
        x = VectorVariable("x", 3)
        solution = {"x[0]": 1.0, "x[1]": 2.0, "x[2]": 3.0}
        arr = x.to_numpy(solution)
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0])

    def test_to_numpy_order(self):
        """to_numpy preserves element order."""
        x = VectorVariable("x", 5)
        solution = {f"x[{i}]": float(i * 10) for i in range(5)}
        arr = x.to_numpy(solution)
        expected = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        np.testing.assert_array_equal(arr, expected)

    def test_from_numpy(self):
        """from_numpy creates vector matching array size."""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        x = VectorVariable.from_numpy("x", data, lb=0)
        assert len(x) == 4
        assert x.lb == 0

    def test_from_numpy_with_bounds(self):
        """from_numpy propagates bounds."""
        data = np.array([1.0, 2.0, 3.0])
        x = VectorVariable.from_numpy("x", data, lb=-1, ub=1)
        assert x.lb == -1
        assert x.ub == 1
        assert len(x) == 3

    def test_from_numpy_rejects_2d(self):
        """from_numpy rejects 2D arrays."""
        data = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="1D array"):
            VectorVariable.from_numpy("x", data)

    def test_from_numpy_with_list(self):
        """from_numpy accepts list (converts to array)."""
        data = [1.0, 2.0, 3.0]
        x = VectorVariable.from_numpy("x", np.array(data))
        assert len(x) == 3

    def test_portfolio_example(self):
        """Full portfolio optimization example."""
        # Asset returns
        returns = np.array([0.12, 0.08, 0.10, 0.15])

        # Create weights vector
        weights = VectorVariable.from_numpy("w", returns, lb=0, ub=1)

        # Portfolio return: returns @ weights
        portfolio_return = returns @ weights

        # Evaluate at equal weights
        equal_weights = {f"w[{i}]": 0.25 for i in range(4)}
        result = portfolio_return.evaluate(equal_weights)
        expected = np.mean(returns)  # 0.1125
        assert abs(result - expected) < 1e-10
