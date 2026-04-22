"""Tests for modeling syntax conveniences (Issue #97).

Tests cover:
1. Expression.between(lb, ub) and VectorVariable.between()
2. subject_to() accepting generators and iterables
3. Problem context manager (with Problem() as p:)
4. Variable(obj=) shorthand for linear objective coefficients
"""

import numpy as np
import pytest

from optyx import Constant, Problem, Variable, VectorVariable
from optyx.constraints import Constraint


# ============================================================
# 1. Expression.between(lb, ub) — Range Constraints
# ============================================================


class TestExpressionBetween:
    """Tests for Expression.between() and VectorVariable.between()."""

    def test_variable_between_returns_two_constraints(self):
        """between() returns [self >= lb, self <= ub]."""
        x = Variable("x")
        constraints = x.between(0, 10)
        assert len(constraints) == 2
        assert all(isinstance(c, Constraint) for c in constraints)

    def test_variable_between_senses(self):
        """between() creates >= and <= constraints."""
        x = Variable("x")
        constraints = x.between(0, 10)
        senses = {c.sense for c in constraints}
        assert senses == {">=", "<="}

    def test_variable_between_satisfied(self):
        """between() constraints are satisfied for interior points."""
        x = Variable("x")
        constraints = x.between(2.0, 8.0)
        point = {"x": 5.0}
        assert all(c.is_satisfied(point) for c in constraints)

    def test_variable_between_violated_below(self):
        """between() detects violations below lower bound."""
        x = Variable("x")
        constraints = x.between(2.0, 8.0)
        point = {"x": 1.0}
        assert not all(c.is_satisfied(point) for c in constraints)

    def test_variable_between_violated_above(self):
        """between() detects violations above upper bound."""
        x = Variable("x")
        constraints = x.between(2.0, 8.0)
        point = {"x": 9.0}
        assert not all(c.is_satisfied(point) for c in constraints)

    def test_expression_between(self):
        """between() works on compound expressions."""
        x = Variable("x")
        y = Variable("y")
        constraints = (x + y).between(-1, 1)
        assert len(constraints) == 2
        assert all(c.is_satisfied({"x": 0.3, "y": 0.2}) for c in constraints)
        assert not all(c.is_satisfied({"x": 1.0, "y": 1.0}) for c in constraints)

    def test_between_in_problem(self):
        """between() constraints work in a solve."""
        x = Variable("x", lb=-10, ub=10)
        prob = Problem()
        prob.minimize(x)
        prob.subject_to(x.between(3.0, 7.0))
        sol = prob.solve()
        assert sol.is_optimal
        assert abs(sol[x] - 3.0) < 1e-6

    def test_vector_between_scalar_bounds(self):
        """VectorVariable.between() with scalar bounds."""
        x = VectorVariable("x", 3)
        constraints = x.between(0, 10)
        assert len(constraints) == 6  # 3 >= + 3 <=

    def test_vector_between_array_bounds(self):
        """VectorVariable.between() with array bounds."""
        x = VectorVariable("x", 3)
        lb = np.array([0.0, 1.0, 2.0])
        ub = np.array([10.0, 11.0, 12.0])
        constraints = x.between(lb, ub)
        assert len(constraints) == 6

    def test_vector_between_in_problem(self):
        """VectorVariable.between() works in a solve."""
        x = VectorVariable("x", 3, lb=-10, ub=10)
        c = np.array([1.0, 1.0, 1.0])
        prob = Problem()
        prob.minimize(c @ x)
        prob.subject_to(x.between(2.0, 5.0))
        sol = prob.solve()
        assert sol.is_optimal
        vals = sol[x]
        assert np.allclose(vals, 2.0, atol=1e-6)


# ============================================================
# 2. subject_to() Accepts Generators and Iterables
# ============================================================


class TestSubjectToGenerators:
    """Tests for subject_to() accepting generators and iterables."""

    def test_subject_to_generator(self):
        """subject_to() accepts a generator expression."""
        x = VectorVariable("x", 5)
        prob = Problem()
        prob.minimize(x[0])
        prob.subject_to(x[i] >= 0 for i in range(5))
        assert prob.n_constraints == 5

    def test_subject_to_list(self):
        """subject_to() accepts a list of constraints."""
        x = Variable("x")
        y = Variable("y")
        prob = Problem()
        prob.minimize(x + y)
        prob.subject_to([x >= 0, y >= 0, x + y <= 10])
        assert prob.n_constraints == 3

    def test_subject_to_tuple(self):
        """subject_to() accepts a tuple of constraints."""
        x = Variable("x")
        prob = Problem()
        prob.minimize(x)
        prob.subject_to((x >= 1, x <= 5))
        assert prob.n_constraints == 2

    def test_subject_to_single_constraint(self):
        """subject_to() still accepts a single constraint."""
        x = Variable("x")
        prob = Problem()
        prob.minimize(x)
        prob.subject_to(x >= 0)
        assert prob.n_constraints == 1

    def test_subject_to_generator_solve(self):
        """subject_to() with generator produces correct solution."""
        x = VectorVariable("x", 3, lb=-10, ub=10)
        c = np.array([1.0, 1.0, 1.0])
        prob = Problem()
        prob.minimize(c @ x)
        prob.subject_to(x[i] >= float(i) for i in range(3))
        sol = prob.solve()
        assert sol.is_optimal
        vals = sol[x]
        assert np.allclose(vals, [0.0, 1.0, 2.0], atol=1e-6)

    def test_subject_to_between_iterable(self):
        """subject_to() accepts list from between() directly."""
        x = Variable("x", lb=-10, ub=10)
        prob = Problem()
        prob.minimize(x)
        prob.subject_to(x.between(3.0, 7.0))
        sol = prob.solve()
        assert sol.is_optimal
        assert abs(sol[x] - 3.0) < 1e-6

    def test_subject_to_chaining(self):
        """subject_to() returns self for method chaining."""
        x = Variable("x")
        prob = Problem()
        result = prob.minimize(x).subject_to(x >= 0).subject_to(x <= 10)
        assert result is prob
        assert prob.n_constraints == 2

    def test_subject_to_invalid_type_error(self):
        """subject_to() raises ConstraintError for invalid types."""
        from optyx.core.errors import ConstraintError

        prob = Problem()
        with pytest.raises(ConstraintError):
            prob.subject_to(42)  # type: ignore[arg-type]


# ============================================================
# 3. Problem Context Manager
# ============================================================


class TestProblemContextManager:
    """Tests for Problem context manager support."""

    def test_context_manager_returns_self(self):
        """__enter__ returns the Problem instance."""
        prob = Problem()
        with prob as p:
            assert p is prob

    def test_context_manager_basic_solve(self):
        """Problem works correctly inside a with block."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        with Problem() as prob:
            prob.minimize(x + y)
            prob.subject_to(x + y >= 1)
            sol = prob.solve()

        assert sol.is_optimal
        assert abs(sol.objective_value - 1.0) < 1e-6

    def test_context_manager_with_name(self):
        """Named Problem works in context manager."""
        with Problem(name="test") as prob:
            assert prob.name == "test"

    def test_context_manager_no_exception_on_exit(self):
        """__exit__ doesn't raise when no exception occurs."""
        with Problem() as _prob:
            pass  # No operations

    def test_context_manager_propagates_exception(self):
        """Context manager doesn't suppress exceptions."""
        with pytest.raises(ValueError):
            with Problem() as _prob:
                raise ValueError("test error")

    def test_context_manager_lp(self):
        """Context manager works with LP problems."""
        x = VectorVariable("x", 3, lb=0)
        c = np.array([1.0, 2.0, 3.0])

        with Problem() as prob:
            prob.minimize(c @ x)
            prob.subject_to(x[0] + x[1] + x[2] >= 1)
            sol = prob.solve()

        assert sol.is_optimal
        assert abs(sol[x[0]] - 1.0) < 1e-6


# ============================================================
# 4. Variable(obj=) Shorthand
# ============================================================


class TestVariableObjShorthand:
    """Tests for Variable(obj=) linear objective coefficient."""

    def test_obj_default_zero(self):
        """Variable obj defaults to 0.0."""
        x = Variable("x")
        assert x.obj == 0.0

    def test_obj_stored(self):
        """Variable stores obj coefficient."""
        x = Variable("x", obj=5.0)
        assert x.obj == 5.0

    def test_obj_int_converted_to_float(self):
        """Integer obj is converted to float."""
        x = Variable("x", obj=3)
        assert x.obj == 3.0
        assert isinstance(x.obj, float)

    def test_obj_negative(self):
        """Negative obj coefficient works."""
        x = Variable("x", obj=-2.5)
        assert x.obj == -2.5

    def test_obj_lp_minimize(self):
        """Variable.obj contributes to LP objective in minimize."""
        x = Variable("x", lb=0, ub=10, obj=1.0)
        y = Variable("y", lb=0, ub=10, obj=2.0)

        prob = Problem()
        prob.minimize(Constant(0))  # Zero explicit objective
        prob.subject_to(x + y >= 5)
        sol = prob.solve()

        # Effective objective: 1*x + 2*y, minimized
        # Optimal: x=5, y=0 → obj=5
        assert sol.is_optimal
        assert abs(sol[x] - 5.0) < 1e-4
        assert abs(sol[y] - 0.0) < 1e-4

    def test_obj_lp_additive(self):
        """Variable.obj adds to explicit objective coefficients."""
        x = Variable("x", lb=0, ub=10, obj=1.0)
        y = Variable("y", lb=0, ub=10, obj=0.0)

        prob = Problem()
        prob.minimize(x + 3 * y)  # Explicit: 1*x + 3*y
        prob.subject_to(x + y >= 5)
        sol = prob.solve()

        # Effective objective: (1+1)*x + (3+0)*y = 2*x + 3*y
        # Optimal: x=5, y=0 → obj=10
        assert sol.is_optimal
        assert abs(sol[x] - 5.0) < 1e-4
        assert abs(sol[y] - 0.0) < 1e-4

    def test_obj_lp_maximize(self):
        """Variable.obj works with maximize."""
        x = Variable("x", lb=0, ub=10, obj=1.0)
        y = Variable("y", lb=0, ub=10, obj=2.0)

        prob = Problem()
        prob.maximize(Constant(0))
        prob.subject_to(x + y <= 8)
        sol = prob.solve()

        # Effective objective: maximize 1*x + 2*y
        # Optimal: x=0, y=8 → obj=16
        assert sol.is_optimal
        assert abs(sol[x] - 0.0) < 1e-4
        assert abs(sol[y] - 8.0) < 1e-4

    def test_obj_nlp(self):
        """Variable.obj works with NLP solver."""
        x = Variable("x", lb=0, ub=10, obj=2.0)

        prob = Problem()
        prob.minimize(x**2)  # Explicit: x², with obj: +2x
        prob.subject_to(x >= 0)
        sol = prob.solve()

        # Effective: x² + 2x, minimum at x = -1 but lb=0 → x=0
        assert sol.is_optimal
        assert abs(sol[x] - 0.0) < 1e-4

    def test_obj_zero_no_effect(self):
        """Variable with obj=0 doesn't change the objective."""
        x = Variable("x", lb=0, ub=10, obj=0.0)

        prob = Problem()
        prob.minimize(2 * x)
        prob.subject_to(x >= 3)
        sol = prob.solve()

        assert sol.is_optimal
        assert abs(sol[x] - 3.0) < 1e-6

    def test_obj_with_bounds(self):
        """Variable.obj works alongside lb/ub."""
        x = Variable("x", lb=1, ub=5, obj=1.0)
        assert x.lb == 1
        assert x.ub == 5
        assert x.obj == 1.0
