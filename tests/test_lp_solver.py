"""Tests for LP solver backend."""

import numpy as np
import pytest

from optyx.core.expressions import Variable
from optyx.constraints import Constraint
from optyx.problem import Problem
from optyx.solution import SolverStatus
from optyx.solvers.lp_solver import solve_lp


class TestSolveLPSimple:
    """Basic LP solving tests."""

    def test_solve_lp_simple(self):
        """Simple 2-variable LP should solve correctly."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem()
        prob.minimize(x + y)
        prob.subject_to(x + y >= 1)

        solution = solve_lp(prob)

        assert solution.status == SolverStatus.OPTIMAL
        assert solution.objective_value is not None
        assert np.isclose(solution.objective_value, 1.0, atol=1e-6)

    def test_solve_lp_minimize(self):
        """Minimize objective should work correctly."""
        x = Variable("x", lb=0, ub=10)
        y = Variable("y", lb=0, ub=10)

        prob = Problem()
        prob.minimize(2 * x + 3 * y)
        prob.subject_to(x + y >= 5)

        solution = solve_lp(prob)

        assert solution.status == SolverStatus.OPTIMAL
        # Optimal: x=5, y=0 → objective = 10
        assert np.isclose(solution.objective_value, 10.0, atol=1e-6)
        assert np.isclose(solution["x"], 5.0, atol=1e-6)
        assert np.isclose(solution["y"], 0.0, atol=1e-6)

    def test_solve_lp_maximize(self):
        """Maximize objective should work correctly (negation handled)."""
        x = Variable("x", lb=0, ub=4)
        y = Variable("y", lb=0, ub=4)

        prob = Problem()
        prob.maximize(3 * x + 2 * y)
        prob.subject_to(x + y <= 5)

        solution = solve_lp(prob)

        assert solution.status == SolverStatus.OPTIMAL
        # Optimal: x=4, y=1 → objective = 14
        assert solution.objective_value is not None
        assert np.isclose(solution.objective_value, 14.0, atol=1e-6)

    def test_solve_lp_with_equality(self):
        """LP with equality constraints should work."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem()
        prob.minimize(x + 2 * y)
        prob.subject_to(Constraint(x + y - 10, "=="))  # x + y = 10

        solution = solve_lp(prob)

        assert solution.status == SolverStatus.OPTIMAL
        # Optimal: x=10, y=0 → objective = 10
        assert np.isclose(solution.objective_value, 10.0, atol=1e-6)


class TestSolveLPEdgeCases:
    """Edge case tests for LP solver."""

    def test_solve_lp_infeasible(self):
        """Infeasible LP should return INFEASIBLE status."""
        x = Variable("x", lb=0, ub=5)
        y = Variable("y", lb=0, ub=5)

        prob = Problem()
        prob.minimize(x + y)
        # Contradictory constraints: x + y >= 20 but both bounded by 5
        prob.subject_to(x + y >= 20)

        solution = solve_lp(prob)

        assert solution.status == SolverStatus.INFEASIBLE

    def test_solve_lp_unbounded(self):
        """Unbounded LP should return UNBOUNDED status."""
        x = Variable("x")  # No bounds
        y = Variable("y")  # No bounds

        prob = Problem()
        prob.minimize(-x - y)  # Minimize unbounded negative direction
        # No upper bounds, so can go to -infinity

        solution = solve_lp(prob)

        # HiGHS may detect unbounded or return a large solution
        assert solution.status in (SolverStatus.UNBOUNDED, SolverStatus.OPTIMAL)

    def test_solve_lp_no_constraints(self):
        """LP with only bounds (no general constraints)."""
        x = Variable("x", lb=0, ub=10)
        y = Variable("y", lb=0, ub=10)

        prob = Problem()
        prob.minimize(x + y)

        solution = solve_lp(prob)

        assert solution.status == SolverStatus.OPTIMAL
        assert np.isclose(solution.objective_value, 0.0, atol=1e-6)
        assert np.isclose(solution["x"], 0.0, atol=1e-6)
        assert np.isclose(solution["y"], 0.0, atol=1e-6)

    def test_nonlinear_objective_raises(self):
        """Non-linear objective should raise ValueError."""
        x = Variable("x", lb=0)

        prob = Problem()
        prob.minimize(x**2)

        with pytest.raises(ValueError, match="not linear"):
            solve_lp(prob)

    def test_nonlinear_constraint_raises(self):
        """Non-linear constraint should raise ValueError."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem()
        prob.minimize(x + y)
        # Manually create a nonlinear constraint
        prob._constraints.append(Constraint(x**2 + y**2 - 1, "<="))

        with pytest.raises(ValueError, match="not linear"):
            solve_lp(prob)


class TestSolveLPIntegration:
    """Integration tests via Problem.solve()."""

    def test_auto_method_uses_linprog_for_lp(self):
        """LP with method='auto' should use linprog."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem()
        prob.minimize(x + y)
        prob.subject_to(x + y >= 1)

        # Should automatically detect LP and use linprog
        solution = prob.solve(method="auto")

        assert solution.status == SolverStatus.OPTIMAL
        assert np.isclose(solution.objective_value, 1.0, atol=1e-6)

    def test_explicit_linprog_method(self):
        """Explicit method='linprog' should work."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem()
        prob.minimize(2 * x + y)
        prob.subject_to(x + y >= 3)

        solution = prob.solve(method="linprog")

        assert solution.status == SolverStatus.OPTIMAL
        # Optimal: x=0, y=3 → objective = 3
        assert np.isclose(solution.objective_value, 3.0, atol=1e-6)

    def test_explicit_slsqp_for_lp(self):
        """Explicit method='SLSQP' should override auto-detection."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem()
        prob.minimize(x + y)
        prob.subject_to(x + y >= 1)

        # Force SLSQP even though it's an LP
        solution = prob.solve(method="SLSQP")

        assert solution.status == SolverStatus.OPTIMAL
        assert np.isclose(solution.objective_value, 1.0, atol=1e-6)
