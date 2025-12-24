"""Validation benchmarks for constrained optimization problems.

Tests Optyx against Hock-Schittkowski problems and simple LP/QP.
Uses numpy vectorization (@, np.sum, np.array) where appropriate.
"""

from __future__ import annotations

import numpy as np

from optyx import Variable, Problem


class TestSimpleLP:
    """Simple linear programming tests using vectorized operations."""

    def test_simple_lp_maximize(self):
        """Simple LP maximization with vectorized objective.

        max 3x + 2y
        s.t. x + y <= 4
             x <= 2
             y <= 3
             x, y >= 0

        Optimal: (2, 2) with f* = 10
        """
        c = np.array([3.0, 2.0])
        x = np.array([Variable("x", lb=0), Variable("y", lb=0)])

        prob = Problem(name="simple_lp")
        prob.maximize(c @ x)  # Vectorized objective
        prob.subject_to(np.sum(x) <= 4)
        prob.subject_to(x[0] <= 2)
        prob.subject_to(x[1] <= 3)

        sol = prob.solve()

        assert sol.is_optimal
        assert abs(sol["x"] - 2.0) < 1e-4
        assert abs(sol["y"] - 2.0) < 1e-4
        assert abs(sol.objective_value - 10.0) < 1e-4

    def test_simple_lp_minimize(self):
        """Simple LP minimization with vectorized objective.

        min x + 2y
        s.t. x + y >= 3
             x >= 1
             y >= 1
             x, y <= 10

        Optimal: (2, 1) with f* = 4
        """
        c = np.array([1.0, 2.0])
        x = np.array([Variable("x", lb=1, ub=10), Variable("y", lb=1, ub=10)])

        prob = Problem(name="simple_lp_min")
        prob.minimize(c @ x)
        prob.subject_to(np.sum(x) >= 3)

        sol = prob.solve()

        assert sol.is_optimal
        assert abs(sol["x"] - 2.0) < 1e-4
        assert abs(sol["y"] - 1.0) < 1e-4
        assert abs(sol.objective_value - 4.0) < 1e-4


class TestSimpleQP:
    """Simple quadratic programming tests with vectorized operations."""

    def test_constrained_quadratic(self):
        """Constrained quadratic with vectorized objective.

        min ||x||²
        s.t. sum(x) >= 1

        Optimal: (0.5, 0.5) with f* = 0.5
        """
        x = np.array([Variable("x"), Variable("y")])

        prob = Problem(name="constrained_qp")
        prob.minimize(np.sum(x**2))  # Vectorized: ||x||²
        prob.subject_to(np.sum(x) >= 1)

        sol = prob.solve(x0=np.array([0.0, 0.0]))

        assert sol.is_optimal
        assert abs(sol["x"] - 0.5) < 1e-4
        assert abs(sol["y"] - 0.5) < 1e-4
        assert abs(sol.objective_value - 0.5) < 1e-4

    def test_bounded_quadratic(self):
        """Bounded quadratic.

        min (x-3)² + (y-2)²
        s.t. x + y <= 2
             x, y >= 0

        Using Lagrangian: at optimum, gradient (2(x-3), 2(y-2)) is
        proportional to constraint normal (1,1), so x-3 = y-2 => x = y+1.
        With active constraint x + y = 2: x = 1.5, y = 0.5
        Optimal: (1.5, 0.5) with f* = 4.5
        """
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem(name="bounded_qp")
        prob.minimize((x - 3) ** 2 + (y - 2) ** 2)
        prob.subject_to(x + y <= 2)

        sol = prob.solve()

        assert sol.is_optimal
        assert abs(sol["x"] - 1.5) < 1e-3
        assert abs(sol["y"] - 0.5) < 1e-3
        assert abs(sol.objective_value - 4.5) < 1e-3


class TestHS071:
    """Hock-Schittkowski Problem 71.

    A classic NLP test problem.

    min x1*x4*(x1+x2+x3) + x3
    s.t. x1*x2*x3*x4 >= 25
         x1² + x2² + x3² + x4² = 40
         1 <= x1, x2, x3, x4 <= 5

    Starting point: (1, 5, 5, 1)
    Optimal: approximately (1.0, 4.743, 3.821, 1.379) with f* ≈ 17.014
    """

    def test_hs071(self):
        """HS071 problem."""
        x1 = Variable("x1", lb=1, ub=5)
        x2 = Variable("x2", lb=1, ub=5)
        x3 = Variable("x3", lb=1, ub=5)
        x4 = Variable("x4", lb=1, ub=5)

        prob = Problem(name="hs071")
        prob.minimize(x1 * x4 * (x1 + x2 + x3) + x3)
        prob.subject_to(x1 * x2 * x3 * x4 >= 25)
        prob.subject_to((x1**2 + x2**2 + x3**2 + x4**2).constraint_eq(40))

        sol = prob.solve(x0=np.array([1.0, 5.0, 5.0, 1.0]))

        assert sol.is_optimal
        # Known optimal objective
        assert abs(sol.objective_value - 17.014) < 0.1


class TestHS076:
    """Hock-Schittkowski Problem 76.

    min f(x) = x1² + 0.5*x2² + x3² + 0.5*x4² - x1*x3 + x3*x4 - x1 - 3*x2 + x3 - x4
    s.t. x1 + 2*x2 + x3 + x4 <= 5
         3*x1 + x2 + 2*x3 - x4 <= 4
         -x2 - 4*x3 + 1.5 <= 0
         x1, x2, x3, x4 >= 0

    Optimal: approximately (0.2727, 2.0909, 0.2121, 0.5) with f* ≈ -4.6818
    """

    def test_hs076(self):
        """HS076 problem."""
        x1 = Variable("x1", lb=0)
        x2 = Variable("x2", lb=0)
        x3 = Variable("x3", lb=0)
        x4 = Variable("x4", lb=0)

        prob = Problem(name="hs076")
        prob.minimize(
            x1**2
            + 0.5 * x2**2
            + x3**2
            + 0.5 * x4**2
            - x1 * x3
            + x3 * x4
            - x1
            - 3 * x2
            + x3
            - x4
        )
        prob.subject_to(x1 + 2 * x2 + x3 + x4 <= 5)
        prob.subject_to(3 * x1 + x2 + 2 * x3 - x4 <= 4)
        prob.subject_to(-x2 - 4 * x3 + 1.5 <= 0)

        sol = prob.solve(x0=np.array([0.5, 0.5, 0.5, 0.5]))

        assert sol.is_optimal
        # Known optimal objective
        assert abs(sol.objective_value - (-4.6818)) < 0.1


class TestMixedConstraints:
    """Tests with mixed equality and inequality constraints."""

    def test_equality_constraint(self):
        """Problem with equality constraint.

        min x² + y²
        s.t. x + y = 2

        Optimal: (1, 1) with f* = 2
        """
        x = Variable("x")
        y = Variable("y")

        prob = Problem(name="equality_constraint")
        prob.minimize(x**2 + y**2)
        prob.subject_to((x + y).constraint_eq(2))

        sol = prob.solve(x0=np.array([0.0, 0.0]))

        assert sol.is_optimal
        assert abs(sol["x"] - 1.0) < 1e-4
        assert abs(sol["y"] - 1.0) < 1e-4
        assert abs(sol.objective_value - 2.0) < 1e-4

    def test_mixed_constraints(self):
        """Problem with both equality and inequality.

        min x² + y² + z²
        s.t. x + y + z = 3
             x >= 0, y >= 0, z >= 0
             x <= 1

        Optimal: (1, 1, 1) with f* = 3
        """
        x = Variable("x", lb=0, ub=1)
        y = Variable("y", lb=0)
        z = Variable("z", lb=0)

        prob = Problem(name="mixed_constraints")
        prob.minimize(x**2 + y**2 + z**2)
        prob.subject_to((x + y + z).constraint_eq(3))

        sol = prob.solve(x0=np.array([0.5, 0.5, 0.5]))

        assert sol.is_optimal
        assert abs(sol["x"] - 1.0) < 1e-2
        assert abs(sol["y"] - 1.0) < 1e-2
        assert abs(sol["z"] - 1.0) < 1e-2
        assert abs(sol.objective_value - 3.0) < 1e-2
