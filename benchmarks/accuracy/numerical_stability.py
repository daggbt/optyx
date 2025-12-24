"""Accuracy benchmark: Numerical stability.

Tests Optyx behavior with edge cases:
- Large/small coefficients
- Near-zero values
- Ill-conditioned problems
"""

from __future__ import annotations

import numpy as np

from optyx import Variable, Problem, exp, log

import sys

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])


class TestLargeCoefficients:
    """Test problems with large coefficients."""

    def test_large_linear_coefficients(self):
        """LP with large coefficients: max 1e6*x + 1e6*y"""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem(name="large_coeffs")
        prob.maximize(1e6 * x + 1e6 * y)
        prob.subject_to(x + y <= 10)

        sol = prob.solve()

        assert sol.is_optimal
        assert abs(sol.objective_value - 1e7) < 1e3  # 10 * 1e6

    def test_large_quadratic_coefficients(self):
        """QP with large coefficients."""
        x = Variable("x")
        y = Variable("y")

        prob = Problem(name="large_qp")
        prob.minimize(1e4 * x**2 + 1e4 * y**2)
        prob.subject_to(x + y >= 1)

        sol = prob.solve(x0=np.array([0.0, 0.0]))

        assert sol.is_optimal
        assert abs(sol["x"] - 0.5) < 1e-3
        assert abs(sol["y"] - 0.5) < 1e-3

    def test_mixed_scale_coefficients(self):
        """Problem with mixed scales: 1e6 and 1e-6."""
        x = Variable("x")
        y = Variable("y")

        prob = Problem(name="mixed_scale")
        prob.minimize(1e6 * x**2 + 1e-6 * y**2)

        sol = prob.solve(x0=np.array([0.0, 0.0]))

        assert sol.is_optimal
        assert abs(sol["x"]) < 1e-4
        assert abs(sol["y"]) < 1e-4


class TestSmallCoefficients:
    """Test problems with small coefficients."""

    def test_small_linear_coefficients(self):
        """LP with small coefficients."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem(name="small_coeffs")
        prob.maximize(1e-6 * x + 1e-6 * y)
        prob.subject_to(x + y <= 10)

        sol = prob.solve()

        assert sol.is_optimal
        # Objective should be approximately 10 * 1e-6 = 1e-5
        assert abs(sol.objective_value - 1e-5) < 1e-7

    def test_small_constraint_rhs(self):
        """Constraint with small RHS."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem(name="small_rhs")
        prob.maximize(x + y)
        prob.subject_to(x + y <= 1e-6)

        sol = prob.solve()

        assert sol.is_optimal
        assert sol.objective_value < 1e-5


class TestNearZeroValues:
    """Test behavior near zero."""

    def test_optimum_at_origin(self):
        """Problem with optimum at origin."""
        x = Variable("x")
        y = Variable("y")

        prob = Problem(name="origin_optimum")
        prob.minimize(x**2 + y**2)

        sol = prob.solve(x0=np.array([1.0, 1.0]))

        assert sol.is_optimal
        assert abs(sol["x"]) < 1e-4
        assert abs(sol["y"]) < 1e-4
        assert abs(sol.objective_value) < 1e-6

    def test_near_zero_constraint_slack(self):
        """Active constraint at optimum (zero slack)."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem(name="active_constraint")
        prob.minimize(x + y)
        prob.subject_to(x + y >= 1)

        sol = prob.solve()

        assert sol.is_optimal
        # Constraint should be active: x + y = 1
        assert abs(sol["x"] + sol["y"] - 1) < 1e-6


class TestIllConditionedProblems:
    """Test ill-conditioned problems."""

    def test_near_singular_constraint_matrix(self):
        """Nearly parallel constraints."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem(name="near_singular")
        prob.maximize(x + y)
        prob.subject_to(x + y <= 10)
        prob.subject_to(1.001 * x + y <= 10.01)  # Nearly parallel

        sol = prob.solve()

        assert sol.is_optimal

    def test_elongated_ellipsoid(self):
        """Highly elongated ellipsoid (poor conditioning)."""
        x = Variable("x")
        y = Variable("y")

        prob = Problem(name="elongated")
        # Condition number ≈ 1e6
        prob.minimize(x**2 + 1e6 * y**2)

        sol = prob.solve(x0=np.array([1.0, 1.0]))

        assert sol.is_optimal
        assert abs(sol["x"]) < 1e-3
        assert abs(sol["y"]) < 1e-6


class TestBoundaryBehavior:
    """Test behavior at variable bounds."""

    def test_optimum_at_lower_bound(self):
        """Optimum at lower bound."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem(name="lower_bound")
        prob.minimize(x + y)

        sol = prob.solve()

        assert sol.is_optimal
        assert abs(sol["x"]) < 1e-6
        assert abs(sol["y"]) < 1e-6

    def test_optimum_at_upper_bound(self):
        """Optimum at upper bound."""
        x = Variable("x", lb=0, ub=5)
        y = Variable("y", lb=0, ub=5)

        prob = Problem(name="upper_bound")
        prob.maximize(x + y)

        sol = prob.solve()

        assert sol.is_optimal
        assert abs(sol["x"] - 5) < 1e-6
        assert abs(sol["y"] - 5) < 1e-6

    def test_tight_bounds(self):
        """Very tight bounds forcing specific solution."""
        x = Variable("x", lb=2.999, ub=3.001)
        y = Variable("y", lb=1.999, ub=2.001)

        prob = Problem(name="tight_bounds")
        prob.minimize(x**2 + y**2)

        sol = prob.solve()

        assert sol.is_optimal
        assert abs(sol["x"] - 3) < 0.01
        assert abs(sol["y"] - 2) < 0.01


class TestNumericalOverflow:
    """Test potential overflow scenarios."""

    def test_large_exponent_input(self):
        """Avoid overflow with large exp() inputs."""
        x = Variable("x", lb=-10, ub=10)

        prob = Problem(name="large_exp")
        prob.minimize(exp(x) + exp(-x))

        sol = prob.solve(x0=np.array([0.0]))

        assert sol.is_optimal
        # min at x=0 where f=2
        assert abs(sol["x"]) < 0.1
        assert abs(sol.objective_value - 2) < 0.1

    def test_log_near_zero(self):
        """Handle log() near zero gracefully."""
        x = Variable("x", lb=0.01, ub=10)

        prob = Problem(name="log_near_zero")
        prob.minimize(
            log(x) + x
        )  # min at x where 1/x + 1 = 0, i.e., x=-1 (infeasible), so min at boundary

        sol = prob.solve(x0=np.array([1.0]))

        # Solution exists within bounds
        assert sol.is_optimal
        assert 0.01 <= sol["x"] <= 10


if __name__ == "__main__":
    print("=" * 60)
    print("NUMERICAL STABILITY TESTS")
    print("=" * 60)

    test_large = TestLargeCoefficients()
    test_large.test_large_linear_coefficients()
    test_large.test_large_quadratic_coefficients()
    test_large.test_mixed_scale_coefficients()
    print("✓ Large coefficients OK")

    test_small = TestSmallCoefficients()
    test_small.test_small_linear_coefficients()
    test_small.test_small_constraint_rhs()
    print("✓ Small coefficients OK")

    test_zero = TestNearZeroValues()
    test_zero.test_optimum_at_origin()
    test_zero.test_near_zero_constraint_slack()
    print("✓ Near-zero values OK")

    test_ill = TestIllConditionedProblems()
    test_ill.test_near_singular_constraint_matrix()
    test_ill.test_elongated_ellipsoid()
    print("✓ Ill-conditioned problems OK")

    test_bound = TestBoundaryBehavior()
    test_bound.test_optimum_at_lower_bound()
    test_bound.test_optimum_at_upper_bound()
    test_bound.test_tight_bounds()
    print("✓ Boundary behavior OK")

    test_overflow = TestNumericalOverflow()
    test_overflow.test_large_exponent_input()
    test_overflow.test_log_near_zero()
    print("✓ Numerical overflow handling OK")
