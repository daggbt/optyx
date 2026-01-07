"""Comparison benchmark: Optyx vs CVXPY.

Compares Optyx against CVXPY for convex problems.
CVXPY is an optional dependency - tests gracefully skip if not installed.

Install with: uv sync --extra benchmarks
"""

from __future__ import annotations

import numpy as np
import pytest

import sys

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from utils import compare_timing

# Check if cvxpy is available
try:
    import cvxpy as cp

    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

from optyx import Variable, Problem, VectorVariable


pytestmark = pytest.mark.skipif(not CVXPY_AVAILABLE, reason="cvxpy not installed")


class TestLPComparison:
    """Compare Optyx vs CVXPY for linear programs."""

    def test_small_lp(self):
        """Small LP: 2 variables, 2 constraints."""
        # Optyx
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)
        prob = Problem(name="small_lp")
        prob.maximize(20 * x + 30 * y)
        prob.subject_to(4 * x + 6 * y <= 120)
        prob.subject_to(2 * x + 3 * y <= 60)

        optyx_sol = prob.solve()

        # CVXPY
        x_cv = cp.Variable(nonneg=True)
        y_cv = cp.Variable(nonneg=True)
        objective = cp.Maximize(20 * x_cv + 30 * y_cv)
        constraints = [
            4 * x_cv + 6 * y_cv <= 120,
            2 * x_cv + 3 * y_cv <= 60,
        ]
        cvxpy_prob = cp.Problem(objective, constraints)
        cvxpy_prob.solve()

        # Compare solutions
        assert optyx_sol.is_optimal
        assert cvxpy_prob.status == cp.OPTIMAL
        assert abs(optyx_sol.objective_value - cvxpy_prob.value) < 1e-3

        # Compare timing
        def optyx_run():
            return prob.solve()

        def cvxpy_run():
            cvxpy_prob.solve()
            return cvxpy_prob.value

        result = compare_timing(optyx_run, cvxpy_run, n_warmup=3, n_runs=20)
        print(f"\nSmall LP - Optyx vs CVXPY:\n{result}")

    def test_medium_lp(self):
        """Medium LP: 20 variables, 15 constraints (VectorVariable)."""
        n, m = 20, 15
        np.random.seed(42)

        c = np.random.rand(n)
        A = np.random.rand(m, n)
        b = np.sum(A, axis=1) * 0.5

        # Optyx with VectorVariable
        x = VectorVariable("x", n, lb=0, ub=1)
        prob = Problem(name="medium_lp")
        prob.maximize(c @ x)  # Vectorized objective
        for i in range(m):
            prob.subject_to(A[i] @ x <= b[i])  # Vectorized constraints

        optyx_sol = prob.solve()

        # CVXPY
        x_cv = cp.Variable(n, nonneg=True)
        objective = cp.Maximize(c @ x_cv)
        constraints = [A @ x_cv <= b, x_cv <= 1]
        cvxpy_prob = cp.Problem(objective, constraints)
        cvxpy_prob.solve()

        # Compare solutions
        assert optyx_sol.is_optimal
        assert cvxpy_prob.status == cp.OPTIMAL
        assert abs(optyx_sol.objective_value - cvxpy_prob.value) < 1e-2

        # Compare timing
        def optyx_run():
            return prob.solve()

        def cvxpy_run():
            cvxpy_prob.solve()
            return cvxpy_prob.value

        result = compare_timing(optyx_run, cvxpy_run, n_warmup=3, n_runs=20)
        print(f"\nMedium LP (n=20) - Optyx VectorVariable vs CVXPY:\n{result}")


class TestQPComparison:
    """Compare Optyx vs CVXPY for quadratic programs."""

    def test_simple_qp(self):
        """Simple QP: min x² + y² s.t. x + y >= 1."""
        # Optyx
        x = Variable("x")
        y = Variable("y")
        prob = Problem(name="simple_qp")
        prob.minimize(x**2 + y**2)
        prob.subject_to(x + y >= 1)

        x0 = np.array([0.0, 0.0])
        optyx_sol = prob.solve(x0=x0)

        # CVXPY
        x_cv = cp.Variable()
        y_cv = cp.Variable()
        objective = cp.Minimize(x_cv**2 + y_cv**2)
        constraints = [x_cv + y_cv >= 1]
        cvxpy_prob = cp.Problem(objective, constraints)
        cvxpy_prob.solve()

        # Compare solutions
        assert optyx_sol.is_optimal
        assert cvxpy_prob.status == cp.OPTIMAL
        assert abs(optyx_sol["x"] - x_cv.value) < 1e-3
        assert abs(optyx_sol["y"] - y_cv.value) < 1e-3

        # Compare timing
        def optyx_run():
            return prob.solve(x0=x0)

        def cvxpy_run():
            cvxpy_prob.solve()
            return cvxpy_prob.value

        result = compare_timing(optyx_run, cvxpy_run, n_warmup=3, n_runs=20)
        print(f"\nSimple QP - Optyx vs CVXPY:\n{result}")

    def test_portfolio_qp(self):
        """Portfolio optimization QP with math-like syntax.

        Uses Optyx's w.dot(Σ @ w) for efficient wᵀΣw computation
        with analytic gradients, compared against CVXPY's quad_form.
        """
        n = 10  # assets
        np.random.seed(42)

        returns = np.random.rand(n) * 0.1 + 0.05
        cov = np.eye(n) * 0.04 + np.random.rand(n, n) * 0.01
        cov = (cov + cov.T) / 2  # Symmetrize

        # Optyx with VectorVariable and math-like quadratic form
        w = VectorVariable("w", n, lb=0, ub=1)
        prob = Problem(name="portfolio")
        expected_return = returns @ w  # LinearCombination
        variance = w.dot(cov @ w)  # Math-like: w · (Σw) = wᵀΣw
        prob.maximize(expected_return - 0.5 * variance)
        prob.subject_to(w.sum().eq(1))

        # Auto method selection (now defaults to SLSQP for constrained problems)
        optyx_sol = prob.solve()

        # CVXPY
        w_cv = cp.Variable(n, nonneg=True)
        ret = returns @ w_cv
        var = cp.quad_form(w_cv, cov)
        objective = cp.Maximize(ret - 0.5 * var)
        constraints = [cp.sum(w_cv) == 1, w_cv <= 1]
        cvxpy_prob = cp.Problem(objective, constraints)
        cvxpy_prob.solve()

        # Compare solutions
        assert optyx_sol.is_optimal
        assert cvxpy_prob.status == cp.OPTIMAL

        # Compare timing (warm solves)
        def optyx_run():
            return prob.solve()

        def cvxpy_run():
            cvxpy_prob.solve()
            return cvxpy_prob.value

        result = compare_timing(optyx_run, cvxpy_run, n_warmup=3, n_runs=10)
        print(
            f"\nPortfolio QP (n=10) - Optyx w.dot(Σ @ w) vs CVXPY quad_form:\n{result}"
        )

    def test_large_portfolio_qp(self):
        """Larger portfolio optimization (n=50) to show scaling."""
        n = 50  # assets
        np.random.seed(42)

        returns = np.random.rand(n) * 0.1 + 0.05
        cov = np.eye(n) * 0.04 + np.random.rand(n, n) * 0.01
        cov = (cov + cov.T) / 2  # Symmetrize

        # Optyx with VectorVariable and math-like quadratic form
        w = VectorVariable("w", n, lb=0, ub=1)
        prob = Problem(name="portfolio_large")
        expected_return = returns @ w
        variance = w.dot(cov @ w)  # Math-like: w · (Σw) = wᵀΣw
        prob.maximize(expected_return - 0.5 * variance)
        prob.subject_to(w.sum().eq(1))

        # Auto method selection (now defaults to SLSQP for constrained problems)
        optyx_sol = prob.solve()

        # CVXPY
        w_cv = cp.Variable(n, nonneg=True)
        ret = returns @ w_cv
        var = cp.quad_form(w_cv, cov)
        objective = cp.Maximize(ret - 0.5 * var)
        constraints = [cp.sum(w_cv) == 1, w_cv <= 1]
        cvxpy_prob = cp.Problem(objective, constraints)
        cvxpy_prob.solve()

        # Compare solutions
        assert optyx_sol.is_optimal
        assert cvxpy_prob.status == cp.OPTIMAL

        # Compare timing (warm solves)
        def optyx_run():
            return prob.solve()

        def cvxpy_run():
            cvxpy_prob.solve()
            return cvxpy_prob.value

        result = compare_timing(optyx_run, cvxpy_run, n_warmup=3, n_runs=10)
        print(
            f"\nPortfolio QP (n=50) - Optyx w.dot(Σ @ w) vs CVXPY quad_form:\n{result}"
        )


if __name__ == "__main__":
    if not CVXPY_AVAILABLE:
        print("CVXPY not installed. Install with: uv sync --extra benchmarks")
        exit(1)

    print("=" * 60)
    print("OPTYX VS CVXPY COMPARISON")
    print("=" * 60)

    test_lp = TestLPComparison()
    test_lp.test_small_lp()
    test_lp.test_medium_lp()

    test_qp = TestQPComparison()
    test_qp.test_simple_qp()
    test_qp.test_portfolio_qp()
    test_qp.test_large_portfolio_qp()
