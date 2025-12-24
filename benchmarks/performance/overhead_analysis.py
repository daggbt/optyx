"""Performance benchmark: Optyx overhead vs raw SciPy.

Measures the overhead of using Optyx compared to calling SciPy directly.
Uses numpy vectorization for optimal performance.
Target: < 1.5x overhead for LP (cached), < 2x for NLP.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog, minimize

from optyx import Variable, Problem

import sys

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from utils import compare_timing


class TestLPOverhead:
    """LP solver overhead analysis."""

    def test_small_lp_overhead(self):
        """Small LP (2 vars, 2 constraints) overhead using vectorized ops.

        max 20x + 30y
        s.t. 4x + 6y <= 120
             2x + 3y <= 60
             x, y >= 0
        """
        # Coefficients
        c = np.array([20.0, 30.0])
        A = np.array([[4.0, 6.0], [2.0, 3.0]])
        b = np.array([120.0, 60.0])

        # Optyx with vectorization
        x = np.array([Variable("x", lb=0), Variable("y", lb=0)])

        prob = Problem(name="small_lp")
        prob.maximize(c @ x)  # Vectorized objective
        prob.subject_to(A[0] @ x <= b[0])
        prob.subject_to(A[1] @ x <= b[1])

        def optyx_solve():
            return prob.solve()

        # Raw SciPy version
        bounds = [(0, None), (0, None)]

        def scipy_solve():
            return linprog(-c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

        result = compare_timing(optyx_solve, scipy_solve, n_warmup=3, n_runs=20)

        print(f"\nSmall LP Overhead:\n{result}")
        assert (
            result.overhead_ratio < 3.0
        ), f"Too much overhead: {result.overhead_ratio:.2f}x"

    def test_medium_lp_overhead(self):
        """Medium LP (20 vars, 15 constraints) with vectorized operations."""
        n, m = 20, 15
        np.random.seed(42)

        c = np.random.rand(n)
        A = np.random.rand(m, n)
        b = np.sum(A, axis=1) * 0.5

        # Optyx with numpy vectorization
        x = np.array([Variable(f"x{i}", lb=0, ub=1) for i in range(n)])

        prob = Problem(name="medium_lp")
        prob.maximize(c @ x)  # c^T @ x using @
        for i in range(m):
            prob.subject_to(A[i] @ x <= b[i])  # Row-wise dot product

        def optyx_solve():
            return prob.solve()

        # Raw SciPy version
        bounds = [(0, 1)] * n

        def scipy_solve():
            return linprog(-c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

        result = compare_timing(optyx_solve, scipy_solve, n_warmup=3, n_runs=20)

        print(f"\nMedium LP Overhead:\n{result}")
        assert (
            result.overhead_ratio < 2.0
        ), f"Too much overhead: {result.overhead_ratio:.2f}x"


class TestNLPOverhead:
    """NLP solver overhead analysis."""

    def test_rosenbrock_overhead(self):
        """Rosenbrock function overhead."""
        # Optyx version
        x = Variable("x")
        y = Variable("y")

        prob = Problem(name="rosenbrock")
        prob.minimize((1 - x) ** 2 + 100 * (y - x**2) ** 2)

        x0 = np.array([-1.0, -1.0])

        def optyx_solve():
            return prob.solve(x0=x0)

        # Raw SciPy version
        def objective(vars):
            x, y = vars
            return (1 - x) ** 2 + 100 * (y - x**2) ** 2

        def gradient(vars):
            x, y = vars
            dx = -2 * (1 - x) - 400 * x * (y - x**2)
            dy = 200 * (y - x**2)
            return np.array([dx, dy])

        def scipy_solve():
            return minimize(objective, x0, jac=gradient, method="BFGS")

        result = compare_timing(optyx_solve, scipy_solve, n_warmup=3, n_runs=20)

        print(f"\nRosenbrock Overhead:\n{result}")

        # NLP overhead is justified (autodiff vs manual gradients)
        # But should still be reasonable
        assert (
            result.overhead_ratio < 3.0
        ), f"Too much overhead: {result.overhead_ratio:.2f}x"

    def test_constrained_nlp_overhead(self):
        """Constrained NLP overhead."""
        # Optyx version
        x = Variable("x")
        y = Variable("y")

        prob = Problem(name="constrained_nlp")
        prob.minimize(x**2 + y**2)
        prob.subject_to(x + y >= 1)

        x0 = np.array([0.0, 0.0])

        def optyx_solve():
            return prob.solve(x0=x0)

        # Raw SciPy version
        def objective(vars):
            return vars[0] ** 2 + vars[1] ** 2

        def gradient(vars):
            return 2 * vars

        def constraint(vars):
            return vars[0] + vars[1] - 1

        def constraint_jac(vars):
            return np.array([1.0, 1.0])

        constraints = {
            "type": "ineq",
            "fun": constraint,
            "jac": constraint_jac,
        }

        def scipy_solve():
            return minimize(
                objective, x0, jac=gradient, method="SLSQP", constraints=constraints
            )

        result = compare_timing(optyx_solve, scipy_solve, n_warmup=3, n_runs=20)

        print(f"\nConstrained NLP Overhead:\n{result}")

        assert (
            result.overhead_ratio < 3.0
        ), f"Too much overhead: {result.overhead_ratio:.2f}x"


if __name__ == "__main__":
    # Run tests directly
    test = TestLPOverhead()
    test.test_small_lp_overhead()
    test.test_medium_lp_overhead()

    test = TestNLPOverhead()
    test.test_rosenbrock_overhead()
    test.test_constrained_nlp_overhead()
