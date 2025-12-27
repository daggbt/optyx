"""Comparison benchmark: Optyx vs SciPy with vectorized operations.

Direct comparison with raw SciPy for both LP and NLP problems.
Uses numpy vectorization (@, np.sum, np.array) for optimal performance.
Generates plots comparing performance across problem sizes.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog, minimize

from optyx import Variable, Problem

import sys

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from utils import (
    compare_timing,
    time_function,
    ScalingData,
    RESULTS_DIR,
    plot_scaling_comparison,
    plot_overhead_breakdown,
)


class TestLPComparison:
    """Compare Optyx LP solver vs SciPy linprog with vectorized operations."""

    def test_small_lp(self):
        """Small LP: 2 variables, 2 constraints.

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

        optyx_sol = prob.solve()

        # SciPy
        scipy_sol = linprog(
            -c, A_ub=A, b_ub=b, bounds=[(0, None), (0, None)], method="highs"
        )

        # Compare solutions
        assert optyx_sol.is_optimal
        assert scipy_sol.success
        assert abs(optyx_sol.objective_value - (-scipy_sol.fun)) < 1e-4

        # Compare timing
        result = compare_timing(
            lambda: prob.solve(),
            lambda: linprog(
                -c, A_ub=A, b_ub=b, bounds=[(0, None), (0, None)], method="highs"
            ),
            n_warmup=3,
            n_runs=30,
        )
        print(f"\nSmall LP Comparison:\n{result}")

    def test_medium_lp(self):
        """Medium LP: 20 variables, 15 constraints using vectorized ops."""
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
            prob.subject_to(A[i] @ x <= b[i])  # Row-wise matrix multiplication

        optyx_sol = prob.solve()

        # SciPy
        bounds = [(0, 1)] * n
        scipy_sol = linprog(-c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

        # Compare solutions
        assert optyx_sol.is_optimal
        assert scipy_sol.success
        assert abs(optyx_sol.objective_value - (-scipy_sol.fun)) < 1e-3

        # Compare timing
        result = compare_timing(
            lambda: prob.solve(),
            lambda: linprog(-c, A_ub=A, b_ub=b, bounds=bounds, method="highs"),
            n_warmup=3,
            n_runs=30,
        )
        print(f"\nMedium LP Comparison:\n{result}")
        assert result.overhead_ratio < 2.0

    def test_large_lp(self):
        """Large LP: 100 variables, 50 constraints."""
        n, m = 100, 50
        np.random.seed(42)

        c = np.random.rand(n)
        A = np.random.rand(m, n)
        b = np.sum(A, axis=1) * 0.5

        # Optyx with vectorization
        x = np.array([Variable(f"x{i}", lb=0, ub=1) for i in range(n)])
        prob = Problem(name="large_lp")
        prob.maximize(c @ x)
        for i in range(m):
            prob.subject_to(A[i] @ x <= b[i])

        optyx_sol = prob.solve()

        # SciPy
        bounds = [(0, 1)] * n
        scipy_sol = linprog(-c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

        assert optyx_sol.is_optimal
        assert scipy_sol.success
        assert abs(optyx_sol.objective_value - (-scipy_sol.fun)) < 1e-2

        result = compare_timing(
            lambda: prob.solve(),
            lambda: linprog(-c, A_ub=A, b_ub=b, bounds=bounds, method="highs"),
            n_warmup=3,
            n_runs=20,
        )
        print(f"\nLarge LP Comparison:\n{result}")


class TestNLPComparison:
    """Compare Optyx NLP solver vs SciPy minimize."""

    def test_rosenbrock(self):
        """Rosenbrock function comparison."""
        x = Variable("x")
        y = Variable("y")
        prob = Problem(name="rosenbrock")
        prob.minimize((1 - x) ** 2 + 100 * (y - x**2) ** 2)

        x0 = np.array([-1.0, -1.0])
        optyx_sol = prob.solve(x0=x0)

        # SciPy
        def obj(v):
            return (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2

        def grad(v):
            return np.array(
                [
                    -2 * (1 - v[0]) - 400 * v[0] * (v[1] - v[0] ** 2),
                    200 * (v[1] - v[0] ** 2),
                ]
            )

        scipy_sol = minimize(obj, x0, jac=grad, method="BFGS")

        assert optyx_sol.is_optimal
        assert scipy_sol.success
        assert abs(optyx_sol["x"] - scipy_sol.x[0]) < 1e-3

        result = compare_timing(
            lambda: prob.solve(x0=x0),
            lambda: minimize(obj, x0, jac=grad, method="BFGS"),
            n_warmup=3,
            n_runs=20,
        )
        print(f"\nRosenbrock Comparison:\n{result}")

    def test_constrained_qp(self):
        """Constrained quadratic with vectorized operations."""
        x = np.array([Variable("x"), Variable("y")])
        prob = Problem(name="constrained_qp")
        prob.minimize(np.sum(x**2))  # ||x||² using vectorized ops
        prob.subject_to(np.sum(x) >= 1)

        x0 = np.array([0.0, 0.0])
        optyx_sol = prob.solve(x0=x0)

        # SciPy
        def obj(v):
            return np.sum(v**2)

        def grad(v):
            return 2 * v

        constraints = {
            "type": "ineq",
            "fun": lambda v: np.sum(v) - 1,
            "jac": lambda v: np.ones(2),
        }
        scipy_sol = minimize(obj, x0, jac=grad, method="SLSQP", constraints=constraints)

        assert optyx_sol.is_optimal
        assert scipy_sol.success

        result = compare_timing(
            lambda: prob.solve(x0=x0),
            lambda: minimize(
                obj, x0, jac=grad, method="SLSQP", constraints=constraints
            ),
            n_warmup=3,
            n_runs=20,
        )
        print(f"\nConstrained QP Comparison:\n{result}")


class TestScalingComparison:
    """Generate scaling comparison plots."""

    def test_lp_scaling_plot(self):
        """Generate LP scaling comparison plot."""
        sizes = [10, 25, 50, 100, 200]
        data = ScalingData(label="LP")

        for n in sizes:
            m = n // 2
            np.random.seed(42)

            c = np.random.rand(n)
            A = np.random.rand(m, n)
            b = np.sum(A, axis=1) * 0.5

            # Optyx (vectorized)
            x = np.array([Variable(f"x{i}", lb=0, ub=1) for i in range(n)])
            prob = Problem(name=f"lp_scale_{n}")
            prob.maximize(c @ x)
            for i in range(m):
                prob.subject_to(A[i] @ x <= b[i])

            prob.solve()  # Warm cache
            optyx_timing = time_function(lambda: prob.solve(), n_warmup=2, n_runs=20)

            # SciPy
            bounds = [(0, 1)] * n
            scipy_timing = time_function(
                lambda: linprog(-c, A_ub=A, b_ub=b, bounds=bounds, method="highs"),
                n_warmup=2,
                n_runs=20,
            )

            data.add_point(
                n,
                optyx_timing.mean_ms,
                optyx_timing.std_ms,
                scipy_timing.mean_ms,
                scipy_timing.std_ms,
            )

            print(
                f"n={n:4d}: Optyx={optyx_timing.mean_ms:.3f}ms, "
                f"SciPy={scipy_timing.mean_ms:.3f}ms, "
                f"ratio={optyx_timing.mean_ms / scipy_timing.mean_ms:.2f}x"
            )

        plot_scaling_comparison(
            data,
            title="LP Scaling: Optyx vs SciPy (Vectorized)",
            save_path=RESULTS_DIR / "scipy_lp_scaling.png",
        )

    def test_nlp_scaling_plot(self):
        """Generate NLP scaling comparison plot."""
        sizes = [10, 25, 50, 100]
        data = ScalingData(label="Quadratic NLP")

        for n in sizes:
            x0 = np.zeros(n)

            # Optyx (vectorized)
            x = np.array([Variable(f"x{i}") for i in range(n)])
            prob = Problem(name=f"nlp_scale_{n}")
            prob.minimize(np.sum(x**2) - np.sum(x))

            optyx_timing = time_function(
                lambda p=prob, x0=x0: p.solve(x0=x0), n_warmup=2, n_runs=10
            )

            # SciPy
            def scipy_obj(v):
                return np.sum(v**2) - np.sum(v)

            def scipy_grad(v):
                return 2 * v - 1

            scipy_timing = time_function(
                lambda: minimize(scipy_obj, x0, jac=scipy_grad, method="BFGS"),
                n_warmup=2,
                n_runs=10,
            )

            data.add_point(
                n,
                optyx_timing.mean_ms,
                optyx_timing.std_ms,
                scipy_timing.mean_ms,
                scipy_timing.std_ms,
            )

            print(
                f"n={n:4d}: Optyx={optyx_timing.mean_ms:.3f}ms, "
                f"SciPy={scipy_timing.mean_ms:.3f}ms, "
                f"ratio={optyx_timing.mean_ms / scipy_timing.mean_ms:.2f}x"
            )

        plot_scaling_comparison(
            data,
            title="NLP Scaling: Optyx vs SciPy (Vectorized)",
            save_path=RESULTS_DIR / "scipy_nlp_scaling.png",
        )


class TestOverheadBreakdown:
    """Analyze overhead by problem type."""

    def test_overhead_breakdown_plot(self):
        """Generate overhead breakdown plot across problem types."""
        categories = []
        overheads = []

        # Small LP
        c = np.array([20.0, 30.0])
        A = np.array([[4.0, 6.0], [2.0, 3.0]])
        b = np.array([120.0, 60.0])
        x = np.array([Variable("x", lb=0), Variable("y", lb=0)])
        prob = Problem(name="overhead_small_lp")
        prob.maximize(c @ x)
        prob.subject_to(A[0] @ x <= b[0])
        prob.subject_to(A[1] @ x <= b[1])

        prob.solve()
        optyx_t = time_function(lambda: prob.solve(), n_warmup=2, n_runs=50)
        scipy_t = time_function(
            lambda: linprog(
                -c, A_ub=A, b_ub=b, bounds=[(0, None), (0, None)], method="highs"
            ),
            n_warmup=2,
            n_runs=50,
        )
        categories.append("Small LP (n=2)")
        overheads.append(optyx_t.mean_ms / scipy_t.mean_ms)

        # Medium LP
        n, m = 50, 25
        np.random.seed(42)
        c = np.random.rand(n)
        A_mat = np.random.rand(m, n)
        b_vec = np.sum(A_mat, axis=1) * 0.5
        x = np.array([Variable(f"x{i}", lb=0, ub=1) for i in range(n)])
        prob = Problem(name="overhead_med_lp")
        prob.maximize(c @ x)
        for i in range(m):
            prob.subject_to(A_mat[i] @ x <= b_vec[i])

        prob.solve()
        optyx_t = time_function(lambda: prob.solve(), n_warmup=2, n_runs=30)
        bounds = [(0, 1)] * n
        scipy_t = time_function(
            lambda: linprog(-c, A_ub=A_mat, b_ub=b_vec, bounds=bounds, method="highs"),
            n_warmup=2,
            n_runs=30,
        )
        categories.append("Medium LP (n=50)")
        overheads.append(optyx_t.mean_ms / scipy_t.mean_ms)

        # Rosenbrock
        rx = Variable("rx")
        ry = Variable("ry")
        prob = Problem(name="overhead_rosenbrock")
        prob.minimize((1 - rx) ** 2 + 100 * (ry - rx**2) ** 2)
        x0 = np.array([-1.0, -1.0])

        optyx_t = time_function(lambda: prob.solve(x0=x0), n_warmup=2, n_runs=30)

        def ros_obj(v):
            return (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2

        def ros_grad(v):
            return np.array(
                [
                    -2 * (1 - v[0]) - 400 * v[0] * (v[1] - v[0] ** 2),
                    200 * (v[1] - v[0] ** 2),
                ]
            )

        scipy_t = time_function(
            lambda: minimize(ros_obj, x0, jac=ros_grad, method="BFGS"),
            n_warmup=2,
            n_runs=30,
        )
        categories.append("Rosenbrock")
        overheads.append(optyx_t.mean_ms / scipy_t.mean_ms)

        # Constrained QP
        qx = np.array([Variable("qx"), Variable("qy")])
        prob = Problem(name="overhead_cqp")
        prob.minimize(np.sum(qx**2))
        prob.subject_to(np.sum(qx) >= 1)

        optyx_t = time_function(
            lambda: prob.solve(x0=np.zeros(2)), n_warmup=2, n_runs=30
        )

        def qp_obj(v):
            return np.sum(v**2)

        def qp_grad(v):
            return 2 * v

        constraints = {
            "type": "ineq",
            "fun": lambda v: np.sum(v) - 1,
            "jac": lambda v: np.ones(2),
        }
        scipy_t = time_function(
            lambda: minimize(
                qp_obj,
                np.zeros(2),
                jac=qp_grad,
                method="SLSQP",
                constraints=constraints,
            ),
            n_warmup=2,
            n_runs=30,
        )
        categories.append("Constrained QP")
        overheads.append(optyx_t.mean_ms / scipy_t.mean_ms)

        # Print results
        print("\nOverhead Breakdown:")
        for cat, oh in zip(categories, overheads):
            print(f"  {cat}: {oh:.2f}x")

        # Generate plot
        plot_overhead_breakdown(
            categories,
            overheads,
            title="Optyx Overhead vs SciPy by Problem Type",
            save_path=RESULTS_DIR / "overhead_breakdown.png",
        )


class TestPortfolioComparison:
    """Portfolio optimization syntax and performance comparison."""

    def test_portfolio_vectorized(self):
        """Portfolio optimization with fully vectorized operations."""
        n = 10  # assets
        np.random.seed(42)

        returns = np.random.rand(n) * 0.1 + 0.05  # 5-15% returns
        cov = np.eye(n) * 0.04 + np.random.rand(n, n) * 0.01
        cov = (cov + cov.T) / 2  # Make symmetric

        # Optyx with vectorization
        w = np.array([Variable(f"w{i}", lb=0, ub=1) for i in range(n)])
        prob = Problem(name="portfolio_vectorized")

        # Vectorized objective: returns @ w - 0.5 * w @ cov @ w
        # But quadratic form needs to be element-wise for now
        expected_return = returns @ w
        # For variance, we need element-wise due to expression tree
        variance = sum(cov[i, j] * w[i] * w[j] for i in range(n) for j in range(n))
        prob.maximize(expected_return - 0.5 * variance)
        prob.subject_to(np.sum(w).constraint_eq(1))  # Budget constraint

        optyx_sol = prob.solve()

        # SciPy
        def obj(weights):
            ret = returns @ weights
            var = weights @ cov @ weights
            return -(ret - 0.5 * var)

        def grad(weights):
            return -(returns - cov @ weights)

        x0 = np.ones(n) / n
        bounds = [(0, 1)] * n
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
        scipy_sol = minimize(
            obj, x0, jac=grad, method="SLSQP", bounds=bounds, constraints=constraints
        )

        assert optyx_sol.is_optimal
        assert scipy_sol.success

        result = compare_timing(
            lambda: prob.solve(),
            lambda: minimize(
                obj,
                x0,
                jac=grad,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
            ),
            n_warmup=3,
            n_runs=20,
        )
        print(f"\nPortfolio Optimization Comparison (n={n} assets):\n{result}")

        optyx_weights = np.array([optyx_sol[f"w{i}"] for i in range(n)])
        print(f"  Weights sum: {np.sum(optyx_weights):.4f}")
        print(f"  Objective: {optyx_sol.objective_value:.6f}")


if __name__ == "__main__":
    print("=" * 60)
    print("OPTYX VS SCIPY COMPARISON (VECTORIZED)")
    print("=" * 60)

    # Run tests
    test_lp = TestLPComparison()
    test_lp.test_small_lp()
    test_lp.test_medium_lp()
    test_lp.test_large_lp()

    test_nlp = TestNLPComparison()
    test_nlp.test_rosenbrock()
    test_nlp.test_constrained_qp()

    # Generate plots
    print("\n" + "=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)

    test_scale = TestScalingComparison()
    test_scale.test_lp_scaling_plot()
    test_scale.test_nlp_scaling_plot()

    test_overhead = TestOverheadBreakdown()
    test_overhead.test_overhead_breakdown_plot()

    test_portfolio = TestPortfolioComparison()
    test_portfolio.test_portfolio_vectorized()

    print(f"\nPlots saved to: {RESULTS_DIR}")
