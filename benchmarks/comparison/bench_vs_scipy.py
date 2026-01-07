"""Comparison benchmark: Optyx vs SciPy with vectorized operations.

Direct comparison with raw SciPy for both LP and NLP problems.
Uses VectorVariable and MatrixVariable for optimal performance.
Generates plots comparing performance across problem sizes.

Key features demonstrated:
1. VectorVariable: Efficient 1D variable arrays with @ syntax
2. MatrixVariable: Efficient 2D variable matrices with slicing/sum()
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog, minimize

from optyx import VectorVariable, MatrixVariable, Problem

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

        # Optyx with VectorVariable
        x = VectorVariable("x", 2, lb=0)
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

        # Optyx with VectorVariable
        x = VectorVariable("x", n, lb=0, ub=1)
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

        # Optyx with VectorVariable
        x = VectorVariable("x", n, lb=0, ub=1)
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


class TestScalingComparison:
    """Generate scaling comparison plots."""

    def test_lp_scaling_plot(self):
        """Generate LP scaling comparison plot using VectorVariable.

        Scales up to n=2,000 for plot (larger sizes shown in text output).
        """
        sizes = [100, 500, 1000, 2000]
        data = ScalingData(label="LP (VectorVariable)")

        print("\n" + "=" * 70)
        print("LP SCALING PLOT: Optyx VectorVariable vs SciPy (n up to 2,000)")
        print("=" * 70)

        for n in sizes:
            m = n // 2
            np.random.seed(42)

            c = np.random.rand(n)
            A = np.random.rand(m, n)
            b = np.sum(A, axis=1) * 0.5

            # Optyx with VectorVariable - O(1) formulation
            x = VectorVariable("x", n, lb=0, ub=1)
            prob = Problem(name=f"lp_scale_{n}")
            prob.maximize(c @ x)
            for i in range(m):
                prob.subject_to(A[i] @ x <= b[i])

            prob.solve()  # Warm cache

            # Reduce runs for large problems (solve time dominates)
            n_runs = 5 if n <= 2000 else 3 if n <= 5000 else 2
            optyx_timing = time_function(
                lambda: prob.solve(), n_warmup=1, n_runs=n_runs
            )

            # SciPy
            bounds = [(0, 1)] * n
            scipy_timing = time_function(
                lambda c=c, A=A, b=b, bounds=bounds: linprog(
                    -c, A_ub=A, b_ub=b, bounds=bounds, method="highs"
                ),
                n_warmup=1,
                n_runs=n_runs,
            )

            data.add_point(
                n,
                optyx_timing.mean_ms,
                optyx_timing.std_ms,
                scipy_timing.mean_ms,
                scipy_timing.std_ms,
            )

            ratio = optyx_timing.mean_ms / scipy_timing.mean_ms
            print(
                f"n={n:5d}: Optyx={optyx_timing.mean_ms:8.1f}ms, "
                f"SciPy={scipy_timing.mean_ms:8.1f}ms, "
                f"ratio={ratio:.2f}x"
            )

        plot_scaling_comparison(
            data,
            title="LP Scaling: Optyx VectorVariable vs SciPy (n up to 2,000)",
            save_path=RESULTS_DIR / "scipy_lp_scaling.png",
        )


class TestOverheadBreakdown:
    """Analyze overhead by problem type."""

    def test_overhead_breakdown_plot(self):
        """Generate overhead breakdown plot across problem types."""
        categories = []
        overheads = []

        print("\n" + "=" * 70)
        print("OVERHEAD BREAKDOWN BY PROBLEM TYPE")
        print("=" * 70)

        # Small LP
        c = np.array([20.0, 30.0])
        A = np.array([[4.0, 6.0], [2.0, 3.0]])
        b = np.array([120.0, 60.0])
        x = VectorVariable("x", 2, lb=0)
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
        print(
            f"Small LP (n=2): {optyx_t.mean_ms:.2f}ms vs {scipy_t.mean_ms:.2f}ms = {overheads[-1]:.2f}x"
        )

        # Medium LP with VectorVariable
        n, m = 500, 250
        np.random.seed(42)
        c = np.random.rand(n)
        A_mat = np.random.rand(m, n)
        b_vec = np.sum(A_mat, axis=1) * 0.5
        x = VectorVariable("x", n, lb=0, ub=1)
        prob = Problem(name="overhead_med_lp")
        prob.maximize(c @ x)
        for i in range(m):
            prob.subject_to(A_mat[i] @ x <= b_vec[i])

        prob.solve()
        optyx_t = time_function(lambda: prob.solve(), n_warmup=2, n_runs=20)
        bounds = [(0, 1)] * n
        scipy_t = time_function(
            lambda c=c, A_mat=A_mat, b_vec=b_vec, bounds=bounds: linprog(
                -c, A_ub=A_mat, b_ub=b_vec, bounds=bounds, method="highs"
            ),
            n_warmup=2,
            n_runs=20,
        )
        categories.append("Medium LP (n=500)")
        overheads.append(optyx_t.mean_ms / scipy_t.mean_ms)
        print(
            f"Medium LP (n=500): {optyx_t.mean_ms:.2f}ms vs {scipy_t.mean_ms:.2f}ms = {overheads[-1]:.2f}x"
        )

        # Large LP with VectorVariable
        n, m = 2000, 1000
        np.random.seed(42)
        c = np.random.rand(n)
        A_mat = np.random.rand(m, n)
        b_vec = np.sum(A_mat, axis=1) * 0.5
        x = VectorVariable("x", n, lb=0, ub=1)
        prob = Problem(name="overhead_large_lp")
        prob.maximize(c @ x)
        for i in range(m):
            prob.subject_to(A_mat[i] @ x <= b_vec[i])

        prob.solve()
        optyx_t = time_function(lambda: prob.solve(), n_warmup=1, n_runs=10)
        bounds = [(0, 1)] * n
        scipy_t = time_function(
            lambda c=c, A_mat=A_mat, b_vec=b_vec, bounds=bounds: linprog(
                -c, A_ub=A_mat, b_ub=b_vec, bounds=bounds, method="highs"
            ),
            n_warmup=1,
            n_runs=10,
        )
        categories.append("Large LP (n=2000)")
        overheads.append(optyx_t.mean_ms / scipy_t.mean_ms)
        print(
            f"Large LP (n=2000): {optyx_t.mean_ms:.2f}ms vs {scipy_t.mean_ms:.2f}ms = {overheads[-1]:.2f}x"
        )

        # Print results
        print("\nOverhead Breakdown:")
        for cat, oh in zip(categories, overheads):
            print(f"  {cat}: {oh:.2f}x")

        # Generate plot
        plot_overhead_breakdown(
            categories,
            overheads,
            title="Optyx Overhead vs SciPy by Problem Type",
            save_path=RESULTS_DIR / "bench_vs_scipy_overhead_breakdown.png",
        )


class TestPortfolioComparison:
    """Portfolio optimization syntax and performance comparison."""

    def test_portfolio_vectorized(self):
        """Portfolio LP optimization with VectorVariable at scale.

        Tests linear portfolio optimization (maximize expected return)
        to demonstrate VectorVariable performance. Uses LP formulation
        to avoid deep expression trees from quadratic variance terms.
        """
        sizes = [50, 100, 200, 500, 1000]

        print("\n" + "=" * 70)
        print("PORTFOLIO LP: Optyx VectorVariable vs SciPy (maximize return)")
        print("=" * 70)

        for n in sizes:
            np.random.seed(42)

            returns = np.random.rand(n) * 0.1 + 0.05  # 5-15% returns
            max_weight = 0.2  # Maximum 20% in any single asset

            # Optyx with VectorVariable
            w = VectorVariable("w", n, lb=0, ub=max_weight)
            prob = Problem(name=f"portfolio_lp_{n}")

            # Linear objective: maximize expected return
            expected_return = returns @ w
            prob.maximize(expected_return)

            # Budget constraint: sum of weights = 1
            # Use np.ones() @ w for efficient LinearCombination
            ones = np.ones(n)
            prob.subject_to((ones @ w).eq(1))

            # SciPy
            x0 = np.ones(n) / n
            bounds = [(0, max_weight)] * n

            n_runs = 15 if n <= 200 else 10 if n <= 500 else 5

            # Warmup
            prob.solve()

            optyx_timing = time_function(
                lambda p=prob: p.solve(), n_warmup=2, n_runs=n_runs
            )
            scipy_timing = time_function(
                lambda returns=returns, x0=x0, bounds=bounds: linprog(
                    -returns,
                    A_eq=np.ones((1, len(returns))),
                    b_eq=np.array([1.0]),
                    bounds=bounds,
                    method="highs",
                ),
                n_warmup=2,
                n_runs=n_runs,
            )

            ratio = optyx_timing.mean_ms / scipy_timing.mean_ms
            print(
                f"n={n:4d} assets: Optyx={optyx_timing.mean_ms:8.1f}ms, "
                f"SciPy={scipy_timing.mean_ms:8.1f}ms, ratio={ratio:.2f}x"
            )


class TestVectorVariableScaling:
    """VectorVariable scaling: build and solve vs SciPy."""

    def test_lp_formulation_scaling(self):
        """VectorVariable LP formulation scaling.

        VectorVariable construction is O(1). Constraint addition is O(m).
        Total formulation scales to 10,000+ variables easily.
        """
        sizes = [100, 1000, 2000, 5000, 10000]

        print("\n" + "=" * 70)
        print("LP FORMULATION: VectorVariable Scaling")
        print("=" * 70)
        print(f"{'n':>8} | {'m':>6} | {'Build (ms)':>12}")
        print("-" * 70)

        for n in sizes:
            m = n // 4  # Constraints = n/4
            np.random.seed(42)

            c = np.random.rand(n)
            A = np.random.rand(m, n)
            b = np.sum(A, axis=1) * 0.5

            def build_vector():
                x = VectorVariable("x", n, lb=0, ub=1)
                prob = Problem(name="lp_vector")
                prob.maximize(c @ x)
                for i in range(m):
                    prob.subject_to(A[i] @ x <= b[i])
                return prob

            timing = time_function(build_vector, n_warmup=1, n_runs=5)

            print(f"{n:>8} | {m:>6} | {timing.mean_ms:>12.2f}")

        print("-" * 70)

    def test_lp_solve_comparison(self):
        """Compare VectorVariable solve performance vs SciPy."""
        sizes = [100, 500, 1000, 2000]

        print("\n" + "=" * 70)
        print("LP SOLVE COMPARISON: VectorVariable vs SciPy")
        print("=" * 70)
        print(f"{'n':>6} | {'Optyx':>10} | {'SciPy':>10} | {'Ratio':>10}")
        print("-" * 70)

        for n in sizes:
            m = n // 4
            np.random.seed(42)

            c = np.random.rand(n)
            A = np.random.rand(m, n)
            b = np.sum(A, axis=1) * 0.5
            bounds_list = [(0, 1)] * n

            # Build VectorVariable problem
            x_vec = VectorVariable("x", n, lb=0, ub=1)
            prob_vec = Problem(name="lp_vector")
            prob_vec.maximize(c @ x_vec)
            for i in range(m):
                prob_vec.subject_to(A[i] @ x_vec <= b[i])

            # Warm up
            prob_vec.solve()

            # Time solves
            vector_timing = time_function(
                lambda: prob_vec.solve(), n_warmup=2, n_runs=10
            )
            scipy_timing = time_function(
                lambda: linprog(-c, A_ub=A, b_ub=b, bounds=bounds_list, method="highs"),
                n_warmup=2,
                n_runs=10,
            )

            ratio = vector_timing.mean_ms / scipy_timing.mean_ms

            print(
                f"{n:>6} | {vector_timing.mean_ms:>9.2f}ms | "
                f"{scipy_timing.mean_ms:>9.2f}ms | {ratio:>9.2f}x"
            )

        print("-" * 70)


class TestMatrixVariableScaling:
    """MatrixVariable scaling and features."""

    def test_assignment_problem_scaling(self):
        """Assignment problem using MatrixVariable with row/column slicing."""
        sizes = [10, 25, 50, 100, 150]

        print("\n" + "=" * 70)
        print("ASSIGNMENT PROBLEM: MatrixVariable Scaling")
        print("=" * 70)
        print(f"{'n':>6} | {'Variables':>10} | {'Build (ms)':>12}")
        print("-" * 70)

        for n in sizes:
            np.random.seed(42)
            cost = np.random.rand(n, n)

            def build_matrix():
                X = MatrixVariable("X", n, n, lb=0, ub=1)
                prob = Problem(name="assign_matrix")
                # Objective: sum of cost-weighted variables
                prob.minimize(
                    sum(cost[i, j] * X[i, j] for i in range(n) for j in range(n))
                )
                # Row constraints using VectorVariable.sum()
                for i in range(n):
                    prob.subject_to(X[i, :].sum().eq(1))
                # Column constraints using VectorVariable.sum()
                for j in range(n):
                    prob.subject_to(X[:, j].sum().eq(1))
                return prob

            timing = time_function(build_matrix, n_warmup=1, n_runs=3)

            print(f"{n:>6} | {n * n:>10} | {timing.mean_ms:>11.2f}ms")

        print("-" * 70)

    def test_symmetric_matrix_scaling(self):
        """Symmetric MatrixVariable: uses n(n+1)/2 variables instead of n²."""
        sizes = [10, 25, 50, 100, 150]

        print("\n" + "=" * 70)
        print("SYMMETRIC MATRIX: Variable Count Reduction")
        print("=" * 70)
        print(f"{'n':>6} | {'Full Vars':>10} | {'Sym Vars':>10} | {'Build (ms)':>12}")
        print("-" * 70)

        for n in sizes:
            # Full matrix has n^2 variables
            full_vars = n * n
            # Symmetric has n*(n+1)/2 variables
            sym_vars = n * (n + 1) // 2

            def build_symmetric():
                X = MatrixVariable("X", n, n, lb=-1, ub=1, symmetric=True)
                prob = Problem(name="sym_matrix")
                prob.minimize(sum(X[i, j] ** 2 for i in range(n) for j in range(n)))
                return prob

            timing = time_function(build_symmetric, n_warmup=1, n_runs=3)

            print(
                f"{n:>6} | {full_vars:>10} | {sym_vars:>10} | {timing.mean_ms:>11.2f}ms"
            )

        print("-" * 70)


class TestLargeScaleLP:
    """Large-scale LP benchmarks up to n=5,000."""

    def test_very_large_lp_vectorvariable(self):
        """Solve very large LP using VectorVariable."""
        sizes = [500, 1000, 2000, 5000]

        print("\n" + "=" * 70)
        print("LARGE-SCALE LP: VectorVariable vs SciPy (n up to 5,000)")
        print("=" * 70)
        print(f"{'n':>8} | {'m':>6} | {'Build':>10} | {'Solve':>10} | {'SciPy':>10}")
        print("-" * 70)

        for n in sizes:
            m = n // 5  # 20% density for constraints
            np.random.seed(42)

            c = np.random.rand(n)
            A = np.random.rand(m, n)
            b = np.sum(A, axis=1) * 0.5
            bounds_list = [(0, 1)] * n

            # Build with VectorVariable
            build_start = time_function(
                lambda: self._build_lp(n, m, c, A, b), n_warmup=0, n_runs=1
            )

            # Build and solve
            x = VectorVariable("x", n, lb=0, ub=1)
            prob = Problem(name=f"large_lp_{n}")
            prob.maximize(c @ x)
            for i in range(m):
                prob.subject_to(A[i] @ x <= b[i])

            # Solve timing (reduce runs for large n)
            prob.solve()  # Warm up
            n_runs = 3 if n <= 2000 else 2
            solve_timing = time_function(
                lambda: prob.solve(), n_warmup=1, n_runs=n_runs
            )

            # SciPy timing
            scipy_timing = time_function(
                lambda: linprog(-c, A_ub=A, b_ub=b, bounds=bounds_list, method="highs"),
                n_warmup=1,
                n_runs=n_runs,
            )

            print(
                f"{n:>8} | {m:>6} | {build_start.mean_ms:>9.1f}ms | "
                f"{solve_timing.mean_ms:>9.1f}ms | {scipy_timing.mean_ms:>9.1f}ms"
            )

        print("-" * 70)

    def _build_lp(self, n, m, c, A, b):
        """Helper to build LP problem."""
        x = VectorVariable("x", n, lb=0, ub=1)
        prob = Problem(name=f"lp_{n}")
        prob.maximize(c @ x)
        for i in range(m):
            prob.subject_to(A[i] @ x <= b[i])
        return prob


class TestPortfolioVectorized:
    """Portfolio optimization with VectorVariable and MatrixParameter."""

    def test_portfolio_scaling(self):
        """Portfolio optimization at various scales."""
        sizes = [10, 50, 100, 200, 500]

        print("\n" + "=" * 70)
        print("PORTFOLIO OPTIMIZATION: VectorVariable Scaling")
        print("=" * 70)
        print(f"{'n assets':>10} | {'Build':>10} | {'Solve':>10} | {'SciPy':>10}")
        print("-" * 70)

        for n in sizes:
            np.random.seed(42)

            # Generate returns and covariance
            returns = np.random.rand(n) * 0.1 + 0.05
            L = np.random.rand(n, n) * 0.1
            cov = L @ L.T + np.eye(n) * 0.01  # Positive definite

            # Build timing
            def build_portfolio():
                w = VectorVariable("w", n, lb=0, ub=1)
                prob = Problem(name="portfolio")
                # Vectorized return: returns @ w
                expected_return = returns @ w
                # Quadratic variance: w · (Σw) = wᵀΣw
                variance = w.dot(cov @ w)
                prob.maximize(expected_return - 0.5 * variance)
                # Vectorized budget constraint
                prob.subject_to(w.sum().eq(1))
                return prob

            build_timing = time_function(build_portfolio, n_warmup=0, n_runs=1)

            # Solve timing
            prob = build_portfolio()
            solve_timing = time_function(lambda: prob.solve(), n_warmup=1, n_runs=3)

            # SciPy timing
            def scipy_portfolio():
                def obj(weights):
                    ret = returns @ weights
                    var = weights @ cov @ weights
                    return -(ret - 0.5 * var)

                def grad(weights):
                    return -(returns - cov @ weights)

                x0 = np.ones(n) / n
                return minimize(
                    obj,
                    x0,
                    jac=grad,
                    method="SLSQP",
                    bounds=[(0, 1)] * n,
                    constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
                )

            scipy_timing = time_function(scipy_portfolio, n_warmup=1, n_runs=3)

            print(
                f"{n:>10} | {build_timing.mean_ms:>9.1f}ms | "
                f"{solve_timing.mean_ms:>9.1f}ms | {scipy_timing.mean_ms:>9.1f}ms"
            )

        print("-" * 70)


if __name__ == "__main__":
    print("=" * 70)
    print("OPTYX VS SCIPY COMPARISON")
    print("=" * 70)

    # Run basic LP tests
    test_lp = TestLPComparison()
    test_lp.test_small_lp()
    test_lp.test_medium_lp()
    test_lp.test_large_lp()

    # VectorVariable scaling
    test_vector = TestVectorVariableScaling()
    test_vector.test_lp_formulation_scaling()
    test_vector.test_lp_solve_comparison()

    # MatrixVariable scaling
    test_matrix = TestMatrixVariableScaling()
    test_matrix.test_assignment_problem_scaling()
    test_matrix.test_symmetric_matrix_scaling()

    # Large-scale benchmarks
    test_large = TestLargeScaleLP()
    test_large.test_very_large_lp_vectorvariable()

    # Portfolio with VectorVariable
    test_portfolio_vec = TestPortfolioVectorized()
    test_portfolio_vec.test_portfolio_scaling()

    # Generate plots
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    test_scale = TestScalingComparison()
    test_scale.test_lp_scaling_plot()

    test_overhead = TestOverheadBreakdown()
    test_overhead.test_overhead_breakdown_plot()

    test_portfolio = TestPortfolioComparison()
    test_portfolio.test_portfolio_vectorized()

    print(f"\nPlots saved to: {RESULTS_DIR}")
