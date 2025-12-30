"""Comparison benchmark: Optyx vs SciPy with vectorized operations.

Direct comparison with raw SciPy for both LP and NLP problems.
Uses numpy vectorization (@, np.sum, np.array) for optimal performance.
Generates plots comparing performance across problem sizes.

Includes comparisons of three Optyx approaches:
1. Loop-based: np.array([Variable(f"x{i}", ...) for i in range(n)])
2. VectorVariable: VectorVariable("x", n, ...)
3. MatrixVariable: For 2D problems
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog, minimize

from optyx import Variable, VectorVariable, MatrixVariable, Problem

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
        """Generate LP scaling comparison plot using VectorVariable.

        Scales up to n=5,000 to demonstrate VectorVariable performance.
        """
        sizes = [100, 500, 1000, 2000, 5000]
        data = ScalingData(label="LP (VectorVariable)")

        print("\n" + "=" * 70)
        print("LP SCALING: Optyx VectorVariable vs SciPy (n up to 5,000)")
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

            # Reduce runs for large problems
            n_runs = 10 if n <= 2000 else 5 if n <= 5000 else 3
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
            title="LP Scaling: Optyx VectorVariable vs SciPy (n up to 5,000)",
            save_path=RESULTS_DIR / "scipy_lp_scaling.png",
        )

    def test_nlp_scaling_plot(self):
        """Generate NLP scaling comparison plot using VectorVariable.

        Scales up to n=250 for NLP problems (limited by expression tree depth).
        """
        sizes = [10, 25, 50, 100, 250]
        data = ScalingData(label="Quadratic NLP (VectorVariable)")

        print("\n" + "=" * 70)
        print("NLP SCALING: Optyx VectorVariable vs SciPy (n up to 250)")
        print("=" * 70)

        for n in sizes:
            x0 = np.zeros(n)

            # Optyx with VectorVariable
            x = VectorVariable("x", n)
            prob = Problem(name=f"nlp_scale_{n}")
            # Quadratic objective: sum(x^2) - sum(x)
            obj = sum(x[i] ** 2 - x[i] for i in range(n))
            prob.minimize(obj)

            n_runs = 15 if n <= 100 else 10
            optyx_timing = time_function(
                lambda p=prob, x0=x0: p.solve(x0=x0), n_warmup=1, n_runs=n_runs
            )

            # SciPy
            def scipy_obj(v):
                return np.sum(v**2) - np.sum(v)

            def scipy_grad(v):
                return 2 * v - 1

            scipy_timing = time_function(
                lambda x0=x0: minimize(scipy_obj, x0, jac=scipy_grad, method="BFGS"),
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
            title="NLP Scaling: Optyx VectorVariable vs SciPy (n up to 250)",
            save_path=RESULTS_DIR / "scipy_nlp_scaling.png",
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
    """Compare Variable loop vs VectorVariable at scale."""

    def test_lp_scaling_comparison(self):
        """Compare LP formulation: loop vs VectorVariable up to n=10,000."""
        sizes = [100, 500, 1000, 2000, 5000, 10000]

        print("\n" + "=" * 70)
        print("LP FORMULATION COMPARISON: Variable Loop vs VectorVariable")
        print("=" * 70)
        print(f"{'n':>8} | {'Loop (ms)':>12} | {'Vector (ms)':>12} | {'Speedup':>8}")
        print("-" * 70)

        for n in sizes:
            m = n // 4  # Constraints = n/4
            np.random.seed(42)

            c = np.random.rand(n)
            A = np.random.rand(m, n)
            b = np.sum(A, axis=1) * 0.5

            # ===== LOOP-BASED APPROACH =====
            def build_loop():
                x = np.array([Variable(f"x{i}", lb=0, ub=1) for i in range(n)])
                prob = Problem(name="lp_loop")
                prob.maximize(c @ x)
                for i in range(m):
                    prob.subject_to(A[i] @ x <= b[i])
                return prob

            loop_timing = time_function(build_loop, n_warmup=1, n_runs=5)

            # ===== VECTORVARIABLE APPROACH =====
            def build_vector():
                x = VectorVariable("x", n, lb=0, ub=1)
                prob = Problem(name="lp_vector")
                prob.maximize(c @ x)
                for i in range(m):
                    prob.subject_to(A[i] @ x <= b[i])
                return prob

            vector_timing = time_function(build_vector, n_warmup=1, n_runs=5)

            speedup = loop_timing.mean_ms / vector_timing.mean_ms

            print(
                f"{n:>8} | {loop_timing.mean_ms:>12.2f} | "
                f"{vector_timing.mean_ms:>12.2f} | {speedup:>7.2f}x"
            )

        print("-" * 70)

    def test_lp_solve_comparison(self):
        """Compare solve performance: loop vs VectorVariable vs SciPy.

        Note: Loop-based approach is limited to smaller sizes due to
        recursion depth limits from deep expression trees.
        """
        sizes = [100, 200, 500]  # Limited due to recursion depth in loop approach

        print("\n" + "=" * 70)
        print("LP SOLVE COMPARISON: Loop vs VectorVariable vs SciPy")
        print("=" * 70)
        print(
            f"{'n':>6} | {'Loop':>10} | {'Vector':>10} | "
            f"{'SciPy':>10} | {'V/S Ratio':>10}"
        )
        print("-" * 70)

        for n in sizes:
            m = n // 4
            np.random.seed(42)

            c = np.random.rand(n)
            A = np.random.rand(m, n)
            b = np.sum(A, axis=1) * 0.5
            bounds_list = [(0, 1)] * n

            # Build problems once
            x_loop = np.array([Variable(f"x{i}", lb=0, ub=1) for i in range(n)])
            prob_loop = Problem(name="lp_loop")
            prob_loop.maximize(c @ x_loop)
            for i in range(m):
                prob_loop.subject_to(A[i] @ x_loop <= b[i])

            x_vec = VectorVariable("x", n, lb=0, ub=1)
            prob_vec = Problem(name="lp_vector")
            prob_vec.maximize(c @ x_vec)
            for i in range(m):
                prob_vec.subject_to(A[i] @ x_vec <= b[i])

            # Warm up
            prob_loop.solve()
            prob_vec.solve()

            # Time solves
            loop_timing = time_function(
                lambda: prob_loop.solve(), n_warmup=2, n_runs=10
            )
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
                f"{n:>6} | {loop_timing.mean_ms:>9.2f}ms | "
                f"{vector_timing.mean_ms:>9.2f}ms | {scipy_timing.mean_ms:>9.2f}ms | "
                f"{ratio:>9.2f}x"
            )

        print("-" * 70)


class TestMatrixVariableScaling:
    """Compare Variable loop vs MatrixVariable for matrix problems."""

    def test_assignment_problem_scaling(self):
        """Assignment problem: compare loop vs MatrixVariable."""
        sizes = [10, 25, 50, 100, 200]

        print("\n" + "=" * 70)
        print("ASSIGNMENT PROBLEM: Variable Loop vs MatrixVariable")
        print("=" * 70)
        print(f"{'n':>6} | {'Loop Build':>12} | {'Matrix Build':>12} | {'Speedup':>8}")
        print("-" * 70)

        for n in sizes:
            np.random.seed(42)
            cost = np.random.rand(n, n)

            # ===== LOOP-BASED APPROACH =====
            def build_loop():
                x = np.array(
                    [
                        [Variable(f"x_{i}_{j}", lb=0, ub=1) for j in range(n)]
                        for i in range(n)
                    ]
                )
                prob = Problem(name="assign_loop")
                prob.minimize(np.sum(cost * x))
                # Row constraints
                for i in range(n):
                    prob.subject_to(np.sum(x[i, :]) == 1)
                # Column constraints
                for j in range(n):
                    prob.subject_to(np.sum(x[:, j]) == 1)
                return prob

            loop_timing = time_function(build_loop, n_warmup=1, n_runs=3)

            # ===== MATRIXVARIABLE APPROACH =====
            def build_matrix():
                X = MatrixVariable("X", n, n, lb=0, ub=1)
                prob = Problem(name="assign_matrix")
                prob.minimize(
                    sum(cost[i, j] * X[i, j] for i in range(n) for j in range(n))
                )
                # Row constraints
                for i in range(n):
                    prob.subject_to(sum(X[i, j] for j in range(n)) == 1)
                # Column constraints
                for j in range(n):
                    prob.subject_to(sum(X[i, j] for i in range(n)) == 1)
                return prob

            matrix_timing = time_function(build_matrix, n_warmup=1, n_runs=3)

            speedup = loop_timing.mean_ms / matrix_timing.mean_ms

            print(
                f"{n:>6} | {loop_timing.mean_ms:>11.2f}ms | "
                f"{matrix_timing.mean_ms:>11.2f}ms | {speedup:>7.2f}x"
            )

        print("-" * 70)

    def test_symmetric_matrix_scaling(self):
        """Compare symmetric MatrixVariable vs full matrix loop."""
        sizes = [10, 25, 50, 100]

        print("\n" + "=" * 70)
        print("SYMMETRIC MATRIX: Full Loop vs Symmetric MatrixVariable")
        print("=" * 70)
        print(
            f"{'n':>6} | {'Full Vars':>10} | {'Sym Vars':>10} | "
            f"{'Loop (ms)':>10} | {'Sym (ms)':>10}"
        )
        print("-" * 70)

        for n in sizes:
            np.random.seed(42)

            # Full matrix has n^2 variables
            full_vars = n * n
            # Symmetric has n*(n+1)/2 variables
            sym_vars = n * (n + 1) // 2

            # ===== LOOP-BASED (FULL MATRIX) =====
            def build_full():
                X = np.array(
                    [
                        [Variable(f"x_{i}_{j}", lb=-1, ub=1) for j in range(n)]
                        for i in range(n)
                    ]
                )
                prob = Problem(name="full_matrix")
                prob.minimize(np.sum(X**2))
                return prob

            full_timing = time_function(build_full, n_warmup=1, n_runs=3)

            # ===== SYMMETRIC MATRIXVARIABLE =====
            def build_symmetric():
                X = MatrixVariable("X", n, n, lb=-1, ub=1, symmetric=True)
                prob = Problem(name="sym_matrix")
                prob.minimize(sum(X[i, j] ** 2 for i in range(n) for j in range(n)))
                return prob

            sym_timing = time_function(build_symmetric, n_warmup=1, n_runs=3)

            print(
                f"{n:>6} | {full_vars:>10} | {sym_vars:>10} | "
                f"{full_timing.mean_ms:>9.2f}ms | {sym_timing.mean_ms:>9.2f}ms"
            )

        print("-" * 70)


class TestLargeScaleLP:
    """Large-scale LP benchmarks up to n=10,000."""

    def test_very_large_lp_vectorvariable(self):
        """Solve very large LP using VectorVariable."""
        sizes = [1000, 2000, 5000, 10000]

        print("\n" + "=" * 70)
        print("LARGE-SCALE LP: VectorVariable vs SciPy (n up to 10,000)")
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

            # Solve timing
            prob.solve()  # Warm up
            solve_timing = time_function(lambda: prob.solve(), n_warmup=1, n_runs=5)

            # SciPy timing
            scipy_timing = time_function(
                lambda: linprog(-c, A_ub=A, b_ub=b, bounds=bounds_list, method="highs"),
                n_warmup=1,
                n_runs=5,
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
                # Linear return
                expected_return = sum(returns[i] * w[i] for i in range(n))
                # Quadratic variance (simplified for large n)
                variance = sum(
                    cov[i, j] * w[i] * w[j] for i in range(n) for j in range(n)
                )
                prob.maximize(expected_return - 0.5 * variance)
                prob.subject_to(sum(w[i] for i in range(n)) == 1)
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
    print("OPTYX VS SCIPY COMPARISON (VECTORIZED)")
    print("=" * 70)

    # Run basic tests
    test_lp = TestLPComparison()
    test_lp.test_small_lp()
    test_lp.test_medium_lp()
    test_lp.test_large_lp()

    test_nlp = TestNLPComparison()
    test_nlp.test_rosenbrock()
    test_nlp.test_constrained_qp()

    # VectorVariable scaling comparisons
    test_vector = TestVectorVariableScaling()
    test_vector.test_lp_scaling_comparison()
    test_vector.test_lp_solve_comparison()

    # MatrixVariable comparisons
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
    test_scale.test_nlp_scaling_plot()

    test_overhead = TestOverheadBreakdown()
    test_overhead.test_overhead_breakdown_plot()

    test_portfolio = TestPortfolioComparison()
    test_portfolio.test_portfolio_vectorized()

    print(f"\nPlots saved to: {RESULTS_DIR}")
