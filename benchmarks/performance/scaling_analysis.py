"""Performance benchmark: Scaling analysis with vectorized operations.

Measures how Optyx performance scales with problem size using numpy vectorization.
Generates plots comparing Optyx vs SciPy across different problem sizes.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import linprog, minimize

from optyx import Variable, Problem

import sys

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from utils import (
    time_function,
    time_solve,
    ScalingData,
    RESULTS_DIR,
    plot_scaling_comparison,
    plot_cache_benefit,
    plot_multi_scaling,
)


# Problem sizes to test
LP_SIZES = [10, 25, 50, 100, 200, 500]
NLP_SIZES = [10, 25, 50, 100]


class TestLPScaling:
    """LP scaling analysis using vectorized numpy operations."""

    @pytest.mark.parametrize("n", LP_SIZES)
    def test_lp_scaling(self, n: int):
        """LP scaling with n variables and n/2 constraints.

        Uses numpy array variables with @ for matrix operations.
        """
        m = n // 2
        np.random.seed(42)

        c = np.random.rand(n)
        A = np.random.rand(m, n)
        b = np.sum(A, axis=1) * 0.5

        # Optyx with numpy vectorization
        x = np.array([Variable(f"x{i}", lb=0, ub=1) for i in range(n)])

        prob = Problem(name=f"lp_{n}")
        prob.maximize(c @ x)  # Vectorized dot product
        for i in range(m):
            prob.subject_to(A[i] @ x <= b[i])  # Matrix-vector product

        timing, sol = time_solve(prob, n_warmup=2, n_runs=10)

        print(f"\nLP n={n}: {timing}")
        print(f"  Status: {sol.status.value}")
        print(f"  Objective: {sol.objective_value:.4f}")

        assert sol.is_optimal

    @pytest.mark.parametrize("n", LP_SIZES)
    def test_lp_scaling_cached(self, n: int):
        """LP scaling with caching benefit (repeated solve)."""
        m = n // 2
        np.random.seed(42)

        c = np.random.rand(n)
        A = np.random.rand(m, n)
        b = np.sum(A, axis=1) * 0.5

        x = np.array([Variable(f"x{i}", lb=0, ub=1) for i in range(n)])

        prob = Problem(name=f"lp_{n}_cached")
        prob.maximize(c @ x)
        for i in range(m):
            prob.subject_to(A[i] @ x <= b[i])

        # First solve (cache miss)
        timing_cold, _ = time_solve(prob, n_warmup=0, n_runs=1)

        # Subsequent solves (cache hit)
        timing_warm, sol = time_solve(prob, n_warmup=0, n_runs=10)

        speedup = (
            timing_cold.mean_ms / timing_warm.mean_ms
            if timing_warm.mean_ms > 0
            else float("inf")
        )

        print(f"\nLP n={n} caching:")
        print(f"  Cold: {timing_cold.mean_ms:.3f} ms")
        print(f"  Warm: {timing_warm.mean_ms:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")

        assert sol.is_optimal


class TestLPVsSciPyScaling:
    """Compare LP scaling between Optyx and SciPy."""

    def test_lp_vs_scipy_scaling(self):
        """Compare Optyx LP vs SciPy linprog across problem sizes."""
        data = ScalingData(label="LP")

        for n in LP_SIZES:
            m = n // 2
            np.random.seed(42)

            c = np.random.rand(n)
            A = np.random.rand(m, n)
            b = np.sum(A, axis=1) * 0.5

            # Optyx (vectorized)
            x = np.array([Variable(f"x{i}", lb=0, ub=1) for i in range(n)])
            prob = Problem(name=f"lp_compare_{n}")
            prob.maximize(c @ x)
            for i in range(m):
                prob.subject_to(A[i] @ x <= b[i])

            # Warm up and time
            prob.solve()  # Warm cache
            optyx_timing = time_function(lambda: prob.solve(), n_warmup=2, n_runs=20)

            # SciPy
            bounds = [(0, 1)] * n

            def scipy_solve():
                return linprog(-c, A_ub=A, b_ub=b, bounds=bounds, method="highs")

            scipy_timing = time_function(scipy_solve, n_warmup=2, n_runs=20)

            data.add_point(
                n,
                optyx_timing.mean_ms,
                optyx_timing.std_ms,
                scipy_timing.mean_ms,
                scipy_timing.std_ms,
            )

            ratio = optyx_timing.mean_ms / scipy_timing.mean_ms
            print(
                f"LP n={n:4d}: Optyx={optyx_timing.mean_ms:7.3f}ms, "
                f"SciPy={scipy_timing.mean_ms:7.3f}ms, ratio={ratio:.2f}x"
            )

        # Generate plot
        plot_scaling_comparison(
            data,
            title="LP Scaling: Optyx vs SciPy",
            save_path=RESULTS_DIR / "lp_scaling_comparison.png",
        )

        # Overhead should converge for larger problems
        final_ratio = data.overhead_ratios()[-1]
        print(f"\nFinal overhead ratio (n={LP_SIZES[-1]}): {final_ratio:.2f}x")


class TestNLPScaling:
    """NLP scaling analysis using vectorized operations."""

    @pytest.mark.parametrize("n", NLP_SIZES)
    def test_quadratic_scaling(self, n: int):
        """Quadratic problem scaling with vectorized objective.

        min ||x||² - sum(x)
        Optimal: x_i = 0.5 for all i
        """
        x = np.array([Variable(f"x{i}") for i in range(n)])

        prob = Problem(name=f"quadratic_{n}")
        # Vectorized: np.sum(x**2) - np.sum(x)
        prob.minimize(np.sum(x**2) - np.sum(x))

        timing, sol = time_solve(prob, x0=np.zeros(n), n_warmup=2, n_runs=5)

        print(f"\nQuadratic n={n}: {timing}")
        print(f"  Status: {sol.status.value}")
        print(f"  Objective: {sol.objective_value:.4f}")

        assert sol.is_optimal
        expected = -n / 4
        assert abs(sol.objective_value - expected) < 0.1

    @pytest.mark.parametrize("n", [10, 25, 50])
    def test_rosenbrock_scaling(self, n: int):
        """Rosenbrock chain scaling with vectorized operations."""
        x = np.array([Variable(f"x{i}") for i in range(n)])

        prob = Problem(name=f"rosenbrock_{n}")
        # Vectorized Rosenbrock
        objective = np.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)
        prob.minimize(objective)

        timing, sol = time_solve(prob, x0=np.zeros(n), n_warmup=1, n_runs=3)

        print(f"\nRosenbrock n={n}: {timing}")
        print(f"  Status: {sol.status.value}")
        print(f"  Objective: {sol.objective_value:.6f}")

        assert sol.objective_value < 1e-3


class TestNLPVsSciPyScaling:
    """Compare NLP scaling between Optyx and SciPy."""

    def test_quadratic_vs_scipy_scaling(self):
        """Compare quadratic optimization across problem sizes."""
        data = ScalingData(label="Quadratic NLP")

        for n in NLP_SIZES:
            x0 = np.zeros(n)

            # Optyx (vectorized)
            x = np.array([Variable(f"x{i}") for i in range(n)])
            prob = Problem(name=f"quad_compare_{n}")
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

            ratio = optyx_timing.mean_ms / scipy_timing.mean_ms
            print(
                f"Quadratic n={n:4d}: Optyx={optyx_timing.mean_ms:7.3f}ms, "
                f"SciPy={scipy_timing.mean_ms:7.3f}ms, ratio={ratio:.2f}x"
            )

        # Generate plot
        plot_scaling_comparison(
            data,
            title="Quadratic NLP Scaling: Optyx vs SciPy",
            save_path=RESULTS_DIR / "nlp_quadratic_scaling.png",
        )


class TestConstrainedScaling:
    """Constrained problem scaling with vectorized operations."""

    @pytest.mark.parametrize("n", NLP_SIZES)
    def test_constrained_quadratic_scaling(self, n: int):
        """Constrained quadratic scaling with vectorized objective.

        min sum(x²)
        s.t. sum(x) >= 1
             x >= 0
        """
        x = np.array([Variable(f"x{i}", lb=0) for i in range(n)])

        prob = Problem(name=f"constrained_qp_{n}")
        prob.minimize(np.sum(x**2))
        prob.subject_to(np.sum(x) >= 1)

        timing, sol = time_solve(prob, x0=np.full(n, 0.1), n_warmup=2, n_runs=5)

        print(f"\nConstrained QP n={n}: {timing}")
        print(f"  Status: {sol.status.value}")
        print(f"  Objective: {sol.objective_value:.6f}")

        assert sol.is_optimal
        expected = 1.0 / n
        assert abs(sol.objective_value - expected) < 0.01


class TestCacheBenefitAnalysis:
    """Analyze cache benefit across problem sizes."""

    def test_lp_cache_benefit_plot(self):
        """Generate cache benefit plot for LP problems."""
        sizes = [10, 25, 50, 100, 200]
        cold_times = []
        warm_times = []

        for n in sizes:
            m = n // 2
            np.random.seed(42)

            c = np.random.rand(n)
            A = np.random.rand(m, n)
            b = np.sum(A, axis=1) * 0.5

            x = np.array([Variable(f"x{i}", lb=0, ub=1) for i in range(n)])
            prob = Problem(name=f"cache_lp_{n}")
            prob.maximize(c @ x)
            for i in range(m):
                prob.subject_to(A[i] @ x <= b[i])

            # Cold solve
            import time

            start = time.perf_counter()
            prob.solve()
            cold_ms = (time.perf_counter() - start) * 1000
            cold_times.append(cold_ms)

            # Warm solves
            warm_timing = time_function(lambda: prob.solve(), n_warmup=0, n_runs=10)
            warm_times.append(warm_timing.mean_ms)

            speedup = cold_ms / warm_timing.mean_ms
            print(
                f"n={n:4d}: Cold={cold_ms:.3f}ms, Warm={warm_timing.mean_ms:.3f}ms, "
                f"Speedup={speedup:.2f}x"
            )

        # Generate plot
        plot_cache_benefit(
            sizes,
            cold_times,
            warm_times,
            title="LP Cache Benefit Analysis",
            save_path=RESULTS_DIR / "lp_cache_benefit.png",
        )


class TestMultiProblemScaling:
    """Compare scaling across different problem types."""

    def test_all_problem_types_scaling(self):
        """Generate combined scaling plot for all problem types."""
        sizes = [10, 25, 50, 100]

        # LP data
        lp_data = ScalingData(label="LP (cached)")
        for n in sizes:
            m = n // 2
            np.random.seed(42)
            c = np.random.rand(n)
            A = np.random.rand(m, n)
            b = np.sum(A, axis=1) * 0.5

            x = np.array([Variable(f"x{i}", lb=0, ub=1) for i in range(n)])
            prob = Problem(name=f"multi_lp_{n}")
            prob.maximize(c @ x)
            for i in range(m):
                prob.subject_to(A[i] @ x <= b[i])

            prob.solve()  # Warm
            timing = time_function(lambda: prob.solve(), n_warmup=0, n_runs=10)
            lp_data.add_point(n, timing.mean_ms, timing.std_ms)

        # Quadratic NLP data
        quad_data = ScalingData(label="Quadratic NLP")
        for n in sizes:
            x = np.array([Variable(f"x{i}") for i in range(n)])
            prob = Problem(name=f"multi_quad_{n}")
            prob.minimize(np.sum(x**2) - np.sum(x))

            x0 = np.zeros(n)
            timing = time_function(lambda: prob.solve(x0=x0), n_warmup=2, n_runs=5)
            quad_data.add_point(n, timing.mean_ms, timing.std_ms)

        # Constrained QP data
        cqp_data = ScalingData(label="Constrained QP")
        for n in sizes:
            x = np.array([Variable(f"x{i}", lb=0) for i in range(n)])
            prob = Problem(name=f"multi_cqp_{n}")
            prob.minimize(np.sum(x**2))
            prob.subject_to(np.sum(x) >= 1)

            x0 = np.full(n, 0.1)
            timing = time_function(lambda: prob.solve(x0=x0), n_warmup=2, n_runs=5)
            cqp_data.add_point(n, timing.mean_ms, timing.std_ms)

        # Generate combined plot
        plot_multi_scaling(
            [lp_data, quad_data, cqp_data],
            title="Optyx Scaling by Problem Type",
            save_path=RESULTS_DIR / "multi_problem_scaling.png",
        )


if __name__ == "__main__":
    print("=" * 60)
    print("SCALING ANALYSIS WITH PLOTS")
    print("=" * 60)

    # Run comparison tests
    test_lp = TestLPVsSciPyScaling()
    test_lp.test_lp_vs_scipy_scaling()

    test_nlp = TestNLPVsSciPyScaling()
    test_nlp.test_quadratic_vs_scipy_scaling()

    test_cache = TestCacheBenefitAnalysis()
    test_cache.test_lp_cache_benefit_plot()

    test_multi = TestMultiProblemScaling()
    test_multi.test_all_problem_types_scaling()

    print(f"\nPlots saved to: {RESULTS_DIR}")
