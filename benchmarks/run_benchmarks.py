#!/usr/bin/env python
"""Run all benchmarks and generate performance plots.

Usage:
    uv run python benchmarks/run_benchmarks.py

Generates plots in benchmarks/results/:
    - lp_scaling_comparison.png: LP scaling Optyx vs SciPy
    - nlp_quadratic_scaling.png: NLP scaling Optyx vs SciPy
    - lp_cache_benefit.png: Cache benefit analysis
    - multi_problem_scaling.png: Scaling by problem type
    - scipy_lp_scaling.png: Detailed LP comparison
    - scipy_nlp_scaling.png: Detailed NLP comparison
    - overhead_breakdown.png: Overhead by problem type
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add benchmarks to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.optimize import linprog, minimize

from optyx import Variable, Problem
from utils import (
    time_function,
    ScalingData,
    RESULTS_DIR,
    plot_scaling_comparison,
    plot_cache_benefit,
    plot_overhead_breakdown,
    plot_multi_scaling,
)


def run_lp_vs_scipy_scaling():
    """Generate LP scaling comparison plot."""
    print("\n" + "=" * 60)
    print("LP SCALING: Optyx vs SciPy")
    print("=" * 60)

    sizes = [10, 25, 50, 100, 200, 500]
    data = ScalingData(label="LP")

    for n in sizes:
        m = n // 2
        np.random.seed(42)

        c = np.random.rand(n)
        A = np.random.rand(m, n)
        b = np.sum(A, axis=1) * 0.5

        # Optyx (vectorized)
        x = np.array([Variable(f"x{i}", lb=0, ub=1) for i in range(n)])
        prob = Problem(name=f"lp_bench_{n}")
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

        ratio = optyx_timing.mean_ms / scipy_timing.mean_ms
        print(
            f"  n={n:4d}: Optyx={optyx_timing.mean_ms:7.3f}ms, "
            f"SciPy={scipy_timing.mean_ms:7.3f}ms, ratio={ratio:.2f}x"
        )

    plot_scaling_comparison(
        data,
        title="LP Scaling: Optyx vs SciPy (Vectorized)",
        save_path=RESULTS_DIR / "lp_scaling_comparison.png",
    )

    return data


def run_nlp_vs_scipy_scaling():
    """Generate NLP scaling comparison plot."""
    print("\n" + "=" * 60)
    print("NLP SCALING: Optyx vs SciPy")
    print("=" * 60)

    sizes = [10, 25, 50, 100]
    data = ScalingData(label="Quadratic NLP")

    for n in sizes:
        x0 = np.zeros(n)

        # Optyx (vectorized)
        x = np.array([Variable(f"x{i}") for i in range(n)])
        prob = Problem(name=f"nlp_bench_{n}")
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
            f"  n={n:4d}: Optyx={optyx_timing.mean_ms:7.3f}ms, "
            f"SciPy={scipy_timing.mean_ms:7.3f}ms, ratio={ratio:.2f}x"
        )

    plot_scaling_comparison(
        data,
        title="Quadratic NLP Scaling: Optyx vs SciPy",
        save_path=RESULTS_DIR / "nlp_quadratic_scaling.png",
    )

    return data


def run_cache_benefit_analysis():
    """Generate cache benefit plot."""
    print("\n" + "=" * 60)
    print("CACHE BENEFIT ANALYSIS")
    print("=" * 60)

    sizes = [10, 25, 50, 100, 200]
    cold_times = []
    warm_times = []

    import time

    for n in sizes:
        m = n // 2
        np.random.seed(42)

        c = np.random.rand(n)
        A = np.random.rand(m, n)
        b = np.sum(A, axis=1) * 0.5

        x = np.array([Variable(f"x{i}", lb=0, ub=1) for i in range(n)])
        prob = Problem(name=f"cache_bench_{n}")
        prob.maximize(c @ x)
        for i in range(m):
            prob.subject_to(A[i] @ x <= b[i])

        # Cold solve
        start = time.perf_counter()
        prob.solve()
        cold_ms = (time.perf_counter() - start) * 1000
        cold_times.append(cold_ms)

        # Warm solves
        warm_timing = time_function(lambda: prob.solve(), n_warmup=0, n_runs=10)
        warm_times.append(warm_timing.mean_ms)

        speedup = cold_ms / warm_timing.mean_ms
        print(
            f"  n={n:4d}: Cold={cold_ms:.3f}ms, Warm={warm_timing.mean_ms:.3f}ms, "
            f"Speedup={speedup:.2f}x"
        )

    plot_cache_benefit(
        sizes,
        cold_times,
        warm_times,
        title="LP Cache Benefit Analysis",
        save_path=RESULTS_DIR / "lp_cache_benefit.png",
    )


def run_overhead_breakdown():
    """Generate overhead breakdown by problem type."""
    print("\n" + "=" * 60)
    print("OVERHEAD BREAKDOWN BY PROBLEM TYPE")
    print("=" * 60)

    categories = []
    overheads = []

    # Small LP
    c = np.array([20.0, 30.0])
    A = np.array([[4.0, 6.0], [2.0, 3.0]])
    b = np.array([120.0, 60.0])
    x = np.array([Variable("lp_x", lb=0), Variable("lp_y", lb=0)])
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
    print(f"  Small LP: {overheads[-1]:.2f}x")

    # Medium LP
    n, m = 50, 25
    np.random.seed(42)
    c = np.random.rand(n)
    A_mat = np.random.rand(m, n)
    b_vec = np.sum(A_mat, axis=1) * 0.5
    x = np.array([Variable(f"mlp_x{i}", lb=0, ub=1) for i in range(n)])
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
    print(f"  Medium LP: {overheads[-1]:.2f}x")

    # Large LP
    n, m = 200, 100
    np.random.seed(42)
    c = np.random.rand(n)
    A_mat = np.random.rand(m, n)
    b_vec = np.sum(A_mat, axis=1) * 0.5
    x = np.array([Variable(f"llp_x{i}", lb=0, ub=1) for i in range(n)])
    prob = Problem(name="overhead_large_lp")
    prob.maximize(c @ x)
    for i in range(m):
        prob.subject_to(A_mat[i] @ x <= b_vec[i])

    prob.solve()
    optyx_t = time_function(lambda: prob.solve(), n_warmup=2, n_runs=20)
    bounds = [(0, 1)] * n
    scipy_t = time_function(
        lambda: linprog(-c, A_ub=A_mat, b_ub=b_vec, bounds=bounds, method="highs"),
        n_warmup=2,
        n_runs=20,
    )
    categories.append("Large LP (n=200)")
    overheads.append(optyx_t.mean_ms / scipy_t.mean_ms)
    print(f"  Large LP: {overheads[-1]:.2f}x")

    # Rosenbrock
    rx = Variable("ros_x")
    ry = Variable("ros_y")
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
    categories.append("Rosenbrock NLP")
    overheads.append(optyx_t.mean_ms / scipy_t.mean_ms)
    print(f"  Rosenbrock: {overheads[-1]:.2f}x")

    # Constrained QP
    qx = np.array([Variable("cqp_x"), Variable("cqp_y")])
    prob = Problem(name="overhead_cqp")
    prob.minimize(np.sum(qx**2))
    prob.subject_to(np.sum(qx) >= 1)

    optyx_t = time_function(lambda: prob.solve(x0=np.zeros(2)), n_warmup=2, n_runs=30)

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
            qp_obj, np.zeros(2), jac=qp_grad, method="SLSQP", constraints=constraints
        ),
        n_warmup=2,
        n_runs=30,
    )
    categories.append("Constrained QP")
    overheads.append(optyx_t.mean_ms / scipy_t.mean_ms)
    print(f"  Constrained QP: {overheads[-1]:.2f}x")

    plot_overhead_breakdown(
        categories,
        overheads,
        title="Optyx Overhead vs SciPy by Problem Type",
        save_path=RESULTS_DIR / "overhead_breakdown.png",
    )


def run_multi_problem_scaling():
    """Generate combined scaling plot for all problem types."""
    print("\n" + "=" * 60)
    print("MULTI-PROBLEM SCALING COMPARISON")
    print("=" * 60)

    sizes = [10, 25, 50, 100]

    # LP data
    lp_data = ScalingData(label="LP (cached)")
    for n in sizes:
        m = n // 2
        np.random.seed(42)
        c = np.random.rand(n)
        A = np.random.rand(m, n)
        b = np.sum(A, axis=1) * 0.5

        x = np.array([Variable(f"mslp_x{i}", lb=0, ub=1) for i in range(n)])
        prob = Problem(name=f"ms_lp_{n}")
        prob.maximize(c @ x)
        for i in range(m):
            prob.subject_to(A[i] @ x <= b[i])

        prob.solve()  # Warm
        timing = time_function(lambda: prob.solve(), n_warmup=0, n_runs=10)
        lp_data.add_point(n, timing.mean_ms, timing.std_ms)

    print("  LP scaling collected")

    # Quadratic NLP data
    quad_data = ScalingData(label="Quadratic NLP")
    for n in sizes:
        x = np.array([Variable(f"msq_x{i}") for i in range(n)])
        prob = Problem(name=f"ms_quad_{n}")
        prob.minimize(np.sum(x**2) - np.sum(x))

        x0 = np.zeros(n)
        timing = time_function(lambda: prob.solve(x0=x0), n_warmup=2, n_runs=5)
        quad_data.add_point(n, timing.mean_ms, timing.std_ms)

    print("  Quadratic NLP scaling collected")

    # Constrained QP data
    cqp_data = ScalingData(label="Constrained QP")
    for n in sizes:
        x = np.array([Variable(f"mscqp_x{i}", lb=0) for i in range(n)])
        prob = Problem(name=f"ms_cqp_{n}")
        prob.minimize(np.sum(x**2))
        prob.subject_to(np.sum(x) >= 1)

        x0 = np.full(n, 0.1)
        timing = time_function(lambda: prob.solve(x0=x0), n_warmup=2, n_runs=5)
        cqp_data.add_point(n, timing.mean_ms, timing.std_ms)

    print("  Constrained QP scaling collected")

    plot_multi_scaling(
        [lp_data, quad_data, cqp_data],
        title="Optyx Scaling by Problem Type",
        save_path=RESULTS_DIR / "multi_problem_scaling.png",
    )


def main():
    """Run all benchmarks and generate plots."""
    print("=" * 60)
    print("OPTYX BENCHMARK SUITE")
    print("=" * 60)
    print(f"Results will be saved to: {RESULTS_DIR}")

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run all benchmarks
    run_lp_vs_scipy_scaling()
    run_nlp_vs_scipy_scaling()
    run_cache_benefit_analysis()
    run_overhead_breakdown()
    run_multi_problem_scaling()

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Plots saved to: {RESULTS_DIR}")
    print("Files generated:")
    for f in sorted(RESULTS_DIR.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
