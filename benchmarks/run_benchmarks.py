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

from optyx import Variable, VectorVariable, MatrixVariable, Problem
from utils import (
    time_function,
    ScalingData,
    ThreeSeriesData,
    RESULTS_DIR,
    plot_overhead_breakdown,
    plot_multi_scaling,
    plot_three_series_scaling,
)


def run_lp_vs_scipy_scaling():
    """Generate LP scaling comparison plot with Loop, VectorVariable, and SciPy."""
    print("\n" + "=" * 70)
    print("LP SCALING: Loop vs VectorVariable vs SciPy")
    print("=" * 70)

    # Loop-based sizes (up to 500 to avoid deep recursion)
    loop_sizes = [10, 25, 50, 100, 200, 500]
    # VectorVariable and SciPy extend to 5,000 (10,000 takes too long)
    extended_sizes = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]

    data = ThreeSeriesData(label="LP")

    # First: run loop-based benchmarks (up to n=500)
    print("\n--- Loop-based Variable (n up to 500) ---")
    for n in loop_sizes:
        m = n // 2
        np.random.seed(42)

        c = np.random.rand(n)
        A = np.random.rand(m, n)
        b = np.sum(A, axis=1) * 0.5

        # Optyx with loop-based variables
        x = np.array([Variable(f"x{i}", lb=0, ub=1) for i in range(n)])
        prob = Problem(name=f"lp_loop_{n}")
        prob.maximize(c @ x)
        for i in range(m):
            prob.subject_to(A[i] @ x <= b[i])

        prob.solve()  # Warm cache
        n_runs = max(5, 30 - n // 20)
        loop_timing = time_function(lambda: prob.solve(), n_warmup=2, n_runs=n_runs)

        data.add_loop_point(n, loop_timing.mean_ms, loop_timing.std_ms)
        print(f"  n={n:4d}: Loop={loop_timing.mean_ms:8.3f}ms")

    # Second: run VectorVariable and SciPy benchmarks (up to n=5,000)
    print("\n--- VectorVariable and SciPy (n up to 5,000) ---")
    for n in extended_sizes:
        m = n // 2
        np.random.seed(42)

        c = np.random.rand(n)
        A = np.random.rand(m, n)
        b = np.sum(A, axis=1) * 0.5

        # Optyx with VectorVariable
        x = VectorVariable("xv", n, lb=0, ub=1)
        prob = Problem(name=f"lp_vec_{n}")
        prob.maximize(c @ x)
        for i in range(m):
            prob.subject_to(A[i] @ x <= b[i])

        prob.solve()  # Warm cache
        n_runs = max(3, 20 - n // 500)
        vec_timing = time_function(lambda: prob.solve(), n_warmup=2, n_runs=n_runs)

        # SciPy
        bounds = [(0, 1)] * n
        scipy_timing = time_function(
            lambda: linprog(-c, A_ub=A, b_ub=b, bounds=bounds, method="highs"),
            n_warmup=2,
            n_runs=n_runs,
        )

        data.add_vec_point(n, vec_timing.mean_ms, vec_timing.std_ms)
        data.add_scipy_point(n, scipy_timing.mean_ms, scipy_timing.std_ms)

        ratio = vec_timing.mean_ms / scipy_timing.mean_ms
        print(
            f"  n={n:5d}: VectorVar={vec_timing.mean_ms:8.2f}ms, "
            f"SciPy={scipy_timing.mean_ms:8.2f}ms, ratio={ratio:.2f}x"
        )

    plot_three_series_scaling(
        data,
        title="LP Scaling: Loop vs VectorVariable vs SciPy",
        save_path=RESULTS_DIR / "lp_scaling_comparison.png",
    )

    return data


def run_nlp_vs_scipy_scaling():
    """Generate NLP scaling comparison plot with Loop, VectorVariable, and SciPy."""
    print("\n" + "=" * 70)
    print("NLP SCALING: Loop vs VectorVariable vs SciPy")
    print("=" * 70)

    # NLP sizes are limited due to recursive gradient computation depth
    # Both loop and VectorVariable approaches create nested expression trees
    # Maximum safe n is around 200 due to Python's recursion limit
    sizes = [10, 25, 50, 100, 150, 200]

    data = ThreeSeriesData(label="Quadratic NLP")

    # Run all three series at same sizes for fair comparison
    print("\n--- Loop, VectorVariable, and SciPy (n up to 200) ---")
    for n in sizes:
        x0 = np.zeros(n)

        # Optyx with loop-based variables
        x_loop = np.array([Variable(f"x{i}") for i in range(n)])
        prob_loop = Problem(name=f"nlp_loop_{n}")
        prob_loop.minimize(np.sum(x_loop**2) - np.sum(x_loop))

        loop_timing = time_function(
            lambda p=prob_loop, x0=x0: p.solve(x0=x0), n_warmup=2, n_runs=10
        )
        data.add_loop_point(n, loop_timing.mean_ms, loop_timing.std_ms)

        # Optyx with VectorVariable - build sum incrementally
        x = VectorVariable("xv", n)
        obj = x[0] ** 2 - x[0]
        for i in range(1, n):
            obj = obj + x[i] ** 2 - x[i]

        prob = Problem(name=f"nlp_vec_{n}")
        prob.minimize(obj)

        vec_timing = time_function(
            lambda p=prob, x0=x0: p.solve(x0=x0), n_warmup=2, n_runs=10
        )
        data.add_vec_point(n, vec_timing.mean_ms, vec_timing.std_ms)

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
        data.add_scipy_point(n, scipy_timing.mean_ms, scipy_timing.std_ms)

        ratio = vec_timing.mean_ms / scipy_timing.mean_ms
        print(
            f"  n={n:4d}: Loop={loop_timing.mean_ms:7.2f}ms, "
            f"VectorVar={vec_timing.mean_ms:7.2f}ms, "
            f"SciPy={scipy_timing.mean_ms:7.2f}ms, ratio={ratio:.2f}x"
        )

    plot_three_series_scaling(
        data,
        title="Quadratic NLP: Loop vs VectorVariable vs SciPy",
        save_path=RESULTS_DIR / "nlp_quadratic_scaling.png",
    )

    return data


def run_cache_benefit_analysis():
    """Generate cache benefit plot comparing Loop vs VectorVariable."""
    print("\n" + "=" * 70)
    print("CACHE BENEFIT ANALYSIS: Loop vs VectorVariable")
    print("=" * 70)

    # Loop-based sizes (up to 500)
    loop_sizes = [10, 25, 50, 100, 200, 500]
    # VectorVariable extends to 5,000
    vec_sizes = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]

    loop_cold_times = []
    loop_warm_times = []
    vec_cold_times = []
    vec_warm_times = []

    import time

    print("\n--- Loop-based Variable (n up to 500) ---")
    for n in loop_sizes:
        m = n // 2
        np.random.seed(42)

        c = np.random.rand(n)
        A = np.random.rand(m, n)
        b = np.sum(A, axis=1) * 0.5

        x = np.array([Variable(f"x{i}", lb=0, ub=1) for i in range(n)])
        prob = Problem(name=f"cache_loop_{n}")
        prob.maximize(c @ x)
        for i in range(m):
            prob.subject_to(A[i] @ x <= b[i])

        # Cold solve
        start = time.perf_counter()
        prob.solve()
        cold_ms = (time.perf_counter() - start) * 1000
        loop_cold_times.append(cold_ms)

        # Warm solves
        warm_timing = time_function(lambda: prob.solve(), n_warmup=0, n_runs=10)
        loop_warm_times.append(warm_timing.mean_ms)

        speedup = cold_ms / warm_timing.mean_ms
        print(
            f"  n={n:4d}: Cold={cold_ms:8.3f}ms, Warm={warm_timing.mean_ms:8.3f}ms, "
            f"Speedup={speedup:.2f}x"
        )

    print("\n--- VectorVariable (n up to 5,000) ---")
    for n in vec_sizes:
        m = n // 2
        np.random.seed(42)

        c = np.random.rand(n)
        A = np.random.rand(m, n)
        b = np.sum(A, axis=1) * 0.5

        x = VectorVariable("xv", n, lb=0, ub=1)
        prob = Problem(name=f"cache_vec_{n}")
        prob.maximize(c @ x)
        for i in range(m):
            prob.subject_to(A[i] @ x <= b[i])

        # Cold solve
        start = time.perf_counter()
        prob.solve()
        cold_ms = (time.perf_counter() - start) * 1000
        vec_cold_times.append(cold_ms)

        # Warm solves
        n_runs = max(3, 10 - n // 1000)
        warm_timing = time_function(lambda: prob.solve(), n_warmup=0, n_runs=n_runs)
        vec_warm_times.append(warm_timing.mean_ms)

        speedup = cold_ms / warm_timing.mean_ms
        print(
            f"  n={n:5d}: Cold={cold_ms:8.1f}ms, Warm={warm_timing.mean_ms:8.1f}ms, "
            f"Speedup={speedup:.2f}x"
        )

    # Plot both Loop and VectorVariable cache benefits
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Loop-based
    ax1 = axes[0]
    x_pos = np.arange(len(loop_sizes))
    width = 0.35
    ax1.bar(
        x_pos - width / 2,
        loop_cold_times,
        width,
        label="Cold (first solve)",
        color="tab:red",
        alpha=0.8,
    )
    ax1.bar(
        x_pos + width / 2,
        loop_warm_times,
        width,
        label="Warm (cached)",
        color="tab:green",
        alpha=0.8,
    )
    ax1.set_xlabel("Problem Size (n)")
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Loop-based Variable Cache Benefit (n ≤ 500)")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([str(s) for s in loop_sizes])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Right: VectorVariable
    ax2 = axes[1]
    x_pos = np.arange(len(vec_sizes))
    ax2.bar(
        x_pos - width / 2,
        vec_cold_times,
        width,
        label="Cold (first solve)",
        color="tab:red",
        alpha=0.8,
    )
    ax2.bar(
        x_pos + width / 2,
        vec_warm_times,
        width,
        label="Warm (cached)",
        color="tab:green",
        alpha=0.8,
    )
    ax2.set_xlabel("Problem Size (n)")
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("VectorVariable Cache Benefit (n up to 5,000)")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(s) for s in vec_sizes], rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = RESULTS_DIR / "lp_cache_benefit.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {save_path}")
    plt.close()


def run_overhead_breakdown():
    """Generate overhead breakdown by problem type, including VectorVariable."""
    print("\n" + "=" * 70)
    print("OVERHEAD BREAKDOWN BY PROBLEM TYPE")
    print("=" * 70)

    categories = []
    overheads = []

    # Small LP (Loop)
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
    categories.append("Small LP\n(Loop, n=2)")
    overheads.append(optyx_t.mean_ms / scipy_t.mean_ms)
    print(f"  Small LP (Loop): {overheads[-1]:.2f}x")

    # Medium LP (Loop)
    n, m = 50, 25
    np.random.seed(42)
    c = np.random.rand(n)
    A_mat = np.random.rand(m, n)
    b_vec = np.sum(A_mat, axis=1) * 0.5
    x = np.array([Variable(f"mlp_x{i}", lb=0, ub=1) for i in range(n)])
    prob = Problem(name="overhead_med_lp_loop")
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
    categories.append("Medium LP\n(Loop, n=50)")
    overheads.append(optyx_t.mean_ms / scipy_t.mean_ms)
    print(f"  Medium LP (Loop): {overheads[-1]:.2f}x")

    # Medium LP (VectorVariable)
    x = VectorVariable("mlp_xv", n, lb=0, ub=1)
    prob = Problem(name="overhead_med_lp_vec")
    prob.maximize(c @ x)
    for i in range(m):
        prob.subject_to(A_mat[i] @ x <= b_vec[i])

    prob.solve()
    optyx_t = time_function(lambda: prob.solve(), n_warmup=2, n_runs=30)
    categories.append("Medium LP\n(VectorVar, n=50)")
    overheads.append(optyx_t.mean_ms / scipy_t.mean_ms)
    print(f"  Medium LP (VectorVar): {overheads[-1]:.2f}x")

    # Large LP (VectorVariable)
    n, m = 1000, 500
    np.random.seed(42)
    c = np.random.rand(n)
    A_mat = np.random.rand(m, n)
    b_vec = np.sum(A_mat, axis=1) * 0.5
    x = VectorVariable("llp_xv", n, lb=0, ub=1)
    prob = Problem(name="overhead_large_lp_vec")
    prob.maximize(c @ x)
    for i in range(m):
        prob.subject_to(A_mat[i] @ x <= b_vec[i])

    prob.solve()
    optyx_t = time_function(lambda: prob.solve(), n_warmup=2, n_runs=10)
    bounds = [(0, 1)] * n
    scipy_t = time_function(
        lambda: linprog(-c, A_ub=A_mat, b_ub=b_vec, bounds=bounds, method="highs"),
        n_warmup=2,
        n_runs=10,
    )
    categories.append("Large LP\n(VectorVar, n=1000)")
    overheads.append(optyx_t.mean_ms / scipy_t.mean_ms)
    print(f"  Large LP (VectorVar, n=1000): {overheads[-1]:.2f}x")

    # Very Large LP (VectorVariable)
    n, m = 5000, 2500
    np.random.seed(42)
    c = np.random.rand(n)
    A_mat = np.random.rand(m, n)
    b_vec = np.sum(A_mat, axis=1) * 0.5
    x = VectorVariable("vlp_xv", n, lb=0, ub=1)
    prob = Problem(name="overhead_vlarge_lp_vec")
    prob.maximize(c @ x)
    for i in range(m):
        prob.subject_to(A_mat[i] @ x <= b_vec[i])

    prob.solve()
    optyx_t = time_function(lambda: prob.solve(), n_warmup=1, n_runs=5)
    bounds = [(0, 1)] * n
    scipy_t = time_function(
        lambda: linprog(-c, A_ub=A_mat, b_ub=b_vec, bounds=bounds, method="highs"),
        n_warmup=1,
        n_runs=5,
    )
    categories.append("Very Large LP\n(VectorVar, n=5000)")
    overheads.append(optyx_t.mean_ms / scipy_t.mean_ms)
    print(f"  Very Large LP (VectorVar, n=5000): {overheads[-1]:.2f}x")

    # Rosenbrock NLP
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
    categories.append("Rosenbrock\nNLP")
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
    categories.append("Constrained\nQP")
    overheads.append(optyx_t.mean_ms / scipy_t.mean_ms)
    print(f"  Constrained QP: {overheads[-1]:.2f}x")

    plot_overhead_breakdown(
        categories,
        overheads,
        title="Optyx Overhead vs SciPy by Problem Type",
        save_path=RESULTS_DIR / "overhead_breakdown.png",
    )


def run_matrixvariable_scaling():
    """Benchmark MatrixVariable formulation at scale."""
    print("\n" + "=" * 70)
    print("MATRIXVARIABLE FORMULATION BENCHMARK")
    print("=" * 70)

    sizes = [10, 25, 50, 100, 150, 200]

    print("\n--- Formulation Speed: Loop vs MatrixVariable ---")
    for n in sizes:
        import time

        # Loop-based formulation
        start = time.perf_counter()
        _ = np.array(
            [[Variable(f"x_{i}_{j}", lb=0, ub=1) for j in range(n)] for i in range(n)]
        )
        loop_ms = (time.perf_counter() - start) * 1000

        # MatrixVariable formulation
        start = time.perf_counter()
        _ = MatrixVariable("X", n, n, lb=0, ub=1)
        mat_ms = (time.perf_counter() - start) * 1000

        speedup = loop_ms / mat_ms
        print(
            f"  {n:3d}x{n:<3d} ({n*n:5d} vars): Loop={loop_ms:8.2f}ms, "
            f"MatrixVar={mat_ms:6.3f}ms, Speedup={speedup:7.1f}x"
        )

    print("\n--- Large MatrixVariable (up to 316x316 = 100,000 vars) ---")
    large_sizes = [100, 150, 200, 250, 316]  # 316^2 ≈ 100,000

    for n in large_sizes:
        import time

        start = time.perf_counter()
        X = MatrixVariable("X", n, n, lb=0, ub=1)
        create_ms = (time.perf_counter() - start) * 1000

        # Access all variables (tests internal variable creation)
        start = time.perf_counter()
        _ = X[0, 0]  # First element
        _ = X[-1, -1]  # Last element
        access_ms = (time.perf_counter() - start) * 1000

        print(
            f"  {n:3d}x{n:<3d} ({n*n:6d} vars): Create={create_ms:6.2f}ms, "
            f"Access={access_ms:6.4f}ms"
        )


def run_multi_problem_scaling():
    """Generate combined scaling plot for all problem types with Loop and VectorVariable."""
    print("\n" + "=" * 70)
    print("MULTI-PROBLEM SCALING COMPARISON")
    print("=" * 70)

    # Loop sizes (up to 500)
    loop_sizes = [10, 25, 50, 100, 200, 500]
    # VectorVariable sizes (up to 5,000)
    vec_sizes = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]

    # LP data - Loop
    lp_loop_data = ScalingData(label="LP (Loop)")
    print("\n--- LP Loop-based ---")
    for n in loop_sizes:
        m = n // 2
        np.random.seed(42)
        c = np.random.rand(n)
        A = np.random.rand(m, n)
        b = np.sum(A, axis=1) * 0.5

        x = np.array([Variable(f"mslp_x{i}", lb=0, ub=1) for i in range(n)])
        prob = Problem(name=f"ms_lp_loop_{n}")
        prob.maximize(c @ x)
        for i in range(m):
            prob.subject_to(A[i] @ x <= b[i])

        prob.solve()  # Warm
        timing = time_function(lambda: prob.solve(), n_warmup=0, n_runs=10)
        lp_loop_data.add_point(n, timing.mean_ms, timing.std_ms)
        print(f"  n={n:4d}: {timing.mean_ms:.3f}ms")

    # LP data - VectorVariable
    lp_vec_data = ScalingData(label="LP (VectorVariable)")
    print("\n--- LP VectorVariable ---")
    for n in vec_sizes:
        m = n // 2
        np.random.seed(42)
        c = np.random.rand(n)
        A = np.random.rand(m, n)
        b = np.sum(A, axis=1) * 0.5

        x = VectorVariable("mslp_xv", n, lb=0, ub=1)
        prob = Problem(name=f"ms_lp_vec_{n}")
        prob.maximize(c @ x)
        for i in range(m):
            prob.subject_to(A[i] @ x <= b[i])

        prob.solve()  # Warm
        n_runs = max(3, 10 - n // 1000)
        timing = time_function(lambda: prob.solve(), n_warmup=0, n_runs=n_runs)
        lp_vec_data.add_point(n, timing.mean_ms, timing.std_ms)
        print(f"  n={n:5d}: {timing.mean_ms:.2f}ms")

    # Quadratic NLP data - Loop (limited sizes due to recursion)
    quad_loop_data = ScalingData(label="Quadratic NLP (Loop)")
    print("\n--- Quadratic NLP Loop-based ---")
    nlp_sizes = [10, 25, 50, 100, 150, 200]
    for n in nlp_sizes:
        x = np.array([Variable(f"msq_x{i}") for i in range(n)])
        prob = Problem(name=f"ms_quad_loop_{n}")
        prob.minimize(np.sum(x**2) - np.sum(x))

        x0 = np.zeros(n)
        timing = time_function(lambda: prob.solve(x0=x0), n_warmup=2, n_runs=5)
        quad_loop_data.add_point(n, timing.mean_ms, timing.std_ms)
        print(f"  n={n:4d}: {timing.mean_ms:.3f}ms")

    # Quadratic NLP data - VectorVariable (same sizes as Loop due to recursion limit)
    quad_vec_data = ScalingData(label="Quadratic NLP (VectorVariable)")
    print("\n--- Quadratic NLP VectorVariable ---")
    for n in nlp_sizes:
        x = VectorVariable("msq_xv", n)
        # Build sum incrementally
        obj = x[0] ** 2 - x[0]
        for i in range(1, n):
            obj = obj + x[i] ** 2 - x[i]

        prob = Problem(name=f"ms_quad_vec_{n}")
        prob.minimize(obj)

        x0 = np.zeros(n)
        timing = time_function(lambda: prob.solve(x0=x0), n_warmup=2, n_runs=5)
        quad_vec_data.add_point(n, timing.mean_ms, timing.std_ms)
        print(f"  n={n:4d}: {timing.mean_ms:.2f}ms")

    # Constrained QP data - Loop (limited sizes due to recursion)
    cqp_loop_data = ScalingData(label="Constrained QP (Loop)")
    print("\n--- Constrained QP Loop-based ---")
    cqp_sizes = [10, 25, 50, 100, 150, 200]
    for n in cqp_sizes:
        x = np.array([Variable(f"mscqp_x{i}", lb=0) for i in range(n)])
        prob = Problem(name=f"ms_cqp_loop_{n}")
        prob.minimize(np.sum(x**2))
        prob.subject_to(np.sum(x) >= 1)

        x0 = np.full(n, 0.1)
        timing = time_function(lambda: prob.solve(x0=x0), n_warmup=2, n_runs=5)
        cqp_loop_data.add_point(n, timing.mean_ms, timing.std_ms)
        print(f"  n={n:4d}: {timing.mean_ms:.3f}ms")

    # Constrained QP data - VectorVariable (same sizes as Loop due to recursion limit)
    cqp_vec_data = ScalingData(label="Constrained QP (VectorVariable)")
    print("\n--- Constrained QP VectorVariable ---")
    for n in cqp_sizes:
        x = VectorVariable("mscqp_xv", n, lb=0)
        # Build sums incrementally
        obj = x[0] ** 2
        constraint_sum = x[0]
        for i in range(1, n):
            obj = obj + x[i] ** 2
            constraint_sum = constraint_sum + x[i]

        prob = Problem(name=f"ms_cqp_vec_{n}")
        prob.minimize(obj)
        prob.subject_to(constraint_sum >= 1)

        x0 = np.full(n, 0.1)
        timing = time_function(lambda: prob.solve(x0=x0), n_warmup=2, n_runs=5)
        cqp_vec_data.add_point(n, timing.mean_ms, timing.std_ms)
        print(f"  n={n:4d}: {timing.mean_ms:.2f}ms")

    plot_multi_scaling(
        [
            lp_loop_data,
            lp_vec_data,
            quad_loop_data,
            quad_vec_data,
            cqp_loop_data,
            cqp_vec_data,
        ],
        title="Optyx Scaling by Problem Type: Loop vs VectorVariable",
        save_path=RESULTS_DIR / "multi_problem_scaling.png",
    )


def main():
    """Run all benchmarks and generate plots."""
    print("=" * 70)
    print("OPTYX BENCHMARK SUITE")
    print("=" * 70)
    print(f"Results will be saved to: {RESULTS_DIR}")

    # Ensure results directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run core benchmarks with Loop vs VectorVariable vs SciPy
    run_lp_vs_scipy_scaling()
    run_nlp_vs_scipy_scaling()
    run_cache_benefit_analysis()
    run_overhead_breakdown()
    run_multi_problem_scaling()

    # Run MatrixVariable formulation benchmark
    run_matrixvariable_scaling()

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print(f"Plots saved to: {RESULTS_DIR}")
    print("Files generated:")
    for f in sorted(RESULTS_DIR.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
