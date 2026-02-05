#!/usr/bin/env python
"""Run all benchmarks and generate performance plots.

This benchmark suite measures END-TO-END performance, including:
- Variable/Problem creation time
- Constraint setup time
- Cold solve (first solve, includes compilation)
- Warm solve (cached, subsequent solves)

This provides a FAIR comparison against SciPy, which has no build phase.

Usage:
    uv run python benchmarks/run_benchmarks.py

Generates plots in benchmarks/results/:
    - lp_scaling_comparison.png: LP scaling with cold/warm breakdown
    - nlp_scaling_comparison.png: NLP scaling with cold/warm breakdown
    - overhead_breakdown.png: Overhead by problem type
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Add benchmarks to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.optimize import linprog, minimize

from optyx import Variable, VectorVariable, Problem
from utils import RESULTS_DIR

import matplotlib.pyplot as plt


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    n: int
    build_ms: float  # Time to create variables, problem, constraints
    cold_solve_ms: float  # First solve (includes compilation)
    warm_solve_ms: float  # Average of subsequent solves
    scipy_ms: float  # SciPy baseline

    @property
    def cold_total_ms(self) -> float:
        """Total time for cold solve (build + first solve)."""
        return self.build_ms + self.cold_solve_ms

    @property
    def warm_total_ms(self) -> float:
        """Total time for warm solve (build + cached solve)."""
        return self.build_ms + self.warm_solve_ms

    @property
    def cold_overhead(self) -> float:
        """Cold overhead vs SciPy."""
        return self.cold_total_ms / self.scipy_ms if self.scipy_ms > 0 else float("inf")

    @property
    def warm_overhead(self) -> float:
        """Warm overhead vs SciPy."""
        return self.warm_total_ms / self.scipy_ms if self.scipy_ms > 0 else float("inf")


@dataclass
class ScalingResults:
    """Results for a scaling benchmark series."""

    label: str
    results: list[BenchmarkResult] = field(default_factory=list)

    def add(self, result: BenchmarkResult) -> None:
        self.results.append(result)

    @property
    def sizes(self) -> list[int]:
        return [r.n for r in self.results]

    @property
    def cold_totals(self) -> list[float]:
        return [r.cold_total_ms for r in self.results]

    @property
    def warm_totals(self) -> list[float]:
        return [r.warm_total_ms for r in self.results]

    @property
    def scipy_times(self) -> list[float]:
        return [r.scipy_ms for r in self.results]


def time_scipy_lp(
    c: np.ndarray, A: np.ndarray, b: np.ndarray, n_runs: int = 5
) -> float:
    """Time SciPy LP solve (average of n_runs)."""
    bounds = [(0, 1)] * len(c)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        linprog(-c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        times.append((time.perf_counter() - start) * 1000)
    return np.mean(times)


def time_scipy_nlp(n: int, n_runs: int = 5) -> float:
    """Time SciPy unconstrained NLP solve (average of n_runs)."""

    def obj(v):
        return np.sum(v**2) - np.sum(v)

    def grad(v):
        return 2 * v - 1

    x0 = np.zeros(n)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        minimize(obj, x0, jac=grad, method="BFGS")
        times.append((time.perf_counter() - start) * 1000)
    return np.mean(times)


def time_scipy_constrained_nlp(n: int, n_runs: int = 5) -> float:
    """Time SciPy constrained NLP solve (average of n_runs)."""

    def obj(v):
        return np.sum(v**2)

    def grad(v):
        return 2 * v

    constraints = {
        "type": "ineq",
        "fun": lambda v: np.sum(v) - 1,
        "jac": lambda v: np.ones(n),
    }

    # Bounds to match Optyx: x >= 0
    bounds = [(0, None) for _ in range(n)]

    x0 = np.full(n, 0.1)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        minimize(
            obj, x0, jac=grad, method="SLSQP", constraints=constraints, bounds=bounds
        )
        times.append((time.perf_counter() - start) * 1000)
    return np.mean(times)


def benchmark_lp_loop(
    n: int, c: np.ndarray, A: np.ndarray, b: np.ndarray
) -> BenchmarkResult:
    """Benchmark LP with loop-based variables (full end-to-end)."""
    m = len(b)

    # Time build phase
    start = time.perf_counter()
    x = np.array([Variable(f"x{i}", lb=0, ub=1) for i in range(n)])
    prob = Problem(name=f"lp_loop_{n}")
    prob.maximize(c @ x)
    for i in range(m):
        prob.subject_to(A[i] @ x <= b[i])
    build_ms = (time.perf_counter() - start) * 1000

    # Time cold solve
    start = time.perf_counter()
    prob.solve()
    cold_solve_ms = (time.perf_counter() - start) * 1000

    # Time warm solves (3 runs)
    warm_times = []
    for _ in range(3):
        start = time.perf_counter()
        prob.solve()
        warm_times.append((time.perf_counter() - start) * 1000)
    warm_solve_ms = np.mean(warm_times)

    # SciPy baseline
    scipy_ms = time_scipy_lp(c, A, b)

    return BenchmarkResult(n, build_ms, cold_solve_ms, warm_solve_ms, scipy_ms)


def benchmark_lp_vector(
    n: int, c: np.ndarray, A: np.ndarray, b: np.ndarray
) -> BenchmarkResult:
    """Benchmark LP with VectorVariable (full end-to-end)."""
    m = len(b)

    # Time build phase
    start = time.perf_counter()
    x = VectorVariable("x", n, lb=0, ub=1)
    prob = Problem(name=f"lp_vec_{n}")
    prob.maximize(c @ x)
    for i in range(m):
        prob.subject_to(A[i] @ x <= b[i])
    build_ms = (time.perf_counter() - start) * 1000

    # Time cold solve
    start = time.perf_counter()
    prob.solve()
    cold_solve_ms = (time.perf_counter() - start) * 1000

    # Time warm solves (3 runs)
    warm_times = []
    for _ in range(3):
        start = time.perf_counter()
        prob.solve()
        warm_times.append((time.perf_counter() - start) * 1000)
    warm_solve_ms = np.mean(warm_times)

    # SciPy baseline
    scipy_ms = time_scipy_lp(c, A, b)

    return BenchmarkResult(n, build_ms, cold_solve_ms, warm_solve_ms, scipy_ms)


def benchmark_nlp_loop(n: int) -> BenchmarkResult:
    """Benchmark unconstrained NLP with loop-based variables."""
    x0 = np.zeros(n)

    # Time build phase
    start = time.perf_counter()
    x = np.array([Variable(f"x{i}") for i in range(n)])
    prob = Problem(name=f"nlp_loop_{n}")
    prob.minimize(np.sum(x**2) - np.sum(x))
    build_ms = (time.perf_counter() - start) * 1000

    # Time cold solve
    start = time.perf_counter()
    prob.solve(x0=x0)
    cold_solve_ms = (time.perf_counter() - start) * 1000

    # Time warm solves (3 runs)
    warm_times = []
    for _ in range(3):
        start = time.perf_counter()
        prob.solve(x0=x0)
        warm_times.append((time.perf_counter() - start) * 1000)
    warm_solve_ms = np.mean(warm_times)

    # SciPy baseline
    scipy_ms = time_scipy_nlp(n)

    return BenchmarkResult(n, build_ms, cold_solve_ms, warm_solve_ms, scipy_ms)


def benchmark_nlp_vector(n: int) -> BenchmarkResult:
    """Benchmark unconstrained NLP with VectorVariable (vectorized ops)."""
    x0 = np.zeros(n)

    # Time build phase
    start = time.perf_counter()
    x = VectorVariable("x", n)
    prob = Problem(name=f"nlp_vec_{n}")
    # Vectorized: x.dot(x) - x.sum() creates O(1) depth tree
    prob.minimize(x.dot(x) - x.sum())
    build_ms = (time.perf_counter() - start) * 1000

    # Time cold solve
    start = time.perf_counter()
    prob.solve(x0=x0)
    cold_solve_ms = (time.perf_counter() - start) * 1000

    # Time warm solves (3 runs)
    warm_times = []
    for _ in range(3):
        start = time.perf_counter()
        prob.solve(x0=x0)
        warm_times.append((time.perf_counter() - start) * 1000)
    warm_solve_ms = np.mean(warm_times)

    # SciPy baseline
    scipy_ms = time_scipy_nlp(n)

    return BenchmarkResult(n, build_ms, cold_solve_ms, warm_solve_ms, scipy_ms)


def benchmark_cqp_loop(n: int) -> BenchmarkResult:
    """Benchmark constrained QP with loop-based variables."""
    x0 = np.full(n, 0.1)

    # Time build phase
    start = time.perf_counter()
    x = np.array([Variable(f"x{i}", lb=0) for i in range(n)])
    prob = Problem(name=f"cqp_loop_{n}")
    prob.minimize(np.sum(x**2))
    prob.subject_to(np.sum(x) >= 1)
    build_ms = (time.perf_counter() - start) * 1000

    # Time cold solve
    start = time.perf_counter()
    prob.solve(x0=x0)
    cold_solve_ms = (time.perf_counter() - start) * 1000

    # Time warm solves (3 runs)
    warm_times = []
    for _ in range(3):
        start = time.perf_counter()
        prob.solve(x0=x0)
        warm_times.append((time.perf_counter() - start) * 1000)
    warm_solve_ms = np.mean(warm_times)

    # SciPy baseline
    scipy_ms = time_scipy_constrained_nlp(n)

    return BenchmarkResult(n, build_ms, cold_solve_ms, warm_solve_ms, scipy_ms)


def benchmark_cqp_vector(n: int) -> BenchmarkResult:
    """Benchmark constrained QP with VectorVariable (vectorized ops)."""
    x0 = np.full(n, 0.1)

    # Time build phase
    start = time.perf_counter()
    x = VectorVariable("x", n, lb=0)
    prob = Problem(name=f"cqp_vec_{n}")
    # Vectorized: x.dot(x) and x.sum() create O(1) depth trees
    prob.minimize(x.dot(x))
    prob.subject_to(x.sum() >= 1)
    build_ms = (time.perf_counter() - start) * 1000

    # Time cold solve
    start = time.perf_counter()
    prob.solve(x0=x0)
    cold_solve_ms = (time.perf_counter() - start) * 1000

    # Time warm solves (3 runs)
    warm_times = []
    for _ in range(3):
        start = time.perf_counter()
        prob.solve(x0=x0)
        warm_times.append((time.perf_counter() - start) * 1000)
    warm_solve_ms = np.mean(warm_times)

    # SciPy baseline
    scipy_ms = time_scipy_constrained_nlp(n)

    return BenchmarkResult(n, build_ms, cold_solve_ms, warm_solve_ms, scipy_ms)


def print_result(label: str, r: BenchmarkResult) -> None:
    """Print a benchmark result with full breakdown."""
    print(
        f"  n={r.n:5d}: Build={r.build_ms:8.1f}ms, "
        f"Cold={r.cold_solve_ms:8.1f}ms, Warm={r.warm_solve_ms:7.1f}ms | "
        f"SciPy={r.scipy_ms:7.1f}ms | "
        f"Cold overhead={r.cold_overhead:5.1f}x, Warm overhead={r.warm_overhead:5.1f}x"
    )


def run_lp_scaling():
    """Run LP scaling benchmarks."""
    print("\n" + "=" * 80)
    print("LP SCALING BENCHMARK (End-to-End)")
    print("=" * 80)
    print("\nMeasures: Build (vars + problem + constraints) + Solve")
    print("Compared against: SciPy linprog (no build phase)")

    # Loop-based: limited to n=100 due to exponential cold solve time
    loop_sizes = [10, 25, 50, 100]
    # VectorVariable: scale to n=5000 to test large problems
    vec_sizes = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]

    loop_results = ScalingResults(label="LP (Loop)")
    vec_results = ScalingResults(label="LP (VectorVariable)")

    print("\n--- Loop-based Variable (n ≤ 100, slow cold solve) ---")
    for n in loop_sizes:
        m = n // 2
        np.random.seed(42)
        c = np.random.rand(n)
        A = np.random.rand(m, n)
        b = np.sum(A, axis=1) * 0.5

        r = benchmark_lp_loop(n, c, A, b)
        loop_results.add(r)
        print_result("Loop", r)

    print("\n--- VectorVariable (n ≤ 5,000) ---")
    for n in vec_sizes:
        m = n // 2
        np.random.seed(42)
        c = np.random.rand(n)
        A = np.random.rand(m, n)
        b = np.sum(A, axis=1) * 0.5

        r = benchmark_lp_vector(n, c, A, b)
        vec_results.add(r)
        print_result("Vec", r)

    # Plot
    plot_scaling_comparison(
        loop_results,
        vec_results,
        title="LP Scaling: End-to-End Time vs SciPy",
        save_path=RESULTS_DIR / "lp_scaling_comparison.png",
    )

    return loop_results, vec_results


def run_nlp_scaling():
    """Run NLP scaling benchmarks."""
    print("\n" + "=" * 80)
    print("UNCONSTRAINED NLP SCALING BENCHMARK (End-to-End)")
    print("=" * 80)
    print("\nObjective: min Σx²ᵢ - Σxᵢ (optimal at x* = 0.5)")
    print("Measures: Build + Solve (includes gradient compilation)")

    # Loop-based: limited to n=100 due to exponential cold solve time
    loop_sizes = [10, 25, 50, 100]
    # VectorVariable with vectorized ops: scale to n=10000 to test large problems
    vec_sizes = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]

    loop_results = ScalingResults(label="NLP (Loop)")
    vec_results = ScalingResults(label="NLP (VectorVariable)")

    print("\n--- Loop-based Variable (n ≤ 100, slow cold solve) ---")
    for n in loop_sizes:
        r = benchmark_nlp_loop(n)
        loop_results.add(r)
        print_result("Loop", r)

    print("\n--- VectorVariable with x.dot(x) - x.sum() (n ≤ 5,000) ---")
    for n in vec_sizes:
        r = benchmark_nlp_vector(n)
        vec_results.add(r)
        print_result("Vec", r)

    # Plot
    plot_scaling_comparison(
        loop_results,
        vec_results,
        title="Unconstrained NLP: End-to-End Time vs SciPy",
        save_path=RESULTS_DIR / "nlp_scaling_comparison.png",
    )

    return loop_results, vec_results


def run_cqp_scaling():
    """Run constrained QP scaling benchmarks."""
    print("\n" + "=" * 80)
    print("CONSTRAINED QP SCALING BENCHMARK (End-to-End)")
    print("=" * 80)
    print("\nObjective: min Σx²ᵢ s.t. Σxᵢ ≥ 1, xᵢ ≥ 0")
    print("Measures: Build + Solve (includes gradient/Jacobian compilation)")

    # Loop-based: limited to n=100 due to exponential cold solve time
    loop_sizes = [10, 25, 50, 100]
    # VectorVariable: scale to n=5000 to test large problems
    vec_sizes = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]

    loop_results = ScalingResults(label="CQP (Loop)")
    vec_results = ScalingResults(label="CQP (VectorVariable)")

    print("\n--- Loop-based Variable (n ≤ 100, slow cold solve) ---")
    for n in loop_sizes:
        r = benchmark_cqp_loop(n)
        loop_results.add(r)
        print_result("Loop", r)

    print("\n--- VectorVariable with x.dot(x), x.sum() (n ≤ 5,000) ---")
    for n in vec_sizes:
        r = benchmark_cqp_vector(n)
        vec_results.add(r)
        print_result("Vec", r)

    # Plot
    plot_scaling_comparison(
        loop_results,
        vec_results,
        title="Constrained QP: End-to-End Time vs SciPy",
        save_path=RESULTS_DIR / "cqp_scaling_comparison.png",
    )

    return loop_results, vec_results


def plot_scaling_comparison(
    loop_results: ScalingResults,
    vec_results: ScalingResults,
    title: str,
    save_path: Path,
) -> None:
    """Plot scaling comparison with cold/warm breakdown."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Absolute times (log scale)
    ax1 = axes[0]

    # Loop cold/warm
    if loop_results.results:
        ax1.plot(
            loop_results.sizes,
            loop_results.cold_totals,
            "o-",
            color="tab:red",
            label="Loop (cold)",
            linewidth=2,
            markersize=6,
        )
        ax1.plot(
            loop_results.sizes,
            loop_results.warm_totals,
            "s--",
            color="tab:red",
            label="Loop (warm)",
            linewidth=1.5,
            markersize=5,
            alpha=0.7,
        )

    # Vector cold/warm
    if vec_results.results:
        ax1.plot(
            vec_results.sizes,
            vec_results.cold_totals,
            "o-",
            color="tab:blue",
            label="VectorVar (cold)",
            linewidth=2,
            markersize=6,
        )
        ax1.plot(
            vec_results.sizes,
            vec_results.warm_totals,
            "s--",
            color="tab:blue",
            label="VectorVar (warm)",
            linewidth=1.5,
            markersize=5,
            alpha=0.7,
        )

    # SciPy baseline (use vec_results since it has more points)
    baseline = vec_results if vec_results.results else loop_results
    ax1.plot(
        baseline.sizes,
        baseline.scipy_times,
        "^-",
        color="tab:green",
        label="SciPy",
        linewidth=2,
        markersize=6,
    )

    ax1.set_xlabel("Problem Size (n)", fontsize=11)
    ax1.set_ylabel("Time (ms)", fontsize=11)
    ax1.set_title("Absolute Time (log scale)", fontsize=12)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: Overhead ratio vs SciPy
    ax2 = axes[1]

    if loop_results.results:
        cold_overhead = [r.cold_overhead for r in loop_results.results]
        warm_overhead = [r.warm_overhead for r in loop_results.results]
        ax2.plot(
            loop_results.sizes,
            cold_overhead,
            "o-",
            color="tab:red",
            label="Loop (cold)",
            linewidth=2,
            markersize=6,
        )
        ax2.plot(
            loop_results.sizes,
            warm_overhead,
            "s--",
            color="tab:red",
            label="Loop (warm)",
            linewidth=1.5,
            markersize=5,
            alpha=0.7,
        )

    if vec_results.results:
        cold_overhead = [r.cold_overhead for r in vec_results.results]
        warm_overhead = [r.warm_overhead for r in vec_results.results]
        ax2.plot(
            vec_results.sizes,
            cold_overhead,
            "o-",
            color="tab:blue",
            label="VectorVar (cold)",
            linewidth=2,
            markersize=6,
        )
        ax2.plot(
            vec_results.sizes,
            warm_overhead,
            "s--",
            color="tab:blue",
            label="VectorVar (warm)",
            linewidth=1.5,
            markersize=5,
            alpha=0.7,
        )

    ax2.axhline(y=1.0, color="green", linestyle="--", linewidth=2, label="SciPy (1.0x)")
    ax2.axhline(y=2.0, color="black", linestyle=":", alpha=0.5, label="2x overhead")
    # ax2.axhline(y=10.0, color="gray", linestyle=":", alpha=0.3, label="10x overhead")

    ax2.set_xlabel("Problem Size (n)", fontsize=11)
    ax2.set_ylabel("Overhead vs SciPy (×)", fontsize=11)
    ax2.set_title("Overhead Ratio (log scale, lower is better)", fontsize=12)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3, which="both")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {save_path}")
    plt.close()


def run_overhead_summary():
    """Generate overhead summary for common problem types."""
    print("\n" + "=" * 80)
    print("OVERHEAD SUMMARY BY PROBLEM TYPE")
    print("=" * 80)

    categories = []
    cold_overheads = []
    warm_overheads = []

    # Small LP (n=50)
    n, m = 50, 25
    np.random.seed(42)
    c = np.random.rand(n)
    A = np.random.rand(m, n)
    b = np.sum(A, axis=1) * 0.5

    r = benchmark_lp_vector(n, c, A, b)
    categories.append(f"LP\nn={n}")
    cold_overheads.append(r.cold_overhead)
    warm_overheads.append(r.warm_overhead)
    print(f"LP (n={n}): Cold={r.cold_overhead:.1f}x, Warm={r.warm_overhead:.1f}x")

    # Medium LP (n=500)
    n, m = 500, 250
    c = np.random.rand(n)
    A = np.random.rand(m, n)
    b = np.sum(A, axis=1) * 0.5

    r = benchmark_lp_vector(n, c, A, b)
    categories.append(f"LP\nn={n}")
    cold_overheads.append(r.cold_overhead)
    warm_overheads.append(r.warm_overhead)
    print(f"LP (n={n}): Cold={r.cold_overhead:.1f}x, Warm={r.warm_overhead:.1f}x")

    # Small NLP (n=50)
    r = benchmark_nlp_vector(50)
    categories.append("NLP\nn=50")
    cold_overheads.append(r.cold_overhead)
    warm_overheads.append(r.warm_overhead)
    print(f"NLP (n=50): Cold={r.cold_overhead:.1f}x, Warm={r.warm_overhead:.1f}x")

    # Medium NLP (n=500)
    r = benchmark_nlp_vector(500)
    categories.append("NLP\nn=500")
    cold_overheads.append(r.cold_overhead)
    warm_overheads.append(r.warm_overhead)
    print(f"NLP (n=500): Cold={r.cold_overhead:.1f}x, Warm={r.warm_overhead:.1f}x")

    # Small CQP (n=50)
    r = benchmark_cqp_vector(50)
    categories.append("CQP\nn=50")
    cold_overheads.append(r.cold_overhead)
    warm_overheads.append(r.warm_overhead)
    print(f"CQP (n=50): Cold={r.cold_overhead:.1f}x, Warm={r.warm_overhead:.1f}x")

    # Medium CQP (n=500)
    r = benchmark_cqp_vector(500)
    categories.append("CQP\nn=500")
    cold_overheads.append(r.cold_overhead)
    warm_overheads.append(r.warm_overhead)
    print(f"CQP (n=500): Cold={r.cold_overhead:.1f}x, Warm={r.warm_overhead:.1f}x")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2,
        cold_overheads,
        width,
        label="Cold (first solve)",
        color="tab:red",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        warm_overheads,
        width,
        label="Warm (cached)",
        color="tab:green",
        alpha=0.8,
    )

    ax.axhline(
        y=1.0, color="black", linestyle="--", linewidth=1, label="SciPy baseline"
    )
    ax.axhline(y=2.0, color="gray", linestyle=":", alpha=0.5)

    ax.set_ylabel("Overhead vs SciPy (×)", fontsize=11)
    ax.set_yscale("log")
    ax.set_title(
        "Optyx End-to-End Overhead by Problem Type\n(VectorVariable, lower is better)",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars1, cold_overheads):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.1f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar, val in zip(bars2, warm_overheads):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.1f}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    save_path = RESULTS_DIR / "overhead_breakdown.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {save_path}")
    plt.close()


def main():
    """Run all benchmarks."""
    print("=" * 80)
    print("OPTYX BENCHMARK SUITE - END-TO-END COMPARISON")
    print("=" * 80)
    print("\nThis benchmark measures TOTAL time including:")
    print("  • Variable creation")
    print("  • Problem setup")
    print("  • Constraint construction")
    print("  • Cold solve (first solve, includes compilation)")
    print("  • Warm solve (cached subsequent solves)")
    print("\nCompared against SciPy (which has no build phase).")
    print(f"\nResults will be saved to: {RESULTS_DIR}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Run scaling benchmarks
    run_lp_scaling()
    run_nlp_scaling()
    run_cqp_scaling()

    # Run overhead summary
    run_overhead_summary()

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"\nPlots saved to: {RESULTS_DIR}")
    for f in sorted(RESULTS_DIR.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
