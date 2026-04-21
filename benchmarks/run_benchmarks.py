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
    - benchmark_metadata.json: machine and dependency metadata for the run
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

# Add benchmarks to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from scipy.optimize import linprog, minimize

from optyx import Variable, VectorVariable, BinaryVariable, Problem
from utils import RESULTS_DIR, write_benchmark_metadata

import matplotlib.pyplot as plt


DOCS_BENCHMARKS_DIR = (
    Path(__file__).resolve().parents[1] / "docs" / "assets" / "benchmarks"
)

PLOT_FILE_NAMES = {
    "lp_scaling": "lp_scaling_comparison.png",
    "nlp_scaling": "nlp_scaling_comparison.png",
    "cqp_scaling": "cqp_scaling_comparison.png",
    "milp_scaling": "milp_scaling_comparison.png",
    "overhead_breakdown": "overhead_breakdown.png",
}

SUMMARY_TARGET_SIZES = {
    "LP": [50, 500, 5000],
    "NLP": [50, 500, 5000],
    "CQP": [50, 500, 5000],
    "MILP": [50, 500, 5000],
}


class Tee:
    """Helper to write to both stdout and a file."""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


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
        """Total time for warm solve (solve-only, excludes build)."""
        return self.warm_solve_ms

    @property
    def cold_overhead(self) -> float:
        """Cold overhead vs SciPy."""
        return self.cold_total_ms / self.scipy_ms if self.scipy_ms > 0 else float("inf")

    @property
    def warm_overhead(self) -> float:
        """Warm overhead vs SciPy (solve-only, excludes build)."""
        return self.warm_solve_ms / self.scipy_ms if self.scipy_ms > 0 else float("inf")


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


def benchmark_result_to_dict(result: BenchmarkResult) -> dict[str, float | int]:
    """Serialize a benchmark result to JSON-compatible primitives."""
    return {
        "n": int(result.n),
        "build_ms": float(result.build_ms),
        "cold_solve_ms": float(result.cold_solve_ms),
        "warm_solve_ms": float(result.warm_solve_ms),
        "scipy_ms": float(result.scipy_ms),
        "cold_total_ms": float(result.cold_total_ms),
        "warm_total_ms": float(result.warm_total_ms),
        "cold_overhead": float(result.cold_overhead),
        "warm_overhead": float(result.warm_overhead),
    }


def scaling_results_to_dict(results: ScalingResults) -> dict[str, object]:
    """Serialize a scaling result series for documentation consumption."""
    return {
        "label": results.label,
        "results": [benchmark_result_to_dict(result) for result in results.results],
    }


def find_benchmark_result(
    results: ScalingResults,
    n: int,
) -> BenchmarkResult:
    """Look up a benchmark result by size."""
    for result in results.results:
        if result.n == n:
            return result
    raise ValueError(f"No benchmark result for n={n} in {results.label}")


def summary_note(problem_type: str, result: BenchmarkResult, *, max_size: int) -> str:
    """Generate a brief note for the performance summary table."""
    if problem_type == "NLP":
        if result.n >= max_size:
            return "Simple quadratic; SciPy converges almost instantly"
        return "Autodiff overhead on a trivially simple objective"
    if problem_type == "LP":
        if result.n >= max_size:
            return "Scales to large LPs while staying near parity"
        return "Near-parity with SciPy linprog"
    if problem_type == "CQP":
        if result.n >= max_size:
            return "Exact Jacobians keep constrained solves near parity"
        return "O(1) Jacobian compilation for vectorized constraints"
    if problem_type == "MILP":
        if result.n >= max_size:
            return "Scales to large binary knapsack problems"
        return "Near-parity with SciPy milp"
    return "Derived from the latest benchmark run"


def build_benchmark_payload(
    *,
    lp_loop_results: ScalingResults,
    lp_vec_results: ScalingResults,
    nlp_loop_results: ScalingResults,
    nlp_vec_results: ScalingResults,
    cqp_loop_results: ScalingResults,
    cqp_vec_results: ScalingResults,
    milp_loop_results: ScalingResults,
    milp_vec_results: ScalingResults,
) -> dict[str, object]:
    """Build structured benchmark data for the documentation page."""
    vector_series = {
        "LP": lp_vec_results,
        "NLP": nlp_vec_results,
        "CQP": cqp_vec_results,
        "MILP": milp_vec_results,
    }

    performance_summary = []
    for problem_type, sizes in SUMMARY_TARGET_SIZES.items():
        series = vector_series[problem_type]
        max_size = max(result.n for result in series.results)
        for size in sizes:
            result = find_benchmark_result(series, size)
            performance_summary.append(
                {
                    "problem_type": problem_type,
                    "size": int(size),
                    "cold_overhead": float(result.cold_overhead),
                    "warm_overhead": float(result.warm_overhead),
                    "note": summary_note(problem_type, result, max_size=max_size),
                }
            )

    overhead_summary = []
    for problem_type, series in vector_series.items():
        for size in (50, max(result.n for result in series.results)):
            result = find_benchmark_result(series, size)
            overhead_summary.append(
                {
                    "problem_type": problem_type,
                    "size": int(size),
                    "cold_overhead": float(result.cold_overhead),
                    "warm_overhead": float(result.warm_overhead),
                }
            )

    return {
        "benchmark_suite": "run_benchmarks",
        "artifacts": {
            "plots": PLOT_FILE_NAMES,
            "metadata": "benchmark_metadata.json",
            "output_log": "benchmark_output.txt",
            "results_json": "benchmark_results.json",
        },
        "performance_summary": performance_summary,
        "overhead_summary": overhead_summary,
        "scaling": {
            "lp": {
                "loop": scaling_results_to_dict(lp_loop_results),
                "vector": scaling_results_to_dict(lp_vec_results),
            },
            "nlp": {
                "loop": scaling_results_to_dict(nlp_loop_results),
                "vector": scaling_results_to_dict(nlp_vec_results),
            },
            "cqp": {
                "loop": scaling_results_to_dict(cqp_loop_results),
                "vector": scaling_results_to_dict(cqp_vec_results),
            },
            "milp": {
                "loop": scaling_results_to_dict(milp_loop_results),
                "vector": scaling_results_to_dict(milp_vec_results),
            },
        },
    }


def write_benchmark_results_json(
    payload: dict[str, object],
    file_name: str = "benchmark_results.json",
) -> Path:
    """Persist structured benchmark results for documentation rendering."""
    output_path = RESULTS_DIR / file_name
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output_path


def sync_results_to_docs_assets(files: list[Path]) -> list[Path]:
    """Copy benchmark artifacts into docs/assets/benchmarks."""
    DOCS_BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    seen: set[str] = set()
    for file_path in files:
        if not file_path.exists() or file_path.name in seen:
            continue
        destination = DOCS_BENCHMARKS_DIR / file_path.name
        shutil.copy2(file_path, destination)
        copied.append(destination)
        seen.add(file_path.name)

    return copied


def collect_results_artifacts() -> list[Path]:
    """Collect result artifacts that should be mirrored into docs assets."""
    artifacts: list[Path] = []
    for pattern in ("*.json", "*.txt", "*.png"):
        artifacts.extend(sorted(RESULTS_DIR.glob(pattern)))
    return artifacts


def time_scipy_lp(
    c: np.ndarray, A: np.ndarray, b: np.ndarray, n_runs: int = 5
) -> float:
    """Time SciPy LP solve (average of n_runs)."""
    bounds = [(0, 1)] * len(c)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        res = linprog(-c, A_ub=A, b_ub=b, bounds=bounds, method="highs")
        times.append((time.perf_counter() - start) * 1000)
        assert res.success, f"SciPy LP failed: {res.message}"
    return float(np.mean(times))


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
        # Use L-BFGS-B to match optyx's auto-selected method for unconstrained NLP
        res = minimize(obj, x0, jac=grad, method="L-BFGS-B")
        times.append((time.perf_counter() - start) * 1000)
        assert res.success, f"SciPy NLP failed: {res.message}"
    return float(np.mean(times))


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
        res = minimize(
            obj, x0, jac=grad, method="SLSQP", constraints=constraints, bounds=bounds
        )
        times.append((time.perf_counter() - start) * 1000)
        assert res.success, f"SciPy CQP failed: {res.message}"
    return float(np.mean(times))


def time_scipy_milp(
    c: np.ndarray,
    capacity: int,
    n_runs: int = 5,
) -> float:
    """Time SciPy MILP solve (average of n_runs).

    Single-constraint binary knapsack: max c'x s.t. sum(x) <= capacity, x in {0,1}.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds

    n = len(c)
    bounds = Bounds(lb=cast(Any, np.zeros(n)), ub=cast(Any, np.ones(n)))
    constraints = LinearConstraint(np.ones((1, n)), -np.inf, capacity)
    integrality = np.ones(n, dtype=int)
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        res = milp(-c, constraints=constraints, integrality=integrality, bounds=bounds)
        times.append((time.perf_counter() - start) * 1000)
        assert res.success, f"SciPy MILP failed: {res.message}"
    return float(np.mean(times))


def benchmark_lp_loop(
    n: int, c: np.ndarray, A: np.ndarray, b: np.ndarray
) -> BenchmarkResult:
    """Benchmark LP with loop-based variables (full end-to-end)."""
    m = len(b)

    # Time build phase
    start = time.perf_counter()
    x = np.array([Variable(f"x{i}", lb=0, ub=1) for i in range(n)])
    prob = Problem(name=f"lp_loop_{n}")
    prob.maximize(sum(float(c[i]) * x[i] for i in range(n)))
    for i in range(m):
        prob.subject_to(A[i] @ x <= b[i])
    build_ms = (time.perf_counter() - start) * 1000

    # Time cold solve
    start = time.perf_counter()
    prob.solve()
    cold_solve_ms = (time.perf_counter() - start) * 1000

    # Time warm solves (5 runs)
    warm_times = []
    for _ in range(5):
        start = time.perf_counter()
        prob.solve()
        warm_times.append((time.perf_counter() - start) * 1000)
    warm_solve_ms = float(np.mean(warm_times))

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
    prob.maximize(x @ c)
    for i in range(m):
        prob.subject_to(A[i] @ x <= b[i])
    build_ms = (time.perf_counter() - start) * 1000

    # Time cold solve
    start = time.perf_counter()
    prob.solve()
    cold_solve_ms = (time.perf_counter() - start) * 1000

    # Time warm solves (5 runs)
    warm_times = []
    for _ in range(5):
        start = time.perf_counter()
        prob.solve()
        warm_times.append((time.perf_counter() - start) * 1000)
    warm_solve_ms = float(np.mean(warm_times))

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

    # Time warm solves (5 runs)
    warm_times = []
    for _ in range(5):
        start = time.perf_counter()
        prob.solve(x0=x0)
        warm_times.append((time.perf_counter() - start) * 1000)
    warm_solve_ms = float(np.mean(warm_times))

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

    # Time warm solves (5 runs)
    warm_times = []
    for _ in range(5):
        start = time.perf_counter()
        prob.solve(x0=x0)
        warm_times.append((time.perf_counter() - start) * 1000)
    warm_solve_ms = float(np.mean(warm_times))

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

    # Time warm solves (5 runs)
    warm_times = []
    for _ in range(5):
        start = time.perf_counter()
        prob.solve(x0=x0)
        warm_times.append((time.perf_counter() - start) * 1000)
    warm_solve_ms = float(np.mean(warm_times))

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

    # Time warm solves (5 runs)
    warm_times = []
    for _ in range(5):
        start = time.perf_counter()
        prob.solve(x0=x0)
        warm_times.append((time.perf_counter() - start) * 1000)
    warm_solve_ms = float(np.mean(warm_times))

    # SciPy baseline
    scipy_ms = time_scipy_constrained_nlp(n)

    return BenchmarkResult(n, build_ms, cold_solve_ms, warm_solve_ms, scipy_ms)


def benchmark_milp_loop(n: int, c: np.ndarray, capacity: int) -> BenchmarkResult:
    """Benchmark MILP with loop-based variables (full end-to-end).

    Single-constraint binary knapsack: max c'x s.t. sum(x) <= capacity, x in {0,1}.
    """
    # Time build phase
    start = time.perf_counter()
    x = np.array([BinaryVariable(f"x{i}") for i in range(n)])
    prob = Problem(name=f"milp_loop_{n}")
    prob.maximize(sum(float(c[i]) * x[i] for i in range(n)))
    prob.subject_to(np.sum(x) <= capacity)
    build_ms = (time.perf_counter() - start) * 1000

    # Time cold solve
    start = time.perf_counter()
    prob.solve()
    cold_solve_ms = (time.perf_counter() - start) * 1000

    # Time warm solves (5 runs)
    warm_times = []
    for _ in range(5):
        start = time.perf_counter()
        prob.solve()
        warm_times.append((time.perf_counter() - start) * 1000)
    warm_solve_ms = float(np.mean(warm_times))

    # SciPy baseline
    scipy_ms = time_scipy_milp(c, capacity)

    return BenchmarkResult(n, build_ms, cold_solve_ms, warm_solve_ms, scipy_ms)


def benchmark_milp_vector(n: int, c: np.ndarray, capacity: int) -> BenchmarkResult:
    """Benchmark MILP with VectorVariable (full end-to-end).

    Single-constraint binary knapsack: max c'x s.t. sum(x) <= capacity, x in {0,1}.
    Uses a single binary VectorVariable to leverage efficient vectorized ops.
    """
    # Time build phase
    start = time.perf_counter()
    x = VectorVariable("x", n, domain="binary")
    prob = Problem(name=f"milp_vec_{n}")
    prob.maximize(x @ c)
    prob.subject_to(x.sum() <= capacity)
    build_ms = (time.perf_counter() - start) * 1000

    # Time cold solve
    start = time.perf_counter()
    prob.solve()
    cold_solve_ms = (time.perf_counter() - start) * 1000

    # Time warm solves (5 runs)
    warm_times = []
    for _ in range(5):
        start = time.perf_counter()
        prob.solve()
        warm_times.append((time.perf_counter() - start) * 1000)
    warm_solve_ms = float(np.mean(warm_times))

    # SciPy baseline
    scipy_ms = time_scipy_milp(c, capacity)

    return BenchmarkResult(n, build_ms, cold_solve_ms, warm_solve_ms, scipy_ms)


def print_result(label: str, r: BenchmarkResult) -> None:
    """Print a benchmark result with full breakdown."""
    print(
        f"  {label:4s} n={r.n:5d}: Build={r.build_ms:8.1f}ms, "
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

    # Loop-based: limited due to superlinear cold solve time
    loop_sizes = [10, 25, 50, 100, 200, 500]
    # VectorVariable: scale to n=5000 (LP solver dominates at larger sizes)
    vec_sizes = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]

    loop_results = ScalingResults(label="LP (Loop)")
    vec_results = ScalingResults(label="LP (VectorVariable)")

    print(f"\n--- Loop-based Variable (n ≤ {max(loop_sizes)}, slow cold solve) ---")
    for n in loop_sizes:
        m = n // 2
        np.random.seed(42)
        c = np.random.rand(n)
        A = np.random.rand(m, n)
        b = np.sum(A, axis=1) * 0.5

        r = benchmark_lp_loop(n, c, A, b)
        loop_results.add(r)
        print_result("Loop", r)

    print(f"\n--- VectorVariable (n ≤ {max(vec_sizes):,}) ---")
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

    # Loop-based: limited to n=500 due to superlinear cold solve time
    loop_sizes = [10, 25, 50, 100, 200, 500]
    # VectorVariable with vectorized ops: scale to n=10,000
    vec_sizes = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]

    loop_results = ScalingResults(label="NLP (Loop)")
    vec_results = ScalingResults(label="NLP (VectorVariable)")

    print(f"\n--- Loop-based Variable (n ≤ {max(loop_sizes)}, slow cold solve) ---")
    for n in loop_sizes:
        r = benchmark_nlp_loop(n)
        loop_results.add(r)
        print_result("Loop", r)

    print(f"\n--- VectorVariable with x.dot(x) - x.sum() (n ≤ {max(vec_sizes):,}) ---")
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

    # Loop-based: limited due to superlinear cold solve time
    loop_sizes = [10, 25, 50, 100, 200, 500]
    # VectorVariable: scale to n=5000 (SLSQP solver is O(n²), dominates at larger sizes)
    vec_sizes = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]

    loop_results = ScalingResults(label="CQP (Loop)")
    vec_results = ScalingResults(label="CQP (VectorVariable)")

    print(f"\n--- Loop-based Variable (n ≤ {max(loop_sizes)}, slow cold solve) ---")
    for n in loop_sizes:
        r = benchmark_cqp_loop(n)
        loop_results.add(r)
        print_result("Loop", r)

    print(f"\n--- VectorVariable with x.dot(x), x.sum() (n ≤ {max(vec_sizes)}) ---")
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


def run_milp_scaling():
    """Run MILP scaling benchmarks."""
    print("\n" + "=" * 80)
    print("MILP SCALING BENCHMARK (End-to-End)")
    print("=" * 80)
    print("\nMeasures: Build (vars + problem + constraints) + Solve")
    print("Compared against: SciPy milp (no build phase)")
    print("Problem: Single-constraint binary knapsack (sum(x) <= n//2)")

    # Loop-based: push to n=500 to measure build overhead at scale
    loop_sizes = [10, 25, 50, 100, 200, 500]
    # VectorVariable: push to n=5000 to measure solve scaling
    vec_sizes = [10, 25, 50, 100, 200, 500, 1000, 2000, 5000]

    loop_results = ScalingResults(label="MILP (Loop)")
    vec_results = ScalingResults(label="MILP (VectorVariable)")

    print(f"\n--- Loop-based Variable (n ≤ {max(loop_sizes)}, slow cold solve) ---")
    for n in loop_sizes:
        np.random.seed(42)
        c = np.random.rand(n)
        capacity = n // 2  # Pick at most half the items

        r = benchmark_milp_loop(n, c, capacity)
        loop_results.add(r)
        print_result("Loop", r)

    print(f"\n--- VectorVariable (n ≤ {max(vec_sizes)}) ---")
    for n in vec_sizes:
        np.random.seed(42)
        c = np.random.rand(n)
        capacity = n // 2

        r = benchmark_milp_vector(n, c, capacity)
        vec_results.add(r)
        print_result("Vec", r)

    # Plot
    plot_scaling_comparison(
        loop_results,
        vec_results,
        title="MILP Scaling: End-to-End Time vs SciPy",
        save_path=RESULTS_DIR / "milp_scaling_comparison.png",
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
    ax2.set_title(
        "Overhead Ratio (log scale, lower is better)\nWarm = solve-only (excludes build)",
        fontsize=11,
    )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3, which="both")

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {save_path}")
    plt.close()


def run_overhead_summary(
    lp_results: ScalingResults | None = None,
    nlp_results: ScalingResults | None = None,
    cqp_results: ScalingResults | None = None,
    milp_results: ScalingResults | None = None,
):
    """Generate overhead summary from previously recorded scaling results.

    When pre-recorded results are provided the summary is derived directly
    from them so that the numbers are consistent with the detailed tables.
    Falls back to a fresh measurement only when a result set is missing.
    """
    print("\n" + "=" * 80)
    print("OVERHEAD SUMMARY BY PROBLEM TYPE")
    print("=" * 80)

    categories = []
    cold_overheads = []
    warm_overheads = []

    def _add(label: str, result: BenchmarkResult):
        categories.append(label)
        cold_overheads.append(result.cold_overhead)
        warm_overheads.append(result.warm_overhead)
        print(
            f"{label.replace(chr(10), ' ')}: "
            f"Cold={result.cold_overhead:.1f}x, Warm={result.warm_overhead:.1f}x"
        )

    def _find(results: ScalingResults | None, n: int) -> BenchmarkResult | None:
        if results is None:
            return None
        for r in results.results:
            if r.n == n:
                return r
        return None

    # --- LP ---
    for n in (50, 5000):
        r = _find(lp_results, n)
        if r is None:
            m = n // 2
            np.random.seed(42)
            c = np.random.rand(n)
            A = np.random.rand(m, n)
            b = np.sum(A, axis=1) * 0.5
            r = benchmark_lp_vector(n, c, A, b)
        _add(f"LP\nn={n}", r)

    # --- NLP ---
    for n in (50, 5000):
        r = _find(nlp_results, n)
        if r is None:
            r = benchmark_nlp_vector(n)
        _add(f"NLP\nn={n}", r)

    # --- CQP ---
    for n in (50, 5000):
        r = _find(cqp_results, n)
        if r is None:
            r = benchmark_cqp_vector(n)
        _add(f"CQP\nn={n}", r)

    # --- MILP ---
    for n in (50, 5000):
        r = _find(milp_results, n)
        if r is None:
            np.random.seed(42)
            c = np.random.rand(n)
            capacity = n // 2
            r = benchmark_milp_vector(n, c, capacity)
        _add(f"MILP\nn={n}", r)

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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = write_benchmark_metadata(
        RESULTS_DIR,
        extra={"benchmark_suite": "run_benchmarks"},
    )

    # Capture output to file
    output_path = RESULTS_DIR / "benchmark_output.txt"
    original_stdout = sys.stdout

    results_payload: dict[str, object] | None = None

    with open(output_path, "w") as log_file:
        sys.stdout = Tee(sys.stdout, log_file)  # type: ignore

        try:
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
            print(f"Terminal output being saved to: {output_path}")
            print(f"Benchmark metadata saved to: {metadata_path}")

            # Run scaling benchmarks (capture results for summary)
            lp_loop_results, lp_vec_results = run_lp_scaling()
            nlp_loop_results, nlp_vec_results = run_nlp_scaling()
            cqp_loop_results, cqp_vec_results = run_cqp_scaling()
            milp_loop_results, milp_vec_results = run_milp_scaling()

            results_payload = build_benchmark_payload(
                lp_loop_results=lp_loop_results,
                lp_vec_results=lp_vec_results,
                nlp_loop_results=nlp_loop_results,
                nlp_vec_results=nlp_vec_results,
                cqp_loop_results=cqp_loop_results,
                cqp_vec_results=cqp_vec_results,
                milp_loop_results=milp_loop_results,
                milp_vec_results=milp_vec_results,
            )
            benchmark_results_path = write_benchmark_results_json(results_payload)
            print(f"Structured benchmark results saved to: {benchmark_results_path}")

            # Run overhead summary from recorded data (no fresh recompute)
            run_overhead_summary(
                lp_results=lp_vec_results,
                nlp_results=nlp_vec_results,
                cqp_results=cqp_vec_results,
                milp_results=milp_vec_results,
            )

            print("\n" + "=" * 80)
            print("BENCHMARK COMPLETE")
            print("=" * 80)
            print(f"\nPlots saved to: {RESULTS_DIR}")
            for f in sorted(RESULTS_DIR.glob("*.png")):
                print(f"  - {f.name}")

        finally:
            sys.stdout = original_stdout

    if results_payload is not None:
        artifacts_to_sync = collect_results_artifacts()
        copied_files = sync_results_to_docs_assets(artifacts_to_sync)

        print(f"Synced benchmark artifacts to: {DOCS_BENCHMARKS_DIR}")
        for file_path in copied_files:
            print(f"  - {file_path.name}")


if __name__ == "__main__":
    main()
