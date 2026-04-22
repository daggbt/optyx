"""Performance benchmark: end-to-end sparse solve path.

This benchmark exercises the sparse trust-constr Jacobian path through the
solver runtime on a sparse chain-constrained quadratic problem.

It compares two solver paths on the same sparse NLP model:
- trust-constr: uses Optyx's vector-valued sparse Jacobian path
- SLSQP: uses the scalar constraint callback path

This is a solver-path reference benchmark, not a strict apples-to-apples
algorithm comparison.

Usage:
    uv run python benchmarks/performance/sparse_solve_end_to_end.py

Outputs:
    - sparse_solve_end_to_end.png
    - sparse_solve_end_to_end_results.json
    - sparse_solve_end_to_end_metadata.json
"""

from __future__ import annotations

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse

from optyx import Problem, Variable
from optyx.solvers.scipy_solver import _build_solver_cache

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from utils import RESULTS_DIR, time_function, write_benchmark_metadata


def build_sparse_chain_problem(n: int) -> tuple[Problem, np.ndarray]:
    """Build a sparse nonlinear chain-constrained problem."""
    variables = [Variable(f"x{i}", lb=0.0, ub=2.0) for i in range(n)]

    prob = Problem(name=f"sparse_chain_{n}")
    prob.minimize(sum((var - 1.0) ** 2 for var in variables))

    for i in range(n - 1):
        prob.subject_to(variables[i + 1] - variables[i] >= 0)
    prob.subject_to(variables[0] <= 0.5)
    prob.subject_to(variables[-1].eq(1.0))

    x0 = np.linspace(0.2, 1.0, n)
    return prob, x0


def inspect_sparse_jacobian(n: int) -> tuple[str, int, float]:
    """Inspect the cached sparse Jacobian format for a given sparse problem."""
    prob, x0 = build_sparse_chain_problem(n)
    cache = _build_solver_cache(prob, prob.variables)
    jac = cache["sparse_constraint_jac_fn"](x0)

    if sparse.issparse(jac):
        nnz = int(jac.nnz)
        density = (
            nnz / (jac.shape[0] * jac.shape[1])
            if jac.shape[0] and jac.shape[1]
            else 0.0
        )
        return type(jac).__name__, nnz, density

    nnz = int(np.count_nonzero(jac))
    density = nnz / jac.size if jac.size else 0.0
    return type(jac).__name__, nnz, density


def measure_cold_end_to_end(method: str, n: int, n_runs: int = 3):
    """Measure cold build+solve timing for a solver path."""

    def build_and_solve():
        prob, x0 = build_sparse_chain_problem(n)
        sol = prob.solve(method=method, x0=x0, warm_start=False)
        assert sol.is_optimal
        return sol

    return time_function(build_and_solve, n_warmup=0, n_runs=n_runs)


def measure_warm_cached(method: str, n: int, n_runs: int = 5):
    """Measure cached solve timing for a solver path."""
    prob, x0 = build_sparse_chain_problem(n)
    sol = prob.solve(method=method, x0=x0, warm_start=False)
    assert sol.is_optimal

    return time_function(
        lambda p=prob, x0=x0: p.solve(method=method, x0=x0, warm_start=False),
        n_warmup=1,
        n_runs=n_runs,
    )


def main() -> None:
    """Run the sparse solver-path benchmark."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metadata_path = write_benchmark_metadata(
        RESULTS_DIR,
        file_name="sparse_solve_end_to_end_metadata.json",
        extra={"benchmark_suite": "sparse_solve_end_to_end"},
    )

    sizes = [20, 40, 80, 120]
    results: list[dict[str, float | int | str]] = []

    trust_cold = []
    trust_warm = []
    slsqp_cold = []
    slsqp_warm = []

    print("=" * 72)
    print("SPARSE SOLVE END-TO-END BENCHMARK")
    print("=" * 72)
    print("trust-constr uses the sparse batched Jacobian path.")
    print("SLSQP uses the scalar constraint callback path.")
    print(f"Metadata saved to: {metadata_path}")
    print()
    print(
        f"{'n':>6} | {'Jac':>10} | {'nnz':>8} | {'Cold TC':>10} | {'Cold SLSQP':>11} | {'Warm TC':>10} | {'Warm SLSQP':>11}"
    )
    print("-" * 72)

    for n in sizes:
        jac_type, nnz, density = inspect_sparse_jacobian(n)
        cold_tc = measure_cold_end_to_end("trust-constr", n)
        cold_slsqp = measure_cold_end_to_end("SLSQP", n)
        warm_tc = measure_warm_cached("trust-constr", n)
        warm_slsqp = measure_warm_cached("SLSQP", n)

        trust_cold.append(cold_tc.median_ms)
        trust_warm.append(warm_tc.median_ms)
        slsqp_cold.append(cold_slsqp.median_ms)
        slsqp_warm.append(warm_slsqp.median_ms)

        results.append(
            {
                "n": n,
                "jacobian_type": jac_type,
                "nnz": nnz,
                "density": density,
                "cold_trust_constr_median_ms": cold_tc.median_ms,
                "cold_slsqp_median_ms": cold_slsqp.median_ms,
                "warm_trust_constr_median_ms": warm_tc.median_ms,
                "warm_slsqp_median_ms": warm_slsqp.median_ms,
            }
        )

        print(
            f"{n:6d} | {jac_type:>10} | {nnz:8d} | {cold_tc.median_ms:10.1f} | "
            f"{cold_slsqp.median_ms:11.1f} | {warm_tc.median_ms:10.1f} | {warm_slsqp.median_ms:11.1f}"
        )

    print("-" * 72)

    output_path = RESULTS_DIR / "sparse_solve_end_to_end_results.json"
    output_path.write_text(json.dumps(results, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(sizes, trust_cold, marker="o", label="trust-constr (sparse)")
    axes[0].plot(sizes, slsqp_cold, marker="o", label="SLSQP (scalar constraints)")
    axes[0].set_title("Cold End-to-End Build + Solve")
    axes[0].set_xlabel("Variables")
    axes[0].set_ylabel("Median time (ms)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(sizes, trust_warm, marker="o", label="trust-constr (sparse)")
    axes[1].plot(sizes, slsqp_warm, marker="o", label="SLSQP (scalar constraints)")
    axes[1].set_title("Warm Cached Solve-Only")
    axes[1].set_xlabel("Variables")
    axes[1].set_ylabel("Median time (ms)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Sparse Chain NLP: Solver-Path Reference Benchmark", fontsize=12)
    plt.tight_layout()
    plot_path = RESULTS_DIR / "sparse_solve_end_to_end.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Results saved to: {output_path}")
    print(f"Plot saved to: {plot_path}")


if __name__ == "__main__":
    main()
