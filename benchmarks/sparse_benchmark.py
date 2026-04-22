"""Sparse vs Dense Jacobian compilation benchmark.

Demonstrates the memory and performance benefit of compile_sparse_jacobian
vs compile_jacobian for problems with sparse constraint structure.

Generates comparison plots saved to benchmarks/results/.
"""

import time
import numpy as np
from optyx import Variable
from optyx.core.autodiff import compile_jacobian, compile_sparse_jacobian
from scipy import sparse
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def benchmark_chain_constraints(n_values: list[int]) -> dict:
    """Benchmark chain constraints: x_i + x_{i+1} <= 1.

    Each row of the Jacobian has exactly 2 non-zeros → density = 2/n.
    This is the classic pattern for sparse benefit.
    """
    print("=" * 70)
    print("CHAIN CONSTRAINTS: x_i + x_{i+1} (2 nnz per row, density=2/n)")
    print("=" * 70)
    print(
        f"{'n':>6} | {'Dense Compile':>14} | {'Sparse Compile':>15} | {'Dense Eval':>11} | {'Sparse Eval':>12} | {'Compile Speedup':>16} | {'Eval Speedup':>13} | {'Density':>8}"
    )
    print("-" * 120)

    data = {
        "n": [],
        "dense_compile": [],
        "sparse_compile": [],
        "dense_eval": [],
        "sparse_eval": [],
        "compile_speedup": [],
        "eval_speedup": [],
    }

    for n in n_values:
        variables = [Variable(f"x{i}") for i in range(n)]
        exprs = [variables[i] + variables[i + 1] for i in range(n - 1)]
        x = np.ones(n)

        # Dense compilation
        t0 = time.perf_counter()
        dense_fn = compile_jacobian(exprs, variables)
        t_dense_compile = time.perf_counter() - t0

        # Sparse compilation
        t0 = time.perf_counter()
        sparse_fn = compile_sparse_jacobian(exprs, variables)
        t_sparse_compile = time.perf_counter() - t0

        # Dense evaluation (average of 10 runs)
        for _ in range(3):
            dense_fn(x)
        t0 = time.perf_counter()
        for _ in range(10):
            dense_result = dense_fn(x)
        t_dense_eval = (time.perf_counter() - t0) / 10

        # Sparse evaluation (average of 10 runs)
        for _ in range(3):
            sparse_fn(x)
        t0 = time.perf_counter()
        for _ in range(10):
            sparse_result = sparse_fn(x)
        t_sparse_eval = (time.perf_counter() - t0) / 10

        # Verify correctness
        if sparse.issparse(sparse_result):
            np.testing.assert_array_almost_equal(sparse_result.toarray(), dense_result)

        density = 2.0 / n
        compile_speedup = (
            t_dense_compile / t_sparse_compile if t_sparse_compile > 0 else float("inf")
        )
        eval_speedup = (
            t_dense_eval / t_sparse_eval if t_sparse_eval > 0 else float("inf")
        )

        data["n"].append(n)
        data["dense_compile"].append(t_dense_compile * 1000)
        data["sparse_compile"].append(t_sparse_compile * 1000)
        data["dense_eval"].append(t_dense_eval * 1000)
        data["sparse_eval"].append(t_sparse_eval * 1000)
        data["compile_speedup"].append(compile_speedup)
        data["eval_speedup"].append(eval_speedup)

        print(
            f"  {n:>4} | {t_dense_compile * 1000:>12.2f}ms | {t_sparse_compile * 1000:>13.2f}ms | "
            f"{t_dense_eval * 1000:>9.3f}ms | {t_sparse_eval * 1000:>10.3f}ms | "
            f"{compile_speedup:>14.1f}x | {eval_speedup:>11.1f}x | {density:>7.1%}"
        )

    return data


def benchmark_pairwise_constraints(n_values: list[int]) -> dict:
    """Benchmark pairwise constraints on random pairs.

    Each constraint involves exactly 2 of n variables.
    Number of constraints = n (so Jacobian is n × n with ~2n non-zeros).
    """
    print()
    print("=" * 70)
    print("PAIRWISE CONSTRAINTS: x_i - x_j (2 nnz per row, m=n)")
    print("=" * 70)
    print(
        f"{'n':>6} | {'Dense Eval':>11} | {'Sparse Eval':>12} | {'Eval Speedup':>13} | {'Dense Memory':>13} | {'Sparse Memory':>14}"
    )
    print("-" * 90)

    data = {
        "n": [],
        "dense_eval": [],
        "sparse_eval": [],
        "eval_speedup": [],
        "dense_mem": [],
        "sparse_mem": [],
    }

    for n in n_values:
        variables = [Variable(f"x{i}") for i in range(n)]
        rng = np.random.RandomState(42)
        pairs = [(rng.randint(0, n), rng.randint(0, n)) for _ in range(n)]
        exprs = [variables[i] - variables[j] for i, j in pairs if i != j]
        x = np.ones(n)

        dense_fn = compile_jacobian(exprs, variables)
        sparse_fn = compile_sparse_jacobian(exprs, variables)

        # Warmup
        for _ in range(3):
            dense_fn(x)
            sparse_fn(x)

        # Eval
        t0 = time.perf_counter()
        for _ in range(10):
            dense_result = dense_fn(x)
        t_dense = (time.perf_counter() - t0) / 10

        t0 = time.perf_counter()
        for _ in range(10):
            sparse_result = sparse_fn(x)
        t_sparse = (time.perf_counter() - t0) / 10

        dense_mem = dense_result.nbytes
        if sparse.issparse(sparse_result):
            sparse_mem = (
                sparse_result.data.nbytes
                + sparse_result.indices.nbytes
                + sparse_result.indptr.nbytes
            )
        else:
            sparse_mem = sparse_result.nbytes

        speedup = t_dense / t_sparse if t_sparse > 0 else float("inf")

        data["n"].append(n)
        data["dense_eval"].append(t_dense * 1000)
        data["sparse_eval"].append(t_sparse * 1000)
        data["eval_speedup"].append(speedup)
        data["dense_mem"].append(dense_mem)
        data["sparse_mem"].append(sparse_mem)

        print(
            f"  {n:>4} | {t_dense * 1000:>9.3f}ms | {t_sparse * 1000:>10.3f}ms | "
            f"{speedup:>11.1f}x | {dense_mem:>10,} B | {sparse_mem:>11,} B"
        )

    return data


def benchmark_nonlinear_sparse(n_values: list[int]) -> dict:
    """Benchmark non-linear sparse constraints: x_i^2 + x_{i+1}^2.

    Each row has 2 non-zeros but gradient computation is non-trivial.
    """
    print()
    print("=" * 70)
    print("NONLINEAR SPARSE: x_i² + x_{i+1}² (2 nnz per row)")
    print("=" * 70)
    print(f"{'n':>6} | {'Dense Eval':>11} | {'Sparse Eval':>12} | {'Eval Speedup':>13}")
    print("-" * 60)

    data = {
        "n": [],
        "dense_eval": [],
        "sparse_eval": [],
        "eval_speedup": [],
    }

    for n in n_values:
        variables = [Variable(f"x{i}") for i in range(n)]
        exprs = [variables[i] ** 2 + variables[i + 1] ** 2 for i in range(n - 1)]
        x = np.ones(n)

        dense_fn = compile_jacobian(exprs, variables)
        sparse_fn = compile_sparse_jacobian(exprs, variables)

        # Warmup
        for _ in range(3):
            dense_fn(x)
            sparse_fn(x)

        # Eval
        t0 = time.perf_counter()
        for _ in range(10):
            dense_fn(x)
        t_dense = (time.perf_counter() - t0) / 10

        t0 = time.perf_counter()
        for _ in range(10):
            sparse_fn(x)
        t_sparse = (time.perf_counter() - t0) / 10

        speedup = t_dense / t_sparse if t_sparse > 0 else float("inf")

        data["n"].append(n)
        data["dense_eval"].append(t_dense * 1000)
        data["sparse_eval"].append(t_sparse * 1000)
        data["eval_speedup"].append(speedup)

        print(
            f"  {n:>4} | {t_dense * 1000:>9.3f}ms | {t_sparse * 1000:>10.3f}ms | "
            f"{speedup:>11.1f}x"
        )

    return data


def plot_results(chain: dict, pairwise: dict, nonlinear: dict) -> None:
    """Generate comparison plots and save to benchmarks/results/."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Sparse vs Dense Jacobian: Performance Comparison", fontsize=16, y=0.98)

    # --- Row 1: Absolute timings ---

    # 1a. Chain — Compile times
    ax = axes[0, 0]
    ax.plot(chain["n"], chain["dense_compile"], "o-", color="#d62728", label="Dense", linewidth=2)
    ax.plot(chain["n"], chain["sparse_compile"], "s-", color="#2ca02c", label="Sparse", linewidth=2)
    ax.set_xlabel("Problem size (n variables)")
    ax.set_ylabel("Compile time (ms)")
    ax.set_title("Chain: Compilation Time")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # 1b. Chain — Eval times
    ax = axes[0, 1]
    ax.plot(chain["n"], chain["dense_eval"], "o-", color="#d62728", label="Dense", linewidth=2)
    ax.plot(chain["n"], chain["sparse_eval"], "s-", color="#2ca02c", label="Sparse", linewidth=2)
    ax.set_xlabel("Problem size (n variables)")
    ax.set_ylabel("Evaluation time (ms)")
    ax.set_title("Chain: Evaluation Time")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # 1c. Pairwise — Memory
    ax = axes[0, 2]
    dense_kb = [m / 1024 for m in pairwise["dense_mem"]]
    sparse_kb = [m / 1024 for m in pairwise["sparse_mem"]]
    ax.plot(pairwise["n"], dense_kb, "o-", color="#d62728", label="Dense", linewidth=2)
    ax.plot(pairwise["n"], sparse_kb, "s-", color="#2ca02c", label="Sparse", linewidth=2)
    ax.set_xlabel("Problem size (n variables)")
    ax.set_ylabel("Jacobian memory (KB)")
    ax.set_title("Pairwise: Memory Usage")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # --- Row 2: Speedups ---

    # 2a. Chain — Speedups
    ax = axes[1, 0]
    ax.bar(
        [x - 0.15 for x in range(len(chain["n"]))],
        chain["compile_speedup"],
        width=0.3,
        color="#1f77b4",
        label="Compile speedup",
    )
    ax.bar(
        [x + 0.15 for x in range(len(chain["n"]))],
        chain["eval_speedup"],
        width=0.3,
        color="#ff7f0e",
        label="Eval speedup",
    )
    ax.set_xticks(range(len(chain["n"])))
    ax.set_xticklabels([str(n) for n in chain["n"]])
    ax.set_xlabel("Problem size (n)")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Chain: Sparse Speedup Factor")
    ax.legend()
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # 2b. Pairwise — Eval speedup
    ax = axes[1, 1]
    ax.bar(
        range(len(pairwise["n"])),
        pairwise["eval_speedup"],
        color="#9467bd",
        label="Eval speedup",
    )
    ax.set_xticks(range(len(pairwise["n"])))
    ax.set_xticklabels([str(n) for n in pairwise["n"]])
    ax.set_xlabel("Problem size (n)")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Pairwise: Sparse Eval Speedup")
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    # 2c. Nonlinear — Eval speedup
    ax = axes[1, 2]
    ax.plot(nonlinear["n"], nonlinear["dense_eval"], "o-", color="#d62728", label="Dense", linewidth=2)
    ax.plot(nonlinear["n"], nonlinear["sparse_eval"], "s-", color="#2ca02c", label="Sparse", linewidth=2)
    ax.set_xlabel("Problem size (n variables)")
    ax.set_ylabel("Evaluation time (ms)")
    ax.set_title("Nonlinear: x²+x² Evaluation Time")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "sparse_vs_dense_comparison.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to: {plot_path}")

    # --- Memory & speedup plot (2 subplots) ---
    fig2, (ax_mem, ax_spd) = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("Pairwise Constraints: Sparse vs Dense", fontsize=14, y=1.02)

    # Left: actual memory usage
    x_pos = np.arange(len(pairwise["n"]))
    bar_w = 0.35
    dense_kb = [m / 1024 for m in pairwise["dense_mem"]]
    sparse_kb = [m / 1024 for m in pairwise["sparse_mem"]]
    ax_mem.bar(x_pos - bar_w / 2, dense_kb, bar_w, color="#d62728", label="Dense")
    ax_mem.bar(x_pos + bar_w / 2, sparse_kb, bar_w, color="#2ca02c", label="Sparse")
    ax_mem.set_xticks(x_pos)
    ax_mem.set_xticklabels([str(n) for n in pairwise["n"]])
    ax_mem.set_xlabel("Problem size (n variables)")
    ax_mem.set_ylabel("Jacobian memory (KB)")
    ax_mem.set_title("Memory Usage")
    ax_mem.set_yscale("log")
    ax_mem.legend()
    ax_mem.grid(True, alpha=0.3, axis="y")

    # Right: memory reduction factor
    mem_reduction = [d / s if s > 0 else 1 for d, s in zip(pairwise["dense_mem"], pairwise["sparse_mem"])]
    ax_spd.bar(x_pos, mem_reduction, color="#2ca02c", edgecolor="#1a7a1a")
    ax_spd.set_xticks(x_pos)
    ax_spd.set_xticklabels([str(n) for n in pairwise["n"]])
    ax_spd.set_xlabel("Problem size (n variables)")
    ax_spd.set_ylabel("Memory reduction (×)")
    ax_spd.set_title("Sparse Memory Savings (Dense / Sparse)")
    ax_spd.axhline(y=1, color="gray", linestyle="--", alpha=0.5)
    ax_spd.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(mem_reduction):
        ax_spd.text(i, v + 0.3, f"{v:.0f}×", ha="center", fontweight="bold", fontsize=10)

    plt.tight_layout()
    mem_path = RESULTS_DIR / "sparse_memory_reduction.png"
    fig2.savefig(mem_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Plot saved to: {mem_path}")


if __name__ == "__main__":
    print("Sparse vs Dense Jacobian Compilation Benchmark")
    print("=" * 70)
    print()

    chain_sizes = [10, 25, 50, 100, 200, 500]
    pairwise_sizes = [10, 25, 50, 100, 200, 500]
    nonlinear_sizes = [10, 25, 50, 100, 200]

    chain_data = benchmark_chain_constraints(chain_sizes)
    pairwise_data = benchmark_pairwise_constraints(pairwise_sizes)
    nonlinear_data = benchmark_nonlinear_sparse(nonlinear_sizes)

    print()
    print("=" * 70)
    print("Generating comparison plots...")
    plot_results(chain_data, pairwise_data, nonlinear_data)
