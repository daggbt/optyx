"""Sparse vs Dense Jacobian compilation benchmark.

Demonstrates the memory and performance benefit of compile_sparse_jacobian
vs compile_jacobian for problems with sparse constraint structure.
"""

import time
import numpy as np
from optyx import Variable
from optyx.core.autodiff import compile_jacobian, compile_sparse_jacobian
from scipy import sparse


def benchmark_chain_constraints(n_values: list[int]) -> None:
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

        print(
            f"  {n:>4} | {t_dense_compile * 1000:>12.2f}ms | {t_sparse_compile * 1000:>13.2f}ms | "
            f"{t_dense_eval * 1000:>9.3f}ms | {t_sparse_eval * 1000:>10.3f}ms | "
            f"{compile_speedup:>14.1f}x | {eval_speedup:>11.1f}x | {density:>7.1%}"
        )


def benchmark_pairwise_constraints(n_values: list[int]) -> None:
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

        print(
            f"  {n:>4} | {t_dense * 1000:>9.3f}ms | {t_sparse * 1000:>10.3f}ms | "
            f"{speedup:>11.1f}x | {dense_mem:>10,} B | {sparse_mem:>11,} B"
        )


def benchmark_nonlinear_sparse(n_values: list[int]) -> None:
    """Benchmark non-linear sparse constraints: x_i^2 + x_{i+1}^2.

    Each row has 2 non-zeros but gradient computation is non-trivial.
    """
    print()
    print("=" * 70)
    print("NONLINEAR SPARSE: x_i² + x_{i+1}² (2 nnz per row)")
    print("=" * 70)
    print(f"{'n':>6} | {'Dense Eval':>11} | {'Sparse Eval':>12} | {'Eval Speedup':>13}")
    print("-" * 60)

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

        print(
            f"  {n:>4} | {t_dense * 1000:>9.3f}ms | {t_sparse * 1000:>10.3f}ms | "
            f"{speedup:>11.1f}x"
        )


if __name__ == "__main__":
    print("Sparse vs Dense Jacobian Compilation Benchmark")
    print("=" * 70)
    print()

    chain_sizes = [10, 25, 50, 100, 200, 500]
    pairwise_sizes = [10, 25, 50, 100, 200, 500]
    nonlinear_sizes = [10, 25, 50, 100, 200]

    benchmark_chain_constraints(chain_sizes)
    benchmark_pairwise_constraints(pairwise_sizes)
    benchmark_nonlinear_sparse(nonlinear_sizes)
