"""Performance benchmark: Repeated solve timing.

Measures the benefit of caching when solving the same problem multiple times.
Target: > 3x speedup on repeated solves.
"""

from __future__ import annotations

import time
import numpy as np

from optyx import Variable, Problem

import sys

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from utils import time_function


class TestLPResolveTiming:
    """LP resolve timing with caching."""

    def test_lp_cache_benefit(self):
        """Measure cache benefit on LP repeated solves.

        Target: > 3x speedup on cached solves.
        """
        n, m = 20, 15
        np.random.seed(42)

        c = np.random.rand(n)
        A = np.random.rand(m, n)
        b = np.sum(A, axis=1) * 0.5

        variables = [Variable(f"x{i}", lb=0, ub=1) for i in range(n)]

        prob = Problem(name="lp_cache_test")
        prob.maximize(sum(c[i] * variables[i] for i in range(n)))
        for i in range(m):
            prob.subject_to(sum(A[i, j] * variables[j] for j in range(n)) <= b[i])

        # First solve (cache miss - extracts LP data)
        start = time.perf_counter()
        prob.solve()
        first_solve_ms = (time.perf_counter() - start) * 1000

        # Subsequent solves (cache hit)
        times_cached = []
        for _ in range(20):
            start = time.perf_counter()
            sol = prob.solve()
            times_cached.append((time.perf_counter() - start) * 1000)

        mean_cached = np.mean(times_cached)
        std_cached = np.std(times_cached)

        speedup = first_solve_ms / mean_cached if mean_cached > 0 else float("inf")

        print("\nLP Cache Benefit:")
        print(f"  First solve: {first_solve_ms:.3f} ms")
        print(f"  Cached mean: {mean_cached:.3f} ± {std_cached:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")

        assert sol.is_optimal
        assert speedup > 2.0, f"Cache speedup too low: {speedup:.2f}x"

    def test_lp_vs_nlp_resolve(self):
        """Compare LP resolve vs NLP resolve timing."""
        n = 20
        np.random.seed(42)

        c = np.random.rand(n)

        # LP problem
        lp_vars = [Variable(f"x{i}", lb=0, ub=1) for i in range(n)]
        lp_prob = Problem(name="lp_resolve")
        lp_prob.maximize(sum(c[i] * lp_vars[i] for i in range(n)))
        lp_prob.subject_to(sum(lp_vars) <= n * 0.5)

        # NLP problem (same structure but with quadratic term)
        nlp_vars = [Variable(f"y{i}", lb=0, ub=1) for i in range(n)]
        nlp_prob = Problem(name="nlp_resolve")
        nlp_prob.minimize(sum(c[i] * nlp_vars[i] ** 2 for i in range(n)))
        nlp_prob.subject_to(sum(nlp_vars) <= n * 0.5)

        # Warm up both
        lp_prob.solve()
        x0_nlp = np.array([0.5] * n)
        nlp_prob.solve(x0=x0_nlp)

        # Time repeated solves
        lp_timing = time_function(lambda: lp_prob.solve(), n_warmup=0, n_runs=20)
        nlp_timing = time_function(
            lambda: nlp_prob.solve(x0=x0_nlp), n_warmup=0, n_runs=20
        )

        print("\nLP vs NLP Resolve:")
        print(f"  LP:  {lp_timing}")
        print(f"  NLP: {nlp_timing}")
        print(f"  LP advantage: {nlp_timing.mean_ms / lp_timing.mean_ms:.2f}x faster")

        # LP should be faster than NLP due to caching
        assert lp_timing.mean_ms < nlp_timing.mean_ms


class TestNLPResolveTiming:
    """NLP resolve timing with compiled function caching."""

    def test_nlp_compiled_cache(self):
        """Measure benefit of compiled function caching on NLP."""
        x = Variable("x")
        y = Variable("y")

        prob = Problem(name="nlp_cache_test")
        prob.minimize((1 - x) ** 2 + 100 * (y - x**2) ** 2)

        x0 = np.array([-1.0, -1.0])

        # First solve (compiles functions)
        start = time.perf_counter()
        prob.solve(x0=x0)
        first_solve_ms = (time.perf_counter() - start) * 1000

        # Subsequent solves (uses cached compiled functions)
        times_cached = []
        for _ in range(20):
            start = time.perf_counter()
            sol = prob.solve(x0=x0)
            times_cached.append((time.perf_counter() - start) * 1000)

        mean_cached = np.mean(times_cached)
        std_cached = np.std(times_cached)

        speedup = first_solve_ms / mean_cached if mean_cached > 0 else float("inf")

        print("\nNLP Compiled Cache Benefit:")
        print(f"  First solve: {first_solve_ms:.3f} ms")
        print(f"  Cached mean: {mean_cached:.3f} ± {std_cached:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x")

        assert sol.is_optimal

    def test_constrained_nlp_resolve(self):
        """Measure constrained NLP resolve timing."""
        x = Variable("x")
        y = Variable("y")

        prob = Problem(name="constrained_nlp_resolve")
        prob.minimize(x**2 + y**2)
        prob.subject_to(x + y >= 1)
        prob.subject_to(x >= 0)
        prob.subject_to(y >= 0)

        x0 = np.array([0.5, 0.5])

        # Warm up
        prob.solve(x0=x0)

        timing = time_function(lambda: prob.solve(x0=x0), n_warmup=0, n_runs=20)

        print(f"\nConstrained NLP Resolve: {timing}")


class TestCacheInvalidation:
    """Test that cache invalidation works correctly."""

    def test_modify_objective_invalidates_cache(self):
        """Modifying objective should invalidate LP cache."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem(name="cache_invalidation")
        prob.maximize(3 * x + 2 * y)
        prob.subject_to(x + y <= 10)

        # First solve
        sol1 = prob.solve()

        # Modify objective
        prob.maximize(2 * x + 3 * y)  # Swap coefficients

        # Solve again
        sol2 = prob.solve()

        # Should get different results
        assert sol1.is_optimal
        assert sol2.is_optimal
        assert sol1["x"] != sol2["x"] or sol1["y"] != sol2["y"]

    def test_add_constraint_invalidates_cache(self):
        """Adding constraint should invalidate LP cache."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem(name="cache_invalidation_constraint")
        prob.maximize(3 * x + 2 * y)
        prob.subject_to(x + y <= 10)

        # First solve
        sol1 = prob.solve()

        # Add constraint
        prob.subject_to(x <= 3)

        # Solve again
        sol2 = prob.solve()

        assert sol1.is_optimal
        assert sol2.is_optimal
        # New constraint should limit x
        assert sol2["x"] <= 3.0 + 1e-6


if __name__ == "__main__":
    print("=" * 60)
    print("RESOLVE TIMING ANALYSIS")
    print("=" * 60)

    test_lp = TestLPResolveTiming()
    test_lp.test_lp_cache_benefit()
    test_lp.test_lp_vs_nlp_resolve()

    test_nlp = TestNLPResolveTiming()
    test_nlp.test_nlp_compiled_cache()
    test_nlp.test_constrained_nlp_resolve()

    test_cache = TestCacheInvalidation()
    test_cache.test_modify_objective_invalidates_cache()
    test_cache.test_add_constraint_invalidates_cache()
