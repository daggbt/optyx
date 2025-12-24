# Optyx Benchmarks

Benchmark suite for validating correctness, performance, and accuracy of Optyx.
Uses **numpy vectorization** (`@`, `np.sum`, `np.array`) for optimal performance.

## Quick Start

```bash
# Run all benchmarks (tests only)
uv run pytest benchmarks/ -v

# Run benchmarks and generate plots
uv run python benchmarks/run_benchmarks.py

# Run specific category
uv run pytest benchmarks/validation/ -v
uv run pytest benchmarks/performance/ -v
uv run pytest benchmarks/accuracy/ -v
uv run pytest benchmarks/comparison/ -v
```

## Test Summary

| Category    | Tests | Passed | Skipped | Notes                          |
| ----------- | ----- | ------ | ------- | ------------------------------ |
| Validation  | 17    | 17     | 0       | Standard & constrained problems |
| Performance | 37    | 37     | 0       | Overhead, scaling, caching     |
| Accuracy    | 18    | 18     | 0       | Gradients & numerical stability |
| Comparison  | 15    | 15     | 9       | CVXPY/Pyomo require `--extra`  |
| **Total**   | **96**| **87** | **9**   |                                |

## Generated Plots

After running `uv run python benchmarks/run_benchmarks.py`, the following plots are generated in `benchmarks/results/`:

| Plot | Description |
|------|-------------|
| `lp_scaling_comparison.png` | LP solve time: Optyx vs SciPy across problem sizes |
| `nlp_quadratic_scaling.png` | NLP solve time: Optyx vs SciPy for quadratic problems |
| `lp_cache_benefit.png` | Cache speedup: cold vs warm solve times |
| `overhead_breakdown.png` | Overhead ratio by problem type |
| `multi_problem_scaling.png` | Scaling comparison across LP, NLP, and constrained QP |

## Structure

```
benchmarks/
├── run_benchmarks.py    # Main script to generate all plots
├── utils.py             # Timing, comparison, and plotting utilities
├── conftest.py          # Shared pytest fixtures
├── validation/          # Correctness tests against known optima
├── performance/         # Timing and overhead analysis
├── accuracy/            # Numerical precision tests
├── comparison/          # Comparison with other libraries
└── results/             # Generated plots (*.png)
```

## Categories

### Validation (`validation/`)

Tests that Optyx correctly solves standard optimization problems:

- **standard_problems.py**: Unconstrained problems (Rosenbrock, Sphere, Beale, Booth, Matyas)
- **constrained_problems.py**: Constrained problems (LP, QP, HS071, HS076, mixed constraints)

### Performance (`performance/`)

Measures overhead and scaling with plot generation:

- **overhead_analysis.py**: Optyx vs raw SciPy timing
- **scaling_analysis.py**: Performance vs problem size (10 to 500 vars) with plots
- **resolve_timing.py**: Repeated solve with caching benefits

### Accuracy (`accuracy/`)

Validates numerical correctness:

- **gradient_validation.py**: Autodiff vs finite difference
- **numerical_stability.py**: Edge cases (large/small coefficients, boundaries)

### Comparison (`comparison/`)

Compares Optyx with other optimization libraries:

- **bench_vs_scipy.py**: Always available - includes scaling plots
- **bench_vs_cvxpy.py**: Requires `cvxpy` (optional)
- **bench_vs_pyomo.py**: Requires `pyomo` (optional)

Install optional dependencies:

```bash
uv sync --extra benchmarks
```

#### CVXPY Comparison Results

| Problem | Optyx | CVXPY | Overhead | Notes |
|---------|-------|-------|----------|-------|
| Small LP (2 vars) | 1.2ms | 1.1ms | 1.12x | Near parity |
| Medium LP (20 vars) | 1.5ms | 1.5ms | 1.04x | Near parity |
| Simple QP | 0.5ms | 1.2ms | 0.41x | Optyx faster |
| Portfolio QP (10 assets) | 598ms | 2ms | **288x** | Quadratic form limitation |

**Key insight**: Optyx is competitive with CVXPY for LP and simple QP, but CVXPY's specialized `quad_form` is much faster for dense quadratic programs. Use CVXPY for convex QP with dense covariance matrices; use Optyx for NLP where autodiff provides value.

## Vectorization

Benchmarks use numpy vectorization for clean, performant code:

```python
# Create variables as numpy array
x = np.array([Variable(f"x{i}", lb=0, ub=1) for i in range(n)])

# Use @ for matrix operations
prob.maximize(c @ x)              # c^T @ x dot product
prob.subject_to(A[i] @ x <= b[i]) # Row-wise constraint

# Use np.sum for aggregation
prob.minimize(np.sum(x**2))       # ||x||²
prob.subject_to(np.sum(x) >= 1)   # Sum constraint
```

## Success Criteria

| Category       | Target                                    | Status |
| -------------- | ----------------------------------------- | ------ |
| Validation     | All problems converge to known optima     | ✅     |
| LP overhead    | < 1.5x vs SciPy linprog                   | ✅ ~0.94-1.15x |
| NLP overhead   | < 3x vs raw SciPy (with gradients)        | ✅ ~1.4-2.2x |
| Cache benefit  | > 2x speedup on repeated solve            | ✅ 2x-900x |
| Gradient error | < 1e-5 vs finite difference               | ✅ < 1e-10 |

## Adding New Benchmarks

1. Create a new `.py` file in the appropriate category folder
2. Use vectorized numpy operations (`@`, `np.sum`, `np.array`)
3. Follow the existing test patterns (pytest-compatible)
4. Use utilities from `utils.py` for timing, comparison, and plotting
5. **Important**: When comparing with SciPy, provide explicit gradients to SciPy for fair comparison

## Updating Documentation

When regenerating benchmark plots, copy them to the docs folder:

```bash
# Regenerate plots
uv run python benchmarks/run_benchmarks.py

# Copy to docs assets
cp benchmarks/results/*.png docs/assets/benchmarks/
```
