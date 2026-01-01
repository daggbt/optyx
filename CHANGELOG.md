# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

<!-- Next release changes go here -->

## [1.2.0] - 2026-01-01

### Added
- **VectorVariable**: Create vectors of decision variables with `VectorVariable("x", 100, lb=0)`. Supports indexing, slicing, and iteration.
- **MatrixVariable**: Create matrices of decision variables with `MatrixVariable("A", 3, 4)`. Supports 2D indexing, row/column slicing, symmetric matrices, and transpose views.
- **Vector Operations**: `sum()`, `dot()`, `norm()` (L1 and L2), and `LinearCombination` for efficient vector computations.
- **Matrix Operations**: `quadratic_form(x, A)`, `trace()`, `diag()`, `diag_matrix()`, `frobenius_norm()`, and matrix-vector multiplication via `@` operator.
- **Parameter System**: `Parameter`, `VectorParameter`, and `MatrixParameter` for constants that can be updated between solves without rebuilding the problem structure.
- **Problem.summary()**: Human-readable overview of problem structure (variables, constraints, objective).
- **Native Gradient Rules**: O(1) gradient computation for vector expressions (`VectorSum`, `LinearCombination`, `DotProduct`), enabling NLP problems with n=10,000+ variables.
- **Comprehensive Documentation**: New tutorials for vectors, matrices, and parameters. Quartodoc-generated API reference (66 pages).
- **New Examples**: `portfolio-advanced.qmd` (Markowitz optimization with covariance), `resource-allocation.qmd` (large-scale 100+ variables).

### Changed
- Auto-solver now checks constraint degrees (not just objective) for more accurate method selection.
- Solver uses interior initial points (`lb + epsilon`) to avoid singularities at bounds.
- Constraint tolerance is now scaled: `atol + rtol * max(1, |c_val|)` for better numerical stability.
- Iterative tree depth estimation prevents stack overflow for deeply nested expressions.

### Fixed
- Runtime detection and warning for `inf`/`nan` values during optimization.
- LP solver message handling for unbounded cases.

### Performance
- NLP with n=1000 variables solves in <1 second.
- Native gradient rules provide O(1) coefficient lookup vs O(n) tree traversal.

## [1.1.1] - 2025-12-25

### Fixed
- **Version reporting**: `optyx.__version__` now correctly reads from package metadata instead of a hardcoded value. Previously, v1.1.0 incorrectly reported as "1.0.0".

### Added
- **New `eq()` method**: Added concise `Expression.eq()` method for equality constraints. `constraint_eq()` is retained for backwards compatibility.

## [1.1.0] - 2025-12-24

### Added
- **LP Fast Path**: Linear programs are now automatically detected and solved using `scipy.optimize.linprog` with the HiGHS solver, providing significant speedups over the general NLP path.
- **Automatic Solver Selection**: `Problem.solve(method="auto")` is now the default, automatically selecting the best solver:
  - Linear problems → `linprog` (HiGHS)
  - Unconstrained, n ≤ 3 → `Nelder-Mead`
  - Unconstrained, n > 1000 → `L-BFGS-B`
  - Unconstrained, else → `BFGS`
  - Bounds only → `L-BFGS-B`
  - Equality constraints → `trust-constr`
  - Inequality only → `SLSQP`
- **LP/QP Detection**: New `is_linear()` and `is_quadratic()` functions in `optyx.analysis` to classify expressions by polynomial degree.
- **LP Coefficient Extraction**: `LinearProgramExtractor` class to extract coefficient matrices from symbolic expressions for use with `linprog()`.
- **Caching for Performance**:
  - LP coefficient matrices are cached and reused for repeated solves of the same problem.
  - Linearity checks are cached per problem.
  - Expression degree is lazily computed and cached on first access.
- **Expression Methods**: Added `Expression.degree` property and `Expression.is_linear()` method for direct linearity checks.
- **Comprehensive Benchmark Suite**: New `benchmarks/` folder with validation, performance, accuracy, and comparison tests. Includes plot generation for performance analysis.
- **Numpy Vectorization Support**: Variables can be used with numpy arrays and `@` operator for matrix operations.

### Changed
- Default solver method changed from `"SLSQP"` to `"auto"` for optimal solver selection.
- Cache invalidation is now centralized via `Problem._invalidate_caches()`.

### Performance
- **LP Overhead**: ~0.94-1.15x vs raw SciPy (near parity)
- **NLP Overhead**: ~1.4-2.2x vs raw SciPy with gradients
- **Cache Speedup**: 2x-900x for repeated solves (larger problems benefit more)
- **Rosenbrock**: 0.83x - exact gradients help complex optimization landscapes

## [1.0.1] - 2025-12-19

### Added
- **Constant Gradient Detection**: Pre-compute Jacobian when all elements are constants, providing 9.7x speedup for linear expressions.
- **Solver Cache**: Cache compiled callables (objective, gradient, constraint functions) per Problem instance for reuse across `solve()` calls.

### Changed
- **Lazy Sanitization**: Skip `nan_to_num` when all derivative values are finite (3.2x speedup).
- SLSQP "positive directional derivative" message now correctly treated as optimal convergence.

### Performance
- Small LP: 1.66x → 1.54x overhead (-7% improvement)
- Large LP: 8.82x → 7.13x overhead (-19% improvement)

## [1.0.0] - 2025-12-15

### Added
- **Core Expression System**: Symbolic variables, constants, and operators (`+`, `-`, `*`, `/`, `**`).
- **Automatic Differentiation**: Symbolic computation of gradients, Jacobians, and Hessians.
- **Solver Interface**: Integration with `scipy.optimize.minimize` for solving constrained nonlinear problems.
- **Constraint Support**: Equality (`==`) and inequality (`<=`, `>=`) constraints.
- **Strict Mode**: Added `strict=True` parameter to `Problem.solve()` to prevent accidental relaxation of discrete variables (Issue #21).
- **Singularity Handling**: Robust handling of derivatives at singularities (e.g., $1/x$ at $x=0$) using `_sanitize_derivatives` (Issue #20).
- **Examples**: Added domain-specific demos for Mining Scheduling, Fleet Dispatch, and Portfolio Optimization.

### Fixed
- **Hessian Integration**: Fixed issue where Hessian was not being correctly passed to the solver (Issue #19).
- **Derivative Stability**: Fixed `NaN`/`Inf` issues in gradients at zero for functions like `sqrt` and `log`.

### Changed
- **Project Structure**: Renamed package from `optix` to `optyx`.
