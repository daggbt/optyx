# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **LP Fast Path**: Linear programs are now automatically detected and solved using `scipy.optimize.linprog` with the HiGHS solver, providing 3-5x speedups over the general NLP path.
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

### Changed
- Default solver method changed from `"SLSQP"` to `"auto"` for optimal solver selection.
- Cache invalidation is now centralized via `Problem._invalidate_caches()`.

### Performance
- Repeated LP solves are 3.5x faster due to coefficient caching (4.15ms → 1.17ms).
- Large LP problems are 5x faster with the linprog fast path compared to SLSQP.
- Near-parity with bare SciPy for cached LP solves (1.08x overhead).

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
