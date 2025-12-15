# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
