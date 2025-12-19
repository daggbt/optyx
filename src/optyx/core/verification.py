"""Gradient verification utilities.

Provides tools for verifying symbolic gradients against numerical
differentiation using finite differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

import numpy as np

if TYPE_CHECKING:
    from optyx.core.expressions import Expression, Variable


def numerical_gradient(
    expr: Expression,
    wrt: Variable,
    point: Mapping[str, float],
    eps: float = 1e-7,
) -> float:
    """Compute the numerical gradient using central differences.

    Args:
        expr: The expression to differentiate.
        wrt: The variable to differentiate with respect to.
        point: Dictionary of variable values.
        eps: Finite difference step size.

    Returns:
        The numerical derivative at the given point.
    """
    point_plus = dict(point)
    point_minus = dict(point)

    point_plus[wrt.name] = point[wrt.name] + eps
    point_minus[wrt.name] = point[wrt.name] - eps

    f_plus = expr.evaluate(point_plus)
    f_minus = expr.evaluate(point_minus)

    result = (f_plus - f_minus) / (2 * eps)
    # Convert to scalar if needed
    if isinstance(result, np.ndarray):
        return float(result.item())
    return float(result)


def numerical_gradient_array(
    expr: Expression,
    variables: list[Variable],
    x: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """Compute the numerical gradient as an array.

    Args:
        expr: The expression to differentiate.
        variables: List of variables (defines ordering).
        x: Array of variable values.
        eps: Finite difference step size.

    Returns:
        Array of partial derivatives.
    """
    n = len(variables)
    grad = np.zeros(n)

    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps

        point_plus = {v.name: x_plus[j] for j, v in enumerate(variables)}
        point_minus = {v.name: x_minus[j] for j, v in enumerate(variables)}

        f_plus = expr.evaluate(point_plus)
        f_minus = expr.evaluate(point_minus)

        grad[i] = (f_plus - f_minus) / (2 * eps)

    return grad


def verify_gradient(
    expr: Expression,
    wrt: Variable,
    point: Mapping[str, float],
    tol: float = 1e-5,
) -> bool:
    """Verify that the symbolic gradient matches numerical differentiation.

    Args:
        expr: The expression to check.
        wrt: The variable to differentiate with respect to.
        point: Dictionary of variable values.
        tol: Tolerance for comparison.

    Returns:
        True if gradients match within tolerance.
    """
    from optyx.core.autodiff import gradient

    # Compute symbolic gradient
    grad_expr = gradient(expr, wrt)
    symbolic_grad = grad_expr.evaluate(point)

    # Compute numerical gradient
    numerical_grad = numerical_gradient(expr, wrt, point)

    # Compare
    diff = abs(symbolic_grad - numerical_grad)
    return bool(diff < tol)


def verify_gradient_array(
    expr: Expression,
    variables: list[Variable],
    x: np.ndarray,
    tol: float = 1e-5,
) -> tuple[bool, np.ndarray, np.ndarray]:
    """Verify gradient for all variables at once.

    Args:
        expr: The expression to check.
        variables: List of variables.
        x: Array of variable values.
        tol: Tolerance for comparison.

    Returns:
        Tuple of (all_match, symbolic_grad, numerical_grad).
    """
    from optyx.core.autodiff import gradient

    point = {v.name: x[i] for i, v in enumerate(variables)}
    n = len(variables)

    symbolic_grad = np.zeros(n)
    for i, var in enumerate(variables):
        grad_expr = gradient(expr, var)
        symbolic_grad[i] = grad_expr.evaluate(point)

    numerical_grad = numerical_gradient_array(expr, variables, x)

    all_match = np.allclose(symbolic_grad, numerical_grad, atol=tol)

    return all_match, symbolic_grad, numerical_grad


@dataclass
class GradientCheckResult:
    """Result of a gradient check over multiple samples."""

    n_samples: int
    n_checks: int  # Total number of gradient checks (samples × variables)
    n_passed: int
    n_failed: int
    max_error: float
    mean_error: float
    failed_points: list[dict]

    @property
    def all_passed(self) -> bool:
        return self.n_failed == 0

    def __repr__(self) -> str:
        status = "PASSED" if self.all_passed else "FAILED"
        return (
            f"GradientCheckResult({status}: {self.n_passed}/{self.n_checks} checks passed, "
            f"max_error={self.max_error:.2e}, mean_error={self.mean_error:.2e})"
        )


def gradient_check(
    expr: Expression,
    variables: list[Variable],
    n_samples: int = 100,
    bounds: tuple[float, float] = (-10.0, 10.0),
    tol: float = 1e-5,
    seed: int | None = None,
) -> GradientCheckResult:
    """Perform gradient checking over multiple random points.

    Args:
        expr: The expression to check.
        variables: List of variables.
        n_samples: Number of random points to test.
        bounds: Range for random values (min, max).
        tol: Tolerance for comparison.
        seed: Random seed for reproducibility.

    Returns:
        GradientCheckResult with detailed statistics.
    """
    from optyx.core.autodiff import gradient

    rng = np.random.default_rng(seed)
    n_vars = len(variables)

    errors = []
    failed_points = []
    n_failed = 0
    n_checks = n_samples * n_vars

    for _ in range(n_samples):
        # Sample a random point
        x = rng.uniform(bounds[0], bounds[1], size=n_vars)
        point = {v.name: float(x[i]) for i, v in enumerate(variables)}

        # Compute symbolic gradient for each variable
        symbolic_grad = np.zeros(n_vars)
        for i, var in enumerate(variables):
            grad_expr = gradient(expr, var)
            symbolic_grad[i] = grad_expr.evaluate(point)

        # Numerical gradient
        numerical_grad = numerical_gradient_array(expr, variables, x)

        # Per-variable absolute errors
        abs_errors = np.abs(symbolic_grad - numerical_grad)
        errors.append(float(np.max(abs_errors)))

        # Track failed variable checks
        n_failed += int(np.count_nonzero(abs_errors > tol))
        if np.any(abs_errors > tol):
            failed_points.append(
                {
                    "point": point,
                    "symbolic": symbolic_grad.tolist(),
                    "numerical": numerical_grad.tolist(),
                    "errors": abs_errors.tolist(),
                }
            )

    # Summary statistics
    max_error = float(max(errors) if errors else 0.0)
    mean_error = float(np.mean(errors) if errors else 0.0)
    n_passed = n_checks - n_failed

    return GradientCheckResult(
        n_samples=n_samples,
        n_checks=n_checks,
        n_passed=n_passed,
        n_failed=n_failed,
        max_error=max_error,
        mean_error=mean_error,
        failed_points=failed_points,
    )
