"""Shared post-solve feasibility checks for linear solver adapters."""

from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_FEASIBILITY_ATOL = 1e-6
DEFAULT_FEASIBILITY_RTOL = 1e-6


def compute_bound_violation(
    values: np.ndarray,
    lb: Any,
    ub: Any,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> float:
    """Return the maximum bound violation outside the configured tolerance."""
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(vals)):
        return float("inf")

    lb_arr = np.broadcast_to(np.asarray(lb, dtype=np.float64), vals.shape)
    ub_arr = np.broadcast_to(np.asarray(ub, dtype=np.float64), vals.shape)
    max_violation = 0.0

    lower_mask = np.isfinite(lb_arr)
    if np.any(lower_mask):
        lower_vals = vals[lower_mask]
        lower_bounds = lb_arr[lower_mask]
        lower_scale = np.maximum(
            1.0, np.maximum(np.abs(lower_vals), np.abs(lower_bounds))
        )
        lower_violation = lower_bounds - lower_vals
        accepted = lower_violation > atol + rtol * lower_scale
        if np.any(accepted):
            max_violation = max(max_violation, float(np.max(lower_violation[accepted])))

    upper_mask = np.isfinite(ub_arr)
    if np.any(upper_mask):
        upper_vals = vals[upper_mask]
        upper_bounds = ub_arr[upper_mask]
        upper_scale = np.maximum(
            1.0, np.maximum(np.abs(upper_vals), np.abs(upper_bounds))
        )
        upper_violation = upper_vals - upper_bounds
        accepted = upper_violation > atol + rtol * upper_scale
        if np.any(accepted):
            max_violation = max(max_violation, float(np.max(upper_violation[accepted])))

    return max_violation


def compute_linear_problem_violation(
    x: np.ndarray | None,
    *,
    A_ub: Any = None,
    b_ub: Any = None,
    A_eq: Any = None,
    b_eq: Any = None,
    lb: Any = None,
    ub: Any = None,
    integrality: np.ndarray | None = None,
    atol: float = DEFAULT_FEASIBILITY_ATOL,
    rtol: float = DEFAULT_FEASIBILITY_RTOL,
) -> float | None:
    """Check linear constraints, variable bounds, and optional integrality."""
    if x is None:
        return None

    values = np.asarray(x, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(values)):
        return float("inf")

    max_violation = 0.0
    if A_ub is not None and b_ub is not None:
        lhs = np.asarray(A_ub @ values, dtype=np.float64).reshape(-1)
        max_violation = max(
            max_violation,
            compute_bound_violation(lhs, -np.inf, b_ub, atol=atol, rtol=rtol),
        )

    if A_eq is not None and b_eq is not None:
        lhs = np.asarray(A_eq @ values, dtype=np.float64).reshape(-1)
        max_violation = max(
            max_violation,
            compute_bound_violation(lhs, b_eq, b_eq, atol=atol, rtol=rtol),
        )

    if lb is not None and ub is not None:
        max_violation = max(
            max_violation,
            compute_bound_violation(values, lb, ub, atol=atol, rtol=rtol),
        )

    if integrality is not None:
        integer_mask = np.asarray(integrality, dtype=bool)
        if np.any(integer_mask):
            integer_values = values[integer_mask]
            distance = np.abs(integer_values - np.rint(integer_values))
            accepted = distance > atol + rtol * np.maximum(1.0, np.abs(integer_values))
            if np.any(accepted):
                max_violation = max(max_violation, float(np.max(distance[accepted])))

    return max_violation
