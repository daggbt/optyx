"""Shared validation helpers for variable metadata."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from optyx.core.errors import BoundsError, InvalidSizeError


VALID_DOMAINS = frozenset(("continuous", "integer", "binary"))


def validate_domain(domain: str) -> None:
    """Validate a scalar, vector, or matrix variable domain."""
    if domain not in VALID_DOMAINS:
        raise ValueError(
            f"Unknown domain: {domain!r}. Must be 'continuous', 'integer', or 'binary'."
        )


def validate_scalar_bounds(
    name: str,
    lb: float | None,
    ub: float | None,
) -> None:
    """Validate one bound pair without changing either value."""
    if lb is not None and np.isnan(lb):
        raise BoundsError(name, lb, ub, reason="Lower bound cannot be NaN")
    if ub is not None and np.isnan(ub):
        raise BoundsError(name, lb, ub, reason="Upper bound cannot be NaN")
    if lb is not None and ub is not None and lb > ub:
        raise BoundsError(name, lb, ub)


def validate_binary_bounds(
    lb: Any,
    ub: Any,
) -> None:
    """Preserve the existing explicit-bound rules for binary variables."""
    if lb is not None and np.any(np.asarray(lb, dtype=np.float64) != 0.0):
        raise ValueError(f"Binary variable must have lb=0, got {lb!r}")
    if ub is not None and np.any(np.asarray(ub, dtype=np.float64) != 1.0):
        raise ValueError(f"Binary variable must have ub=1, got {ub!r}")


def validate_vector_bounds(
    name: str,
    size: int,
    lb: float | Sequence[float] | np.ndarray[Any, Any] | None,
    ub: float | Sequence[float] | np.ndarray[Any, Any] | None,
) -> None:
    """Validate vector bounds with NumPy broadcasting and report an index."""
    lb_values = None if lb is None else np.asarray(lb, dtype=np.float64)
    ub_values = None if ub is None else np.asarray(ub, dtype=np.float64)

    for param_name, values in (("lb", lb_values), ("ub", ub_values)):
        if values is not None and values.ndim > 0 and values.shape != (size,):
            raise InvalidSizeError(
                entity=f"{param_name} for {name}",
                size=int(values.size),
                reason=f"must be a one-dimensional array of size {size}",
            )

    def value_at(values: np.ndarray[Any, Any] | None, index: int) -> float | None:
        if values is None:
            return None
        return float(values if values.ndim == 0 else values[index])

    for label, values in (("Lower", lb_values), ("Upper", ub_values)):
        if values is None:
            continue
        nan_indices = np.flatnonzero(np.isnan(values))
        if nan_indices.size:
            index = 0 if values.ndim == 0 else int(nan_indices[0])
            raise BoundsError(
                f"{name}[{index}]",
                value_at(lb_values, index),
                value_at(ub_values, index),
                reason=f"{label} bound cannot be NaN",
            )

    if lb_values is None or ub_values is None:
        return

    invalid = np.greater(lb_values, ub_values)
    if np.any(invalid):
        index = 0 if invalid.ndim == 0 else int(np.flatnonzero(invalid)[0])
        lower = value_at(lb_values, index)
        upper = value_at(ub_values, index)
        assert lower is not None and upper is not None
        raise BoundsError(f"{name}[{index}]", lower, upper)
