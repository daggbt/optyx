"""Problem analysis utilities.

Provides linear / quadratic detection and helpers to compute polynomial degree
of expression trees. These utilities are used to detect LP/QP problems for
fast-path solver selection.

Performance optimizations:
- Early termination: stops traversal immediately when non-polynomial detected
- Degree-bounded traversal: is_linear/is_quadratic stop when threshold exceeded
- Memoization: caches results for repeated sub-expressions (common in constraints)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional
import numbers

from optyx.core.expressions import Expression, Constant, Variable, BinaryOp, UnaryOp


def compute_degree(expr: Expression) -> Optional[int]:
    """Compute the polynomial degree of an expression.

    Returns:
        - integer degree >= 0 if the expression is a polynomial
        - ``None`` if the expression is non-polynomial (e.g., sin, exp,
          division by variable, non-integer powers)

    Uses memoization for repeated sub-expressions.
    """
    return _compute_degree_cached(id(expr), expr)


@lru_cache(maxsize=1024)
def _compute_degree_cached(expr_id: int, expr: Expression) -> Optional[int]:
    """Memoized degree computation keyed by expression object id."""
    return _compute_degree_impl(expr)


def _compute_degree_impl(expr: Expression) -> Optional[int]:
    """Core degree computation with early termination."""
    # Fast path: leaf nodes (most common)
    if isinstance(expr, Constant):
        return 0
    if isinstance(expr, Variable):
        return 1

    # Binary operations - early termination on None
    if isinstance(expr, BinaryOp):
        op = expr.op

        # Power operator - check exponent first (often invalid)
        if op == "**":
            if not isinstance(expr.right, Constant):
                return None
            exp_val = expr.right.value
            if not isinstance(exp_val, numbers.Number):
                return None
            exp_float = float(exp_val)
            if not exp_float.is_integer() or exp_float < 0:
                return None
            left_deg = _compute_degree_impl(expr.left)
            if left_deg is None:
                return None
            return left_deg * int(exp_float)

        # Division - check denominator type first
        if op == "/":
            if not isinstance(expr.right, Constant):
                return None
            return _compute_degree_impl(expr.left)

        # Addition/Subtraction - early terminate if either side is None
        if op in ("+", "-"):
            left_deg = _compute_degree_impl(expr.left)
            if left_deg is None:
                return None
            right_deg = _compute_degree_impl(expr.right)
            if right_deg is None:
                return None
            return max(left_deg, right_deg)

        # Multiplication - only allow scalar * polynomial
        if op == "*":
            left_deg = _compute_degree_impl(expr.left)
            if left_deg is None:
                return None
            right_deg = _compute_degree_impl(expr.right)
            if right_deg is None:
                return None
            # x*y (both degree >= 1) is non-polynomial for LP detection
            if left_deg > 0 and right_deg > 0:
                return None
            return left_deg + right_deg

        # Unknown operator
        return None

    # Unary operations
    if isinstance(expr, UnaryOp):
        if expr.op == "neg":
            return _compute_degree_impl(expr.operand)
        return None

    # Unknown node type
    return None


def _check_degree_bounded(expr: Expression, max_degree: int) -> bool:
    """Check if expression degree is at most max_degree.

    Optimized traversal that terminates early when degree exceeds threshold.
    Returns False for non-polynomial expressions.
    """
    result = _check_degree_bounded_impl(expr, max_degree)
    return result is not None and result <= max_degree


def _check_degree_bounded_impl(expr: Expression, max_deg: int) -> Optional[int]:
    """Returns degree if <= max_deg, None if non-polynomial or exceeds bound."""
    # Leaf nodes
    if isinstance(expr, Constant):
        return 0
    if isinstance(expr, Variable):
        return 1 if max_deg >= 1 else None

    # Binary operations
    if isinstance(expr, BinaryOp):
        op = expr.op

        if op == "**":
            if not isinstance(expr.right, Constant):
                return None
            exp_val = expr.right.value
            if not isinstance(exp_val, numbers.Number):
                return None
            exp_float = float(exp_val)
            if not exp_float.is_integer() or exp_float < 0:
                return None
            exp_int = int(exp_float)
            # Early reject: if exponent alone exceeds max, base must be constant
            if exp_int > max_deg:
                left_deg = _check_degree_bounded_impl(expr.left, 0)
                if left_deg != 0:
                    return None
                return 0
            left_deg = _check_degree_bounded_impl(
                expr.left, max_deg // exp_int if exp_int else max_deg
            )
            if left_deg is None:
                return None
            result = left_deg * exp_int
            return result if result <= max_deg else None

        if op == "/":
            if not isinstance(expr.right, Constant):
                return None
            return _check_degree_bounded_impl(expr.left, max_deg)

        if op in ("+", "-"):
            left_deg = _check_degree_bounded_impl(expr.left, max_deg)
            if left_deg is None:
                return None
            right_deg = _check_degree_bounded_impl(expr.right, max_deg)
            if right_deg is None:
                return None
            return max(left_deg, right_deg)

        if op == "*":
            left_deg = _check_degree_bounded_impl(expr.left, max_deg)
            if left_deg is None:
                return None
            # If left is non-constant, right must have degree such that sum <= max_deg
            remaining = max_deg - left_deg if left_deg > 0 else max_deg
            right_deg = _check_degree_bounded_impl(
                expr.right, remaining if left_deg > 0 else max_deg
            )
            if right_deg is None:
                return None
            # x*y is non-polynomial for LP detection
            if left_deg > 0 and right_deg > 0:
                return None
            result = left_deg + right_deg
            return result if result <= max_deg else None

        return None

    # Unary operations
    if isinstance(expr, UnaryOp):
        if expr.op == "neg":
            return _check_degree_bounded_impl(expr.operand, max_deg)
        return None

    return None


def is_linear(expr: Expression) -> bool:
    """Return True if expression is linear (degree ≤ 1).

    Constant expressions are considered linear (degree 0).
    Uses optimized bounded-degree traversal that terminates early.
    """
    return _check_degree_bounded(expr, 1)


def is_quadratic(expr: Expression) -> bool:
    """Return True if expression is quadratic (degree ≤ 2).

    Returns False for non-polynomial expressions.
    Uses optimized bounded-degree traversal that terminates early.
    """
    return _check_degree_bounded(expr, 2)


def clear_degree_cache() -> None:
    """Clear the memoization cache for degree computation.

    Call this if expressions are being reused across different problems
    and memory usage becomes a concern.
    """
    _compute_degree_cached.cache_clear()
