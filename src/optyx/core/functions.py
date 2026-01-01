"""Transcendental and mathematical functions for expressions.

All functions accept Expression objects and return UnaryOp nodes.
Under the hood, evaluation uses numpy's implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from optyx.core.expressions import Expression, UnaryOp, _ensure_expr

if TYPE_CHECKING:
    pass


def sin(x: Expression | float) -> UnaryOp:
    """Sine function.

    Args:
        x: Expression or numeric value.

    Returns:
        Expression representing sin(x).
    """
    # Handle vector inputs element-wise
    from optyx.core.vectors import VectorExpression, VectorVariable

    if isinstance(x, (VectorExpression, VectorVariable)):
        return VectorExpression([sin(xi) for xi in x])  # type: ignore

    return UnaryOp(_ensure_expr(x), "sin")


def cos(x: Expression | float) -> UnaryOp:
    """Cosine function.

    Args:
        x: Expression or numeric value.

    Returns:
        Expression representing cos(x).
    """
    # Handle vector inputs element-wise
    from optyx.core.vectors import VectorExpression, VectorVariable

    if isinstance(x, (VectorExpression, VectorVariable)):
        return VectorExpression([cos(xi) for xi in x])  # type: ignore

    return UnaryOp(_ensure_expr(x), "cos")


def tan(x: Expression | float) -> UnaryOp:
    """Tangent function.

    Args:
        x: Expression or numeric value.

    Returns:
        Expression representing tan(x).
    """
    # Handle vector inputs element-wise
    from optyx.core.vectors import VectorExpression, VectorVariable

    if isinstance(x, (VectorExpression, VectorVariable)):
        return VectorExpression([tan(xi) for xi in x])  # type: ignore

    return UnaryOp(_ensure_expr(x), "tan")


def exp(x: Expression | float) -> UnaryOp:
    """Exponential function (e^x).

    Args:
        x: Expression or numeric value.

    Returns:
        Expression representing exp(x).
    """
    # Handle vector inputs element-wise
    from optyx.core.vectors import VectorExpression, VectorVariable

    if isinstance(x, (VectorExpression, VectorVariable)):
        return VectorExpression([exp(xi) for xi in x])  # type: ignore

    return UnaryOp(_ensure_expr(x), "exp")


def log(x: Expression | float) -> UnaryOp:
    """Natural logarithm.

    Args:
        x: Expression or numeric value (must be positive).

    Returns:
        Expression representing log(x).
    """
    # Handle vector inputs element-wise
    from optyx.core.vectors import VectorExpression, VectorVariable

    if isinstance(x, (VectorExpression, VectorVariable)):
        return VectorExpression([log(xi) for xi in x])  # type: ignore

    return UnaryOp(_ensure_expr(x), "log")


def sqrt(x: Expression | float) -> UnaryOp:
    """Square root.

    Args:
        x: Expression or numeric value (must be non-negative).

    Returns:
        Expression representing sqrt(x).
    """
    # Handle vector inputs element-wise
    from optyx.core.vectors import VectorExpression, VectorVariable

    if isinstance(x, (VectorExpression, VectorVariable)):
        return VectorExpression([sqrt(xi) for xi in x])  # type: ignore

    return UnaryOp(_ensure_expr(x), "sqrt")


def abs_(x: Expression | float) -> UnaryOp:
    """Absolute value.

    Note: Named abs_ to avoid shadowing Python's built-in abs.

    Args:
        x: Expression or numeric value.

    Returns:
        Expression representing |x|.
    """
    # Handle vector inputs element-wise
    from optyx.core.vectors import VectorExpression, VectorVariable

    if isinstance(x, (VectorExpression, VectorVariable)):
        return VectorExpression([abs_(xi) for xi in x])  # type: ignore

    return UnaryOp(_ensure_expr(x), "abs")


def tanh(x: Expression | float) -> UnaryOp:
    """Hyperbolic tangent.

    Args:
        x: Expression or numeric value.

    Returns:
        Expression representing tanh(x).
    """
    return UnaryOp(_ensure_expr(x), "tanh")


def sinh(x: Expression | float) -> UnaryOp:
    """Hyperbolic sine.

    Args:
        x: Expression or numeric value.

    Returns:
        Expression representing sinh(x).
    """
    return UnaryOp(_ensure_expr(x), "sinh")


def cosh(x: Expression | float) -> UnaryOp:
    """Hyperbolic cosine.

    Args:
        x: Expression or numeric value.

    Returns:
        Expression representing cosh(x).
    """
    return UnaryOp(_ensure_expr(x), "cosh")


# =============================================================================
# Inverse Trigonometric Functions
# =============================================================================


def asin(x: Expression | float) -> UnaryOp:
    """Inverse sine (arcsine).

    Args:
        x: Expression or numeric value in [-1, 1].

    Returns:
        Expression representing arcsin(x), result in [-π/2, π/2].
    """
    return UnaryOp(_ensure_expr(x), "asin")


def acos(x: Expression | float) -> UnaryOp:
    """Inverse cosine (arccosine).

    Args:
        x: Expression or numeric value in [-1, 1].

    Returns:
        Expression representing arccos(x), result in [0, π].
    """
    return UnaryOp(_ensure_expr(x), "acos")


def atan(x: Expression | float) -> UnaryOp:
    """Inverse tangent (arctangent).

    Args:
        x: Expression or numeric value.

    Returns:
        Expression representing arctan(x), result in (-π/2, π/2).
    """
    return UnaryOp(_ensure_expr(x), "atan")


# =============================================================================
# Inverse Hyperbolic Functions
# =============================================================================


def asinh(x: Expression | float) -> UnaryOp:
    """Inverse hyperbolic sine.

    Args:
        x: Expression or numeric value.

    Returns:
        Expression representing arcsinh(x).
    """
    return UnaryOp(_ensure_expr(x), "asinh")


def acosh(x: Expression | float) -> UnaryOp:
    """Inverse hyperbolic cosine.

    Args:
        x: Expression or numeric value >= 1.

    Returns:
        Expression representing arccosh(x).
    """
    return UnaryOp(_ensure_expr(x), "acosh")


def atanh(x: Expression | float) -> UnaryOp:
    """Inverse hyperbolic tangent.

    Args:
        x: Expression or numeric value in (-1, 1).

    Returns:
        Expression representing arctanh(x).
    """
    return UnaryOp(_ensure_expr(x), "atanh")


# =============================================================================
# Additional Logarithm Bases
# =============================================================================


def log2(x: Expression | float) -> UnaryOp:
    """Base-2 logarithm.

    Args:
        x: Expression or numeric value (must be positive).

    Returns:
        Expression representing log₂(x).
    """
    return UnaryOp(_ensure_expr(x), "log2")


def log10(x: Expression | float) -> UnaryOp:
    """Base-10 logarithm.

    Args:
        x: Expression or numeric value (must be positive).

    Returns:
        Expression representing log₁₀(x).
    """
    return UnaryOp(_ensure_expr(x), "log10")
