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
    return UnaryOp(_ensure_expr(x), "sin")


def cos(x: Expression | float) -> UnaryOp:
    """Cosine function.
    
    Args:
        x: Expression or numeric value.
        
    Returns:
        Expression representing cos(x).
    """
    return UnaryOp(_ensure_expr(x), "cos")


def tan(x: Expression | float) -> UnaryOp:
    """Tangent function.
    
    Args:
        x: Expression or numeric value.
        
    Returns:
        Expression representing tan(x).
    """
    return UnaryOp(_ensure_expr(x), "tan")


def exp(x: Expression | float) -> UnaryOp:
    """Exponential function (e^x).
    
    Args:
        x: Expression or numeric value.
        
    Returns:
        Expression representing exp(x).
    """
    return UnaryOp(_ensure_expr(x), "exp")


def log(x: Expression | float) -> UnaryOp:
    """Natural logarithm.
    
    Args:
        x: Expression or numeric value (must be positive).
        
    Returns:
        Expression representing log(x).
    """
    return UnaryOp(_ensure_expr(x), "log")


def sqrt(x: Expression | float) -> UnaryOp:
    """Square root.
    
    Args:
        x: Expression or numeric value (must be non-negative).
        
    Returns:
        Expression representing sqrt(x).
    """
    return UnaryOp(_ensure_expr(x), "sqrt")


def abs_(x: Expression | float) -> UnaryOp:
    """Absolute value.
    
    Note: Named abs_ to avoid shadowing Python's built-in abs.
    
    Args:
        x: Expression or numeric value.
        
    Returns:
        Expression representing |x|.
    """
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
