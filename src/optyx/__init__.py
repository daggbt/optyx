"""Optyx: Symbolic optimization without the boilerplate."""

from optyx.core.expressions import (
    Expression,
    Variable,
    Constant,
)
from optyx.core.functions import (
    sin,
    cos,
    tan,
    exp,
    log,
    sqrt,
    abs_,
    tanh,
    sinh,
    cosh,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "Expression",
    "Variable",
    "Constant",
    # Functions
    "sin",
    "cos",
    "tan",
    "exp",
    "log",
    "sqrt",
    "abs_",
    "tanh",
    "sinh",
    "cosh",
]
