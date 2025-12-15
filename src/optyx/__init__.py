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
from optyx.constraints import Constraint
from optyx.problem import Problem
from optyx.solution import Solution, SolverStatus

__version__ = "1.0.0"

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
    # Problem definition
    "Constraint",
    "Problem",
    "Solution",
    "SolverStatus",
]
