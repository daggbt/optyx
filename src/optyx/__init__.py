"""Optyx: Symbolic optimization without the boilerplate."""

from importlib.metadata import version

from optyx.core.expressions import (
    Expression,
    Variable,
    Constant,
)
from optyx.core.vectors import VectorVariable
from optyx.core.matrices import (
    MatrixVariable,
    MatrixVectorProduct,
    QuadraticForm,
    FrobeniusNorm,
    matmul,
    quadratic_form,
    trace,
    diag,
    diag_matrix,
    frobenius_norm,
)
from optyx.core.parameters import Parameter, VectorParameter, MatrixParameter
from optyx.core.functions import (
    sin,
    cos,
    tan,
    exp,
    log,
    log2,
    log10,
    sqrt,
    abs_,
    tanh,
    sinh,
    cosh,
    asin,
    acos,
    atan,
    asinh,
    acosh,
    atanh,
)
from optyx.constraints import Constraint
from optyx.problem import Problem
from optyx.solution import Solution, SolverStatus

__version__ = version("optyx")

__all__ = [
    # Core
    "Expression",
    "Variable",
    "Constant",
    "VectorVariable",
    "MatrixVariable",
    # Parameters
    "Parameter",
    "VectorParameter",
    "MatrixParameter",
    # Matrix operations
    "MatrixVectorProduct",
    "QuadraticForm",
    "FrobeniusNorm",
    "matmul",
    "quadratic_form",
    "trace",
    "diag",
    "diag_matrix",
    "frobenius_norm",
    # Functions
    "sin",
    "cos",
    "tan",
    "exp",
    "log",
    "log2",
    "log10",
    "sqrt",
    "abs_",
    "tanh",
    "sinh",
    "cosh",
    "asin",
    "acos",
    "atan",
    "asinh",
    "acosh",
    "atanh",
    # Problem definition
    "Constraint",
    "Problem",
    "Solution",
    "SolverStatus",
]
