"""Optyx: Symbolic optimization without the boilerplate."""

from importlib.metadata import version

from optyx.core.expressions import (
    Expression,
    Variable,
    Constant,
)
from optyx.core.vectors import VectorVariable
from optyx.core.variable_dict import VariableDict
from optyx.core.matrices import (
    ConstantMatrix,
    MatrixVariable,
    MatrixVectorProduct,
    QuadraticForm,
    FrobeniusNorm,
    as_matrix,
    matmul,
    quadratic_form,
    trace,
    diag,
    diag_matrix,
    frobenius_norm,
)
from optyx.core.parameters import Parameter, VectorParameter, MatrixParameter
from optyx.core.autodiff import increased_recursion_limit
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
from optyx.solution import Solution, SolverStatus, SolverProgress

__version__ = version("optyx")


def BinaryVariable(name: str, **kwargs) -> Variable:
    """Create a binary (0/1) variable.

    Shorthand for ``Variable(name, domain='binary', lb=0, ub=1)``.
    """
    return Variable(name, domain="binary", **kwargs)


def IntegerVariable(name: str, lb=None, ub=None, **kwargs) -> Variable:
    """Create an integer variable.

    Shorthand for ``Variable(name, domain='integer', lb=lb, ub=ub)``.
    """
    return Variable(name, domain="integer", lb=lb, ub=ub, **kwargs)


__all__ = [
    # Core
    "Expression",
    "Variable",
    "Constant",
    "BinaryVariable",
    "IntegerVariable",
    "VectorVariable",
    "VariableDict",
    "MatrixVariable",
    "ConstantMatrix",
    # Parameters
    "Parameter",
    "VectorParameter",
    "MatrixParameter",
    # Matrix operations
    "MatrixVectorProduct",
    "QuadraticForm",
    "FrobeniusNorm",
    "as_matrix",
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
    "SolverProgress",
    # Utilities
    "increased_recursion_limit",
]
