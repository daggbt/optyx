"""Core expression system for optyx."""

from optyx.core.expressions import (
    Expression,
    Variable,
    Constant,
    BinaryOp,
    UnaryOp,
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
from optyx.core.compiler import (
    compile_expression,
    compile_to_dict_function,
    compile_gradient,
    CompiledExpression,
)
from optyx.core.autodiff import (
    gradient,
    compute_jacobian,
    compute_hessian,
    compile_jacobian,
    compile_hessian,
)
from optyx.core.verification import (
    numerical_gradient,
    verify_gradient,
    gradient_check,
    GradientCheckResult,
)
from optyx.core.errors import (
    OptyxError,
    DimensionMismatchError,
    InvalidOperationError,
    BoundsError,
    EmptyContainerError,
    SolverError,
    InfeasibleError,
    UnboundedError,
    NotSolvedError,
    ParameterError,
)

__all__ = [
    # Expressions
    "Expression",
    "Variable",
    "Constant",
    "BinaryOp",
    "UnaryOp",
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
    # Compiler
    "compile_expression",
    "compile_to_dict_function",
    "compile_gradient",
    "CompiledExpression",
    # Autodiff
    "gradient",
    "compute_jacobian",
    "compute_hessian",
    "compile_jacobian",
    "compile_hessian",
    # Verification
    "numerical_gradient",
    "verify_gradient",
    "gradient_check",
    "GradientCheckResult",
    # Errors
    "OptyxError",
    "DimensionMismatchError",
    "InvalidOperationError",
    "BoundsError",
    "EmptyContainerError",
    "SolverError",
    "InfeasibleError",
    "UnboundedError",
    "NotSolvedError",
    "ParameterError",
]
