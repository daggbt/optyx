"""Core expression system for optix."""

from optix.core.expressions import (
    Expression,
    Variable,
    Constant,
    BinaryOp,
    UnaryOp,
)
from optix.core.functions import (
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
from optix.core.compiler import (
    compile_expression,
    compile_to_dict_function,
    compile_gradient,
    CompiledExpression,
)
from optix.core.autodiff import (
    gradient,
    compute_jacobian,
    compute_hessian,
    compile_jacobian,
    compile_hessian,
)
from optix.core.verification import (
    numerical_gradient,
    verify_gradient,
    gradient_check,
    GradientCheckResult,
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
]
