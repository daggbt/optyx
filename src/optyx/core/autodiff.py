"""Automatic differentiation for symbolic expressions.

Implements symbolic differentiation using the chain rule, producing
gradient expressions that can be compiled for fast evaluation.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from optyx.core.expressions import Expression, Variable


def gradient(expr: Expression, wrt: Variable) -> Expression:
    """Compute the symbolic gradient of an expression with respect to a variable.

    Args:
        expr: The expression to differentiate.
        wrt: The variable to differentiate with respect to.

    Returns:
        A new Expression representing the derivative.

    Example:
        >>> x = Variable("x")
        >>> expr = x**2 + 3*x
        >>> grad = gradient(expr, x)  # Returns: 2*x + 3
    """
    return _gradient_cached(expr, wrt)


@lru_cache(maxsize=4096)
def _gradient_cached(expr: Expression, wrt: Variable) -> Expression:
    """Cached gradient computation."""
    from optyx.core.expressions import BinaryOp, Constant, UnaryOp, Variable as Var
    from optyx.core.functions import cos, sin, log, cosh, sinh

    # Constant: d/dx(c) = 0
    if isinstance(expr, Constant):
        return Constant(0.0)

    # Variable: d/dx(x) = 1, d/dx(y) = 0
    if isinstance(expr, Var):
        if expr.name == wrt.name:
            return Constant(1.0)
        else:
            return Constant(0.0)

    # Binary operations
    if isinstance(expr, BinaryOp):
        left = expr.left
        right = expr.right
        d_left = _gradient_cached(left, wrt)
        d_right = _gradient_cached(right, wrt)

        if expr.op == "+":
            # d/dx(a + b) = da + db
            return _simplify_add(d_left, d_right)

        elif expr.op == "-":
            # d/dx(a - b) = da - db
            return _simplify_sub(d_left, d_right)

        elif expr.op == "*":
            # Product rule: d/dx(a * b) = a*db + b*da
            term1 = _simplify_mul(left, d_right)
            term2 = _simplify_mul(right, d_left)
            return _simplify_add(term1, term2)

        elif expr.op == "/":
            # Quotient rule: d/dx(a / b) = (b*da - a*db) / b^2
            numerator = _simplify_sub(
                _simplify_mul(right, d_left), _simplify_mul(left, d_right)
            )
            denominator = _simplify_mul(right, right)
            return _simplify_div(numerator, denominator)

        elif expr.op == "**":
            # Power rule with chain rule
            # If exponent is constant: d/dx(a^n) = n * a^(n-1) * da
            # General case: d/dx(a^b) = a^b * (b' * ln(a) + b * a'/a)
            if isinstance(right, Constant):
                # Simple power rule: n * a^(n-1) * da
                n = right.value
                if n == 0:
                    return Constant(0.0)
                elif n == 1:
                    return d_left
                else:
                    coeff = Constant(n)
                    power = _simplify_pow(left, Constant(n - 1))
                    return _simplify_mul(_simplify_mul(coeff, power), d_left)
            else:
                # General case: a^b * (db * ln(a) + b * da / a)
                # d/dx(a^b) = a^b * (b' * ln(a) + b * a' / a)
                ln_a = log(left)
                term1 = _simplify_mul(d_right, ln_a)
                term2 = _simplify_div(_simplify_mul(right, d_left), left)
                return _simplify_mul(expr, _simplify_add(term1, term2))

        else:
            raise ValueError(f"Unknown binary operator: {expr.op}")

    # Unary operations
    if isinstance(expr, UnaryOp):
        operand = expr.operand
        d_operand = _gradient_cached(operand, wrt)

        if expr.op == "neg":
            # d/dx(-a) = -da
            return _simplify_neg(d_operand)

        elif expr.op == "abs":
            # d/dx(|a|) = sign(a) * da
            # We use a / |a| as sign(a)
            sign_expr = _simplify_div(operand, expr)
            return _simplify_mul(sign_expr, d_operand)

        elif expr.op == "sin":
            # d/dx(sin(a)) = cos(a) * da
            return _simplify_mul(cos(operand), d_operand)

        elif expr.op == "cos":
            # d/dx(cos(a)) = -sin(a) * da
            return _simplify_mul(_simplify_neg(sin(operand)), d_operand)

        elif expr.op == "tan":
            # d/dx(tan(a)) = (1 + tan^2(a)) * da = sec^2(a) * da
            # Using 1 / cos^2(a)
            cos_a = cos(operand)
            sec2 = _simplify_div(Constant(1.0), _simplify_mul(cos_a, cos_a))
            return _simplify_mul(sec2, d_operand)

        elif expr.op == "exp":
            # d/dx(exp(a)) = exp(a) * da
            return _simplify_mul(expr, d_operand)

        elif expr.op == "log":
            # d/dx(log(a)) = (1/a) * da
            return _simplify_mul(_simplify_div(Constant(1.0), operand), d_operand)

        elif expr.op == "sqrt":
            # d/dx(sqrt(a)) = (1 / (2*sqrt(a))) * da
            two_sqrt = _simplify_mul(Constant(2.0), expr)
            return _simplify_mul(_simplify_div(Constant(1.0), two_sqrt), d_operand)

        elif expr.op == "tanh":
            # d/dx(tanh(a)) = (1 - tanh^2(a)) * da
            tanh_squared = _simplify_mul(expr, expr)
            sech2 = _simplify_sub(Constant(1.0), tanh_squared)
            return _simplify_mul(sech2, d_operand)

        elif expr.op == "sinh":
            # d/dx(sinh(a)) = cosh(a) * da
            return _simplify_mul(cosh(operand), d_operand)

        elif expr.op == "cosh":
            # d/dx(cosh(a)) = sinh(a) * da
            return _simplify_mul(sinh(operand), d_operand)

        elif expr.op == "asin":
            # d/dx(asin(a)) = 1 / sqrt(1 - a^2) * da
            from optyx.core.functions import sqrt as sqrt_fn

            inner = _simplify_sub(Constant(1.0), _simplify_mul(operand, operand))
            return _simplify_mul(
                _simplify_div(Constant(1.0), sqrt_fn(inner)), d_operand
            )

        elif expr.op == "acos":
            # d/dx(acos(a)) = -1 / sqrt(1 - a^2) * da
            from optyx.core.functions import sqrt as sqrt_fn

            inner = _simplify_sub(Constant(1.0), _simplify_mul(operand, operand))
            return _simplify_mul(
                _simplify_neg(_simplify_div(Constant(1.0), sqrt_fn(inner))), d_operand
            )

        elif expr.op == "atan":
            # d/dx(atan(a)) = 1 / (1 + a^2) * da
            inner = _simplify_add(Constant(1.0), _simplify_mul(operand, operand))
            return _simplify_mul(_simplify_div(Constant(1.0), inner), d_operand)

        elif expr.op == "asinh":
            # d/dx(asinh(a)) = 1 / sqrt(1 + a^2) * da
            from optyx.core.functions import sqrt as sqrt_fn

            inner = _simplify_add(Constant(1.0), _simplify_mul(operand, operand))
            return _simplify_mul(
                _simplify_div(Constant(1.0), sqrt_fn(inner)), d_operand
            )

        elif expr.op == "acosh":
            # d/dx(acosh(a)) = 1 / sqrt(a^2 - 1) * da
            from optyx.core.functions import sqrt as sqrt_fn

            inner = _simplify_sub(_simplify_mul(operand, operand), Constant(1.0))
            return _simplify_mul(
                _simplify_div(Constant(1.0), sqrt_fn(inner)), d_operand
            )

        elif expr.op == "atanh":
            # d/dx(atanh(a)) = 1 / (1 - a^2) * da
            inner = _simplify_sub(Constant(1.0), _simplify_mul(operand, operand))
            return _simplify_mul(_simplify_div(Constant(1.0), inner), d_operand)

        elif expr.op == "log2":
            # d/dx(log2(a)) = 1 / (a * ln(2)) * da
            ln2 = Constant(np.log(2.0))
            return _simplify_mul(
                _simplify_div(Constant(1.0), _simplify_mul(operand, ln2)), d_operand
            )

        elif expr.op == "log10":
            # d/dx(log10(a)) = 1 / (a * ln(10)) * da
            ln10 = Constant(np.log(10.0))
            return _simplify_mul(
                _simplify_div(Constant(1.0), _simplify_mul(operand, ln10)), d_operand
            )

        else:
            raise ValueError(f"Unknown unary operator: {expr.op}")

    raise TypeError(f"Unknown expression type: {type(expr)}")


# Simplification helpers to reduce expression tree size


def _is_zero(expr: Expression) -> bool:
    """Check if expression is constant zero."""
    from optyx.core.expressions import Constant

    return isinstance(expr, Constant) and expr.value == 0.0


def _is_one(expr: Expression) -> bool:
    """Check if expression is constant one."""
    from optyx.core.expressions import Constant

    return isinstance(expr, Constant) and expr.value == 1.0


def _simplify_add(left: Expression, right: Expression) -> Expression:
    """Simplify addition: 0 + x -> x, x + 0 -> x."""
    if _is_zero(left):
        return right
    if _is_zero(right):
        return left
    return left + right


def _simplify_sub(left: Expression, right: Expression) -> Expression:
    """Simplify subtraction: x - 0 -> x, 0 - x -> -x."""
    if _is_zero(right):
        return left
    if _is_zero(left):
        return _simplify_neg(right)
    return left - right


def _simplify_mul(left: Expression, right: Expression) -> Expression:
    """Simplify multiplication: 0 * x -> 0, 1 * x -> x, x * 0 -> 0, x * 1 -> x."""
    from optyx.core.expressions import Constant

    if _is_zero(left) or _is_zero(right):
        return Constant(0.0)
    if _is_one(left):
        return right
    if _is_one(right):
        return left
    return left * right


def _simplify_div(left: Expression, right: Expression) -> Expression:
    """Simplify division: 0 / x -> 0, x / 1 -> x."""
    from optyx.core.expressions import Constant

    if _is_zero(left):
        return Constant(0.0)
    if _is_one(right):
        return left
    return left / right


def _simplify_neg(expr: Expression) -> Expression:
    """Simplify negation: -0 -> 0, -(-x) -> x."""
    from optyx.core.expressions import Constant, UnaryOp

    if _is_zero(expr):
        return Constant(0.0)
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        return expr.operand
    return -expr


def _simplify_pow(base: Expression, exp: Expression) -> Expression:
    """Simplify power: x^0 -> 1, x^1 -> x, 0^n -> 0 (n>0), 1^n -> 1."""
    from optyx.core.expressions import Constant

    if _is_zero(exp):
        return Constant(1.0)
    if _is_one(exp):
        return base
    if _is_zero(base):
        return Constant(0.0)
    if _is_one(base):
        return Constant(1.0)
    return base**exp


def compute_jacobian(
    exprs: list[Expression],
    variables: list[Variable],
) -> list[list[Expression]]:
    """Compute the Jacobian matrix of expressions with respect to variables.

    Args:
        exprs: List of expressions (constraints or objectives).
        variables: List of variables to differentiate with respect to.

    Returns:
        Jacobian matrix as J[i][j] = d(expr_i)/d(var_j).

    Example:
        >>> x, y = Variable("x"), Variable("y")
        >>> exprs = [x**2 + y, x*y]
        >>> J = compute_jacobian(exprs, [x, y])
        >>> # J[0][0] = 2*x, J[0][1] = 1
        >>> # J[1][0] = y, J[1][1] = x
    """
    return [[gradient(expr, var) for var in variables] for expr in exprs]


def compute_hessian(
    expr: Expression,
    variables: list[Variable],
) -> list[list[Expression]]:
    """Compute the Hessian matrix of an expression.

    Args:
        expr: The expression to differentiate twice.
        variables: List of variables.

    Returns:
        Hessian matrix as H[i][j] = d²(expr)/d(var_i)d(var_j).

    Note:
        The Hessian is symmetric, so H[i][j] = H[j][i].
        We compute the full matrix but could optimize by exploiting symmetry.
    """
    n = len(variables)
    hessian: list[list[Expression]] = []

    # First compute the gradient
    grad = [gradient(expr, var) for var in variables]

    # Then compute second derivatives
    for i in range(n):
        row: list[Expression] = []
        for j in range(n):
            # H[i][j] = d(grad[i])/d(var_j)
            row.append(gradient(grad[i], variables[j]))
        hessian.append(row)

    return hessian


def compile_jacobian(
    exprs: list[Expression],
    variables: list[Variable],
):
    """Compile the Jacobian for fast evaluation.

    Args:
        exprs: List of expressions.
        variables: List of variables.

    Returns:
        A callable that takes a 1D array and returns the Jacobian as a 2D array.

    Performance:
        For linear expressions where all Jacobian elements are constants,
        returns a pre-computed array directly (9.7x speedup vs element-by-element).
    """
    import numpy as np
    from optyx.core.compiler import compile_expression, _sanitize_derivatives
    from optyx.core.expressions import Constant

    jacobian_exprs = compute_jacobian(exprs, variables)
    m = len(exprs)
    n = len(variables)

    # Fast path: if all Jacobian elements are constants, pre-compute once
    all_constant = all(
        isinstance(jacobian_exprs[i][j], Constant) for i in range(m) for j in range(n)
    )

    if all_constant:
        # Pre-compute constant Jacobian matrix
        const_jac = np.array(
            [
                [cast(Constant, jacobian_exprs[i][j]).value for j in range(n)]
                for i in range(m)
            ],
            dtype=np.float64,
        )

        def constant_jacobian_fn(x):
            return const_jac

        return constant_jacobian_fn

    # Standard path: compile each element
    compiled_elements = [
        [compile_expression(jacobian_exprs[i][j], variables) for j in range(n)]
        for i in range(m)
    ]

    def jacobian_fn(x):
        result = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                result[i, j] = compiled_elements[i][j](x)
        return _sanitize_derivatives(result)

    return jacobian_fn


def compile_hessian(
    expr: Expression,
    variables: list[Variable],
):
    """Compile the Hessian for fast evaluation.

    Args:
        expr: The expression to differentiate.
        variables: List of variables.

    Returns:
        A callable that takes a 1D array and returns the Hessian as a 2D array.
    """
    import numpy as np
    from optyx.core.compiler import compile_expression, _sanitize_derivatives

    hessian_exprs = compute_hessian(expr, variables)
    n = len(variables)

    # Compile each element (exploiting symmetry - only upper triangle)
    compiled_elements = {}
    for i in range(n):
        for j in range(i, n):
            compiled_elements[(i, j)] = compile_expression(
                hessian_exprs[i][j], variables
            )

    def hessian_fn(x):
        result = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                val = compiled_elements[(i, j)](x)
                result[i, j] = val
                if i != j:
                    result[j, i] = val  # Symmetry
        return _sanitize_derivatives(result)

    return hessian_fn


# Aliases for convenience
jacobian = compute_jacobian
hessian = compute_hessian
