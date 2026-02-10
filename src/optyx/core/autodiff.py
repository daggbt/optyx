"""Automatic differentiation for symbolic expressions.

Implements symbolic differentiation using the chain rule, producing
gradient expressions that can be compiled for fast evaluation.

Supports native gradient rules for vector expressions (VectorSum,
LinearCombination, DotProduct) with O(1) coefficient lookup for
scalability to n=10,000+ variables.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Iterator, cast

import numpy as np
from numpy.typing import NDArray

from optyx.core.errors import InvalidExpressionError, UnknownOperatorError

if TYPE_CHECKING:
    from optyx.core.expressions import Expression, Variable


# =============================================================================
# Utilities
# =============================================================================


@contextmanager
def increased_recursion_limit(limit: int = 5000) -> Iterator[None]:
    """Temporarily increase Python's recursion limit.

    This context manager can be used as a workaround for deep expression trees
    when the automatic iterative/recursive switching isn't sufficient.

    .. warning::
        Use with caution - very high limits can cause stack overflow crashes.
        The iterative gradient implementation is preferred for deep trees.

    Args:
        limit: The temporary recursion limit (default: 5000).

    Yields:
        None. The limit is restored when the context exits.

    Example:
        >>> with increased_recursion_limit(5000):
        ...     grad = gradient(deep_expr, x)
    """
    old_limit = sys.getrecursionlimit()
    try:
        sys.setrecursionlimit(limit)
        yield
    finally:
        sys.setrecursionlimit(old_limit)


# =============================================================================
# Gradient Registry System
# =============================================================================

# Type alias for gradient functions - uses ... for argument types
# since actual gradient functions take specific Expression subclasses
GradientFunc = Callable[..., "Expression"]

# Registry mapping expression types to gradient functions
_gradient_registry: dict[type, GradientFunc] = {}


def register_gradient(expr_type: type) -> Callable[[GradientFunc], GradientFunc]:
    """Decorator to register a gradient rule for an expression type.

    Registered gradient rules are used by the main `gradient()` function
    before falling back to recursive tree traversal. This enables O(1)
    gradient computation for vector expressions.

    Args:
        expr_type: The expression class to register a gradient rule for.

    Returns:
        A decorator that registers the gradient function.

    Example:
        @register_gradient(VectorSum)
        def gradient_vector_sum(expr: VectorSum, wrt: Variable) -> Expression:
            # O(1) gradient computation
            ...
    """

    def decorator(func: GradientFunc) -> GradientFunc:
        _gradient_registry[expr_type] = func
        return func

    return decorator


def has_gradient_rule(expr: "Expression") -> bool:
    """Check if an expression type has a registered gradient rule.

    Args:
        expr: The expression to check.

    Returns:
        True if a gradient rule is registered for this expression type.
    """
    return type(expr) in _gradient_registry


def apply_gradient_rule(expr: "Expression", wrt: "Variable") -> "Expression":
    """Apply the registered gradient rule for an expression type.

    Args:
        expr: The expression to differentiate.
        wrt: The variable to differentiate with respect to.

    Returns:
        The gradient expression.

    Raises:
        ValueError: If no gradient rule is registered for this expression type.
    """
    func = _gradient_registry.get(type(expr))
    if func is None:
        raise InvalidExpressionError(
            expr_type=type(expr),
            context="gradient computation",
            suggestion=f"Register a gradient rule using @register_gradient({type(expr).__name__}) or use supported expression types.",
        )
    return func(expr, wrt)


# =============================================================================
# Main Gradient Function
# =============================================================================

# Threshold for switching to iterative gradient computation
# Python's default recursion limit is 1000, so we use 400 to be safe
_RECURSION_THRESHOLD = 400


def gradient(expr: Expression, wrt: Variable) -> Expression:
    """Compute the symbolic gradient of an expression with respect to a variable.

    Uses a three-tier approach for optimal performance:
    1. Registered gradient rules (O(1) for vector expressions)
    2. Cached recursive computation (for shallow trees)
    3. Iterative fallback (for deep trees to avoid RecursionError)

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
    # Fast path: use registered gradient rules for vector expressions
    if has_gradient_rule(expr):
        return apply_gradient_rule(expr, wrt)

    # Check tree depth to decide between recursive and iterative
    depth = _estimate_tree_depth(expr)
    if depth >= _RECURSION_THRESHOLD:
        return _gradient_iterative(expr, wrt)

    return _gradient_cached(expr, wrt)


def _estimate_tree_depth(
    expr: Expression, max_check: int = 500, full_traversal: bool = False
) -> int:
    """Estimate expression tree depth.

    Two modes available:

    1. **Left-spine heuristic** (default, full_traversal=False):
       - Time: O(depth) - very fast
       - Accuracy: Exact for left-skewed trees (e.g., ``obj = obj + x[i]``)
       - Use case: Default for gradient() auto-switching

    2. **Full traversal** (full_traversal=True):
       - Time: O(n) where n = total nodes
       - Accuracy: Exact for ANY tree shape
       - Use case: Diagnostics, debugging, right-skewed trees

    Args:
        expr: The expression to check.
        max_check: Maximum depth to check before returning early.
        full_traversal: If True, explore all branches for exact depth.

    Returns:
        Estimated (or exact) tree depth.

    Example:
        >>> # Left-skewed tree from loop (common case)
        >>> x = VectorVariable("x", 1000)
        >>> obj = x[0]
        >>> for i in range(1, 1000):
        ...     obj = obj + x[i]  # Always adds to left
        >>> _estimate_tree_depth(obj)  # Fast: O(depth)
        999

        >>> # Right-skewed tree (rare case)
        >>> obj = x[999]
        >>> for i in range(998, -1, -1):
        ...     obj = x[i] + obj  # Always adds to right
        >>> _estimate_tree_depth(obj)  # Returns 1 (wrong!)
        1
        >>> _estimate_tree_depth(obj, full_traversal=True)  # Correct
        999
    """
    from optyx.core.expressions import BinaryOp, UnaryOp

    if full_traversal:
        # Full traversal: O(n) time, explores ALL branches
        stack: list[tuple[Expression, int]] = [(expr, 0)]
        max_seen = 0

        while stack and max_seen < max_check:
            node, depth = stack.pop()
            max_seen = max(max_seen, depth)

            if isinstance(node, BinaryOp):
                stack.append((node.left, depth + 1))
                stack.append((node.right, depth + 1))
            elif isinstance(node, UnaryOp):
                stack.append((node.operand, depth + 1))

        return max_seen
    else:
        # Left-spine heuristic: O(depth) time, follows left child only
        depth = 0
        current = expr
        while depth < max_check:
            if isinstance(current, BinaryOp):
                current = current.left  # Follow left spine
                depth += 1
            elif isinstance(current, UnaryOp):
                current = current.operand
                depth += 1
            else:
                break
        return depth


@lru_cache(maxsize=4096)
def _gradient_cached(expr: Expression, wrt: Variable) -> Expression:
    """Cached recursive gradient computation.

    Used for shallow expression trees where recursion is safe and fast.
    """
    from optyx.core.expressions import (
        BinaryOp,
        Constant,
        NaryProduct,
        NarySum,
        UnaryOp,
        Variable as Var,
    )
    from optyx.core.functions import cos, sin, log, cosh, sinh
    from optyx.core.parameters import Parameter

    # Fast path: check for registered gradient rules first
    if has_gradient_rule(expr):
        return apply_gradient_rule(expr, wrt)

    # Constant: d/dx(c) = 0
    if isinstance(expr, Constant):
        return Constant(0.0)

    # Parameter: d/dx(p) = 0 (parameters are constants w.r.t. variables)
    if isinstance(expr, Parameter):
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
            raise UnknownOperatorError(
                operator=expr.op,
                context="gradient computation",
            )

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
            raise UnknownOperatorError(
                operator=expr.op,
                context="gradient computation (unary)",
            )

    # N-ary operations
    if isinstance(expr, NarySum):
        # d/dx(a + b + c + ...) = da + db + dc + ...
        result: Expression = Constant(0.0)
        for term in expr.terms:
            result = _simplify_add(result, _gradient_cached(term, wrt))
        return result

    if isinstance(expr, NaryProduct):
        # Generalized product rule:
        # d/dx(a * b * c) = da*b*c + a*db*c + a*b*dc
        factors = expr.factors
        result = Constant(0.0)
        for i, factor in enumerate(factors):
            d_factor = _gradient_cached(factor, wrt)
            # Build product of all other factors
            other_product: Expression = Constant(1.0)
            for j, other in enumerate(factors):
                if j != i:
                    other_product = _simplify_mul(other_product, other)
            result = _simplify_add(result, _simplify_mul(d_factor, other_product))
        return result

    raise InvalidExpressionError(
        expr_type=type(expr),
        context="gradient computation",
        suggestion="Use Variable, Constant, BinaryOp, or UnaryOp expressions.",
    )


# =============================================================================


def _gradient_iterative(expr: Expression, wrt: Variable) -> Expression:
    """Iterative gradient computation using explicit stack.

    Used for deep expression trees that would cause RecursionError with
    the recursive approach. Implements post-order traversal with gradient
    propagation using an explicit stack.

    This enables gradient computation for expressions with depth > 1000
    (Python's default recursion limit), such as incrementally built sums
    with n=10,000+ terms.

    Args:
        expr: The expression to differentiate.
        wrt: The variable to differentiate with respect to.

    Returns:
        The gradient expression.
    """
    from optyx.core.expressions import (
        BinaryOp,
        Constant,
        NaryProduct,
        NarySum,
        UnaryOp,
        Variable as Var,
    )
    from optyx.core.functions import cos, sin, log, cosh, sinh
    from optyx.core.parameters import Parameter

    # Check for registered gradient rules first
    if has_gradient_rule(expr):
        return apply_gradient_rule(expr, wrt)

    # Stack-based iterative computation
    # Each entry: (expr, phase, children_grads)
    # phase 0: first visit, push children
    # phase 1: children processed, compute gradient
    stack: list[tuple[Expression, int, list[Expression]]] = [(expr, 0, [])]
    results: dict[int, Expression] = {}  # id(expr) -> gradient

    while stack:
        current, phase, child_grads = stack.pop()
        node_id = id(current)

        # Check if already computed
        if node_id in results:
            continue

        # Check for registered gradient rules
        if has_gradient_rule(current):
            results[node_id] = apply_gradient_rule(current, wrt)
            continue

        # Base cases
        if isinstance(current, Constant):
            results[node_id] = Constant(0.0)
            continue

        if isinstance(current, Parameter):
            results[node_id] = Constant(0.0)
            continue

        if isinstance(current, Var):
            results[node_id] = (
                Constant(1.0) if current.name == wrt.name else Constant(0.0)
            )
            continue

        # Binary operations
        if isinstance(current, BinaryOp):
            left, right = current.left, current.right

            if phase == 0:
                # First visit: need to process children first
                left_id, right_id = id(left), id(right)

                if left_id in results and right_id in results:
                    # Both children already computed
                    d_left = results[left_id]
                    d_right = results[right_id]
                else:
                    # Push self back with phase 1, then push children
                    stack.append((current, 1, []))
                    if right_id not in results:
                        stack.append((right, 0, []))
                    if left_id not in results:
                        stack.append((left, 0, []))
                    continue

            else:
                # Phase 1: children are computed
                d_left = results[id(left)]
                d_right = results[id(right)]

            # Compute gradient based on operator
            if current.op == "+":
                results[node_id] = _simplify_add(d_left, d_right)
            elif current.op == "-":
                results[node_id] = _simplify_sub(d_left, d_right)
            elif current.op == "*":
                term1 = _simplify_mul(left, d_right)
                term2 = _simplify_mul(right, d_left)
                results[node_id] = _simplify_add(term1, term2)
            elif current.op == "/":
                num = _simplify_sub(
                    _simplify_mul(right, d_left), _simplify_mul(left, d_right)
                )
                denom = _simplify_mul(right, right)
                results[node_id] = _simplify_div(num, denom)
            elif current.op == "**":
                if isinstance(right, Constant):
                    n = right.value
                    if n == 0:
                        results[node_id] = Constant(0.0)
                    elif n == 1:
                        results[node_id] = d_left
                    else:
                        coeff = Constant(n)
                        power = _simplify_pow(left, Constant(n - 1))
                        results[node_id] = _simplify_mul(
                            _simplify_mul(coeff, power), d_left
                        )
                else:
                    ln_a = log(left)
                    term1 = _simplify_mul(d_right, ln_a)
                    term2 = _simplify_div(_simplify_mul(right, d_left), left)
                    results[node_id] = _simplify_mul(
                        current, _simplify_add(term1, term2)
                    )
            else:
                raise UnknownOperatorError(
                    operator=current.op,
                    context="iterative gradient computation (binary)",
                )
            continue

        # Unary operations
        if isinstance(current, UnaryOp):
            operand = current.operand

            if phase == 0:
                operand_id = id(operand)
                if operand_id in results:
                    d_operand = results[operand_id]
                else:
                    stack.append((current, 1, []))
                    stack.append((operand, 0, []))
                    continue
            else:
                d_operand = results[id(operand)]

            # Compute gradient based on operator
            if current.op == "neg":
                results[node_id] = _simplify_neg(d_operand)
            elif current.op == "abs":
                sign_expr = _simplify_div(operand, current)
                results[node_id] = _simplify_mul(sign_expr, d_operand)
            elif current.op == "sin":
                results[node_id] = _simplify_mul(cos(operand), d_operand)
            elif current.op == "cos":
                results[node_id] = _simplify_mul(_simplify_neg(sin(operand)), d_operand)
            elif current.op == "tan":
                cos_a = cos(operand)
                sec2 = _simplify_div(Constant(1.0), _simplify_mul(cos_a, cos_a))
                results[node_id] = _simplify_mul(sec2, d_operand)
            elif current.op == "exp":
                results[node_id] = _simplify_mul(current, d_operand)
            elif current.op == "log":
                results[node_id] = _simplify_mul(
                    _simplify_div(Constant(1.0), operand), d_operand
                )
            elif current.op == "sqrt":
                two_sqrt = _simplify_mul(Constant(2.0), current)
                results[node_id] = _simplify_mul(
                    _simplify_div(Constant(1.0), two_sqrt), d_operand
                )
            elif current.op == "tanh":
                tanh_squared = _simplify_mul(current, current)
                sech2 = _simplify_sub(Constant(1.0), tanh_squared)
                results[node_id] = _simplify_mul(sech2, d_operand)
            elif current.op == "sinh":
                results[node_id] = _simplify_mul(cosh(operand), d_operand)
            elif current.op == "cosh":
                results[node_id] = _simplify_mul(sinh(operand), d_operand)
            else:
                # For other unary ops, fall back to numerical or raise
                raise UnknownOperatorError(
                    operator=current.op,
                    context="iterative gradient computation (unary)",
                )
            continue

        # N-ary operations
        if isinstance(current, NarySum):
            if phase == 0:
                # Check if all children computed
                all_done = all(id(t) in results for t in current.terms)
                if not all_done:
                    stack.append((current, 1, []))
                    for t in reversed(current.terms):
                        if id(t) not in results:
                            stack.append((t, 0, []))
                    continue
            # Sum of gradients
            grad: Expression = Constant(0.0)
            for t in current.terms:
                grad = _simplify_add(grad, results[id(t)])
            results[node_id] = grad
            continue

        if isinstance(current, NaryProduct):
            if phase == 0:
                all_done = all(id(f) in results for f in current.factors)
                if not all_done:
                    stack.append((current, 1, []))
                    for f in reversed(current.factors):
                        if id(f) not in results:
                            stack.append((f, 0, []))
                    continue
            # Generalized product rule
            factors = current.factors
            grad = Constant(0.0)
            for i, factor in enumerate(factors):
                d_factor = results[id(factor)]
                other_product: Expression = Constant(1.0)
                for j, other in enumerate(factors):
                    if j != i:
                        other_product = _simplify_mul(other_product, other)
                grad = _simplify_add(grad, _simplify_mul(d_factor, other_product))
            results[node_id] = grad
            continue

        raise InvalidExpressionError(
            expr_type=type(current),
            context="iterative gradient computation",
            suggestion="Use Variable, Constant, BinaryOp, or UnaryOp expressions.",
        )

    return results.get(id(expr), Constant(0.0))


# =============================================================================
# Registered Gradient Rules for Vector Expressions
# =============================================================================


# Import vector types and register gradient rules after module initialization
def _register_vector_gradient_rules() -> None:
    """Register gradient rules for vector expression types.

    This function is called at module load time to register O(1) gradient
    rules for VectorSum, LinearCombination, DotProduct, L2Norm, L1Norm,
    and QuadraticForm.
    """
    from optyx.core.expressions import Constant
    from optyx.core.vectors import (
        DotProduct,
        L1Norm,
        L2Norm,
        LinearCombination,
        VectorSum,
        VectorVariable,
        VectorPowerSum,
        VectorUnarySum,
        VectorExpressionSum,
    )
    from optyx.core.matrices import QuadraticForm

    @register_gradient(LinearCombination)
    def gradient_linear_combination(
        expr: LinearCombination, wrt: Variable
    ) -> Expression:
        """O(1) gradient for linear combination: ∂(c·x)/∂x_i = c_i.

        For a linear combination c @ x = Σ c_i * x_i:
        - If wrt is x_j, gradient is c_j
        - If wrt is not in x, gradient is 0

        This is a dictionary lookup, O(1) regardless of vector size.
        """
        coeffs = expr.coefficients
        vec = expr.vector

        if isinstance(vec, VectorVariable):
            # O(1) lookup: find coefficient for wrt variable
            for i, var in enumerate(vec._variables):
                if var.name == wrt.name:
                    return Constant(float(coeffs[i]))
            return Constant(0.0)
        else:
            # VectorExpression: need to differentiate each element
            result: Expression = Constant(0.0)
            for i, elem in enumerate(vec._expressions):
                d_elem = gradient(elem, wrt)
                result = _simplify_add(
                    result, _simplify_mul(Constant(float(coeffs[i])), d_elem)
                )
            return result

    @register_gradient(VectorSum)
    def gradient_vector_sum(expr: VectorSum, wrt: Variable) -> Expression:
        """O(1) gradient for vector sum: ∂(Σx_i)/∂x_j = 1 if j in x else 0.

        For sum(x) = x_0 + x_1 + ... + x_{n-1}:
        - If wrt is x_j for some j, gradient is 1
        - If wrt is not in x, gradient is 0

        This is an O(n) lookup in the worst case, but typically O(1) for
        early matches.
        """
        vec = expr.vector
        for var in vec._variables:
            if var.name == wrt.name:
                return Constant(1.0)
        return Constant(0.0)

    @register_gradient(VectorExpressionSum)
    def gradient_vector_expression_sum(
        expr: VectorExpressionSum, wrt: Variable
    ) -> Expression:
        """Gradient for VectorExpressionSum: ∂(Σf_i)/∂x = Σ(∂f_i/∂x).

        For sum(expr) = f_0 + f_1 + ... + f_{n-1}:
        The gradient is the sum of the gradients of each element.
        """
        result: Expression = Constant(0.0)
        for elem in expr.expression._expressions:
            d_elem = gradient(elem, wrt)
            result = _simplify_add(result, d_elem)
        return result

    @register_gradient(DotProduct)
    def gradient_dot_product(expr: DotProduct, wrt: Variable) -> Expression:
        """Gradient for dot product: ∂(u·v)/∂x.

        For u · v = Σ u_i * v_i:
        - Uses product rule: ∂(u·v)/∂x = Σ (u_i * ∂v_i/∂x + v_i * ∂u_i/∂x)
        - For VectorVariable dot VectorVariable, this simplifies to O(1)
          when wrt is in exactly one of the vectors.

        Special cases:
        - c · x where c is constant: ∂/∂x_i = c_i
        - x · x: ∂/∂x_i = 2*x_i
        - x · y: ∂/∂x_i = y_i, ∂/∂y_i = x_i
        """
        left = expr.left
        right = expr.right

        # Get elements
        left_elems = (
            left._variables if isinstance(left, VectorVariable) else left._expressions
        )
        right_elems = (
            right._variables
            if isinstance(right, VectorVariable)
            else right._expressions
        )

        # Check if wrt is in left or right vectors
        left_index = None
        right_index = None

        if isinstance(left, VectorVariable):
            for i, var in enumerate(left._variables):
                if var.name == wrt.name:
                    left_index = i
                    break

        if isinstance(right, VectorVariable):
            for i, var in enumerate(right._variables):
                if var.name == wrt.name:
                    right_index = i
                    break

        # Fast paths for VectorVariable
        if isinstance(left, VectorVariable) and isinstance(right, VectorVariable):
            if left_index is not None and right_index is not None:
                # wrt appears in both: x · x case or overlapping vectors
                # ∂(x·x)/∂x_i = 2*x_i
                if left is right or left.name == right.name:
                    return _simplify_mul(Constant(2.0), wrt)
                else:
                    # Different vectors with same variable name? Sum contributions
                    return _simplify_add(
                        right_elems[left_index], left_elems[right_index]
                    )
            elif left_index is not None:
                # wrt only in left: ∂(x·c)/∂x_i = c_i
                return right_elems[left_index]
            elif right_index is not None:
                # wrt only in right: ∂(c·y)/∂y_i = c_i
                return left_elems[right_index]
            else:
                # wrt not in either vector
                return Constant(0.0)

        # General case: iterate through elements with product rule
        result: Expression = Constant(0.0)
        for l_elem, r_elem in zip(left_elems, right_elems):
            dl = gradient(l_elem, wrt)
            dr = gradient(r_elem, wrt)
            term = _simplify_add(_simplify_mul(l_elem, dr), _simplify_mul(r_elem, dl))
            result = _simplify_add(result, term)
        return result

    @register_gradient(L2Norm)
    def gradient_l2_norm(expr: L2Norm, wrt: Variable) -> Expression:
        """Gradient for L2 norm: ∂||x||/∂x_i = x_i / ||x||.

        For ||x|| = sqrt(x[0]² + x[1]² + ... + x[n-1]²):
        - If wrt is x_j, gradient is x_j / ||x||
        - If wrt is not in x, gradient is 0

        This is O(1) for VectorVariable (just lookup and divide).
        """
        vec = expr.vector

        # Find if wrt is in the vector
        if isinstance(vec, VectorVariable):
            for var in vec._variables:
                if var.name == wrt.name:
                    # ∂||x||/∂x_i = x_i / ||x||
                    return _simplify_div(wrt, expr)
            return Constant(0.0)
        else:
            # VectorExpression: need chain rule
            # ∂||f(x)||/∂x = Σ (f_i / ||f||) * ∂f_i/∂x
            result: Expression = Constant(0.0)
            for elem in vec._expressions:
                d_elem = gradient(elem, wrt)
                term = _simplify_mul(_simplify_div(elem, expr), d_elem)
                result = _simplify_add(result, term)
            return result

    @register_gradient(L1Norm)
    def gradient_l1_norm(expr: L1Norm, wrt: Variable) -> Expression:
        """Gradient for L1 norm: ∂||x||₁/∂x_i = sign(x_i).

        For ||x||₁ = |x[0]| + |x[1]| + ... + |x[n-1]|:
        - If wrt is x_j, gradient is sign(x_j) = x_j / |x_j|
        - If wrt is not in x, gradient is 0

        Note: Not differentiable at x_i = 0, but we use x/|x| which
        matches the subgradient convention.
        """
        from optyx.core.functions import abs_

        vec = expr.vector

        # Find if wrt is in the vector
        if isinstance(vec, VectorVariable):
            for var in vec._variables:
                if var.name == wrt.name:
                    # ∂|x_i|/∂x_i = sign(x_i) = x_i / |x_i|
                    return _simplify_div(wrt, abs_(wrt))
            return Constant(0.0)
        else:
            # VectorExpression: need chain rule
            # ∂||f(x)||₁/∂x = Σ sign(f_i) * ∂f_i/∂x
            result: Expression = Constant(0.0)
            for elem in vec._expressions:
                d_elem = gradient(elem, wrt)
                sign_elem = _simplify_div(elem, abs_(elem))
                term = _simplify_mul(sign_elem, d_elem)
                result = _simplify_add(result, term)
            return result

    @register_gradient(QuadraticForm)
    def gradient_quadratic_form(expr: QuadraticForm, wrt: Variable) -> Expression:
        """Gradient for quadratic form: ∂(x'Qx)/∂x_i = [(Q + Q')x]_i.

        For x'Qx where Q is a constant matrix:
        - ∇(x'Qx) = (Q + Q')x
        - If wrt is x_i, gradient is the i-th element of (Q + Q')x
        - If wrt is not in x, gradient is 0

        For symmetric Q (Q = Q'), this simplifies to 2Qx.
        """
        vec = expr.vector
        Q = expr.matrix

        # Compute Q + Q' (symmetric part times 2)
        Q_sym = Q + Q.T

        # Find if wrt is in the vector
        if isinstance(vec, VectorVariable):
            for i, var in enumerate(vec._variables):
                if var.name == wrt.name:
                    # ∂(x'Qx)/∂x_i = [(Q + Q')x]_i = Σ_j (Q + Q')_{ij} * x_j
                    # This is a LinearCombination with coefficients from row i of Q_sym
                    row_coeffs = Q_sym[i, :]
                    return LinearCombination(row_coeffs, vec)
            return Constant(0.0)
        else:
            # VectorExpression: need chain rule
            # ∂(f'Qf)/∂x = Σ_i [(Q + Q')f]_i * ∂f_i/∂x
            result: Expression = Constant(0.0)
            elems = list(vec._expressions)
            for i, elem in enumerate(elems):
                d_elem = gradient(elem, wrt)
                # [(Q + Q')f]_i = Σ_j Q_sym[i,j] * f_j
                qf_i: Expression = Constant(0.0)
                for j, elem_j in enumerate(elems):
                    coeff = Q_sym[i, j]
                    if coeff != 0:
                        qf_i = _simplify_add(
                            qf_i, _simplify_mul(Constant(coeff), elem_j)
                        )
                term = _simplify_mul(qf_i, d_elem)
                result = _simplify_add(result, term)
            return result

    @register_gradient(VectorPowerSum)
    def gradient_vector_power_sum(expr: VectorPowerSum, wrt: Variable) -> Expression:
        """O(1) gradient for vector power sum: ∂(Σx_i^k)/∂x_j = k * x_j^(k-1).

        For sum(x ** k) = x_0^k + x_1^k + ... + x_{n-1}^k:
        - If wrt is x_j for some j, gradient is k * x_j^(k-1)
        - If wrt is not in x, gradient is 0
        """
        from optyx.core.expressions import BinaryOp

        vec = expr.vector
        k = expr.power

        for var in vec._variables:
            if var.name == wrt.name:
                if k == 1:
                    return Constant(1.0)
                elif k == 2:
                    return BinaryOp(Constant(2.0), var, "*")
                else:
                    # k * x^(k-1)
                    power_term = BinaryOp(var, Constant(k - 1), "**")
                    return BinaryOp(Constant(k), power_term, "*")
        return Constant(0.0)

    @register_gradient(VectorUnarySum)
    def gradient_vector_unary_sum(expr: VectorUnarySum, wrt: Variable) -> Expression:
        """O(1) gradient for vector unary sum: ∂(Σf(x_i))/∂x_j = f'(x_j).

        For sum(f(x)) = f(x_0) + f(x_1) + ... + f(x_{n-1}):
        - If wrt is x_j for some j, gradient is f'(x_j)
        - If wrt is not in x, gradient is 0

        Derivative formulas:
        - sin(x) -> cos(x)
        - cos(x) -> -sin(x)
        - exp(x) -> exp(x)
        - log(x) -> 1/x
        - sqrt(x) -> 1/(2*sqrt(x))
        - tanh(x) -> 1 - tanh(x)^2
        """
        from optyx.core.expressions import BinaryOp, UnaryOp

        vec = expr.vector
        op = expr.op

        for var in vec._variables:
            if var.name == wrt.name:
                # Return derivative based on op
                if op == "sin":
                    return UnaryOp(var, "cos")
                elif op == "cos":
                    return BinaryOp(Constant(-1.0), UnaryOp(var, "sin"), "*")
                elif op == "exp":
                    return UnaryOp(var, "exp")
                elif op == "log":
                    return BinaryOp(Constant(1.0), var, "/")
                elif op == "sqrt":
                    sqrt_x = UnaryOp(var, "sqrt")
                    two_sqrt = BinaryOp(Constant(2.0), sqrt_x, "*")
                    return BinaryOp(Constant(1.0), two_sqrt, "/")
                elif op == "sinh":
                    return UnaryOp(var, "cosh")
                elif op == "cosh":
                    return UnaryOp(var, "sinh")
                elif op == "tanh":
                    tanh_x = UnaryOp(var, "tanh")
                    tanh_sq = BinaryOp(tanh_x, Constant(2.0), "**")
                    return BinaryOp(Constant(1.0), tanh_sq, "-")
                elif op == "tan":
                    cos_x = UnaryOp(var, "cos")
                    cos_sq = BinaryOp(cos_x, Constant(2.0), "**")
                    return BinaryOp(Constant(1.0), cos_sq, "/")
                elif op == "abs":
                    abs_x = UnaryOp(var, "abs")
                    return BinaryOp(var, abs_x, "/")
                else:
                    # Fallback - shouldn't happen for supported ops
                    raise NotImplementedError(f"Gradient not implemented for op: {op}")
        return Constant(0.0)


# Register vector gradient rules at module load time
_register_vector_gradient_rules()


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

    Uses O(1) vectorized jacobian_row() methods when available for vector
    expressions (VectorSum, DotProduct, LinearCombination, QuadraticForm).
    Falls back to individual gradient() calls otherwise.

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
    result: list[list[Expression]] = []
    for expr in exprs:
        # Try vectorized jacobian_row if available
        if hasattr(expr, "jacobian_row"):
            row = expr.jacobian_row(variables)
            if row is not None:
                result.append(row)
                continue
        # Fall back to individual gradient calls
        result.append([gradient(expr, var) for var in variables])
    return result


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


def _is_scaled_variable_pattern(
    jacobian_row: list[Expression],
    variables: list[Variable],
) -> tuple[NDArray[Any] | float | int, bool] | None:
    """Check if a Jacobian row is c*x[i] for all variables.

    For DotProduct(x, x), gradient is 2*x[i] for each x[i].
    We can compile this as: lambda x: c * x

    Returns:
        (scale, match) tuple if all elements are c*var[i], None otherwise.
    """
    from optyx.core.expressions import Constant, BinaryOp

    if len(jacobian_row) != len(variables):
        return None

    scale = None
    for i, (expr, var) in enumerate(zip(jacobian_row, variables)):
        # Check for pattern: Constant(c) * Variable(var)
        if isinstance(expr, BinaryOp) and expr.op == "*":
            if isinstance(expr.left, Constant) and expr.right is var:
                c = expr.left.value
            elif isinstance(expr.right, Constant) and expr.left is var:
                c = expr.right.value
            else:
                return None

            if scale is None:
                scale = c
            elif scale != c:
                return None  # Different scales, not uniform
        else:
            return None

    return (scale, True) if scale is not None else None


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
        - For VectorPowerSum/VectorUnarySum, uses O(1) numpy vectorized gradients.
        - For linear expressions where all Jacobian elements are constants,
          returns a pre-computed array directly (9.7x speedup vs element-by-element).
        - For DotProduct(x, x) pattern (gradient = c*x), uses vectorized NumPy.
    """
    import numpy as np
    from optyx.core.compiler import (
        compile_expression,
        _sanitize_derivatives,
        _compile_vectorized_power_gradient,
        _compile_vectorized_unary_gradient,
    )
    from optyx.core.expressions import Constant
    from optyx.core.vectors import VectorPowerSum, VectorUnarySum

    m = len(exprs)
    n = len(variables)

    # Fast path 0: Single VectorPowerSum or VectorUnarySum - use vectorized gradient
    if m == 1:
        expr = exprs[0]
        if isinstance(expr, VectorPowerSum):
            grad_fn = _compile_vectorized_power_gradient(expr, variables)

            def power_jacobian_fn(x):
                return grad_fn(x).reshape(1, -1)

            return power_jacobian_fn

        if isinstance(expr, VectorUnarySum):
            grad_fn = _compile_vectorized_unary_gradient(expr, variables)

            def unary_jacobian_fn(x):
                return grad_fn(x).reshape(1, -1)

            return unary_jacobian_fn

    jacobian_exprs = compute_jacobian(exprs, variables)

    # Fast path 1: All Jacobian elements are constants - pre-compute once
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

    # Fast path 2: Single row with c*x[i] pattern (e.g., gradient of x.dot(x) = 2*x)
    if m == 1:
        pattern = _is_scaled_variable_pattern(jacobian_exprs[0], variables)
        if pattern is not None:
            scale, _ = pattern

            def scaled_variable_jacobian_fn(x):
                return (scale * x).reshape(1, -1)

            return scaled_variable_jacobian_fn

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

    Performance:
        For VectorPowerSum and VectorUnarySum, the Hessian is diagonal,
        so we use O(n) vectorized computation instead of O(n²).
    """
    import numpy as np
    from optyx.core.compiler import compile_expression, _sanitize_derivatives
    from optyx.core.vectors import VectorPowerSum, VectorUnarySum

    n = len(variables)

    # Fast path: VectorPowerSum has diagonal Hessian
    # For sum(x**k), H[i,i] = k*(k-1)*x[i]^(k-2), H[i,j] = 0 for i != j
    if isinstance(expr, VectorPowerSum):
        k = expr.power
        var_name_to_idx = {v.name: i for i, v in enumerate(variables)}
        vector_vars = expr.vector._variables
        indices = np.array(
            [var_name_to_idx[v.name] for v in vector_vars], dtype=np.intp
        )
        is_full = len(indices) == n and np.array_equal(indices, np.arange(n))

        if k == 1:
            # d²/dx² (sum x) = 0
            zeros = np.zeros((n, n))

            def hess_power_k1(x):
                return zeros

            return hess_power_k1
        elif k == 2:
            # d²/dx² (sum x²) = 2 on diagonal
            hess = np.diag(np.full(n, 2.0)) if is_full else np.zeros((n, n))
            if not is_full:
                for idx in indices:
                    hess[idx, idx] = 2.0

            def hess_power_k2(x):
                return hess

            return hess_power_k2
        else:
            # d²/dx² (sum x^k) = k*(k-1)*x^(k-2) on diagonal
            coeff = k * (k - 1)
            exp = k - 2

            if is_full:

                def hess_power_general(x):
                    diag = coeff * np.power(x, exp)
                    return np.diag(_sanitize_derivatives(diag))

                return hess_power_general
            else:

                def hess_power_sparse(x):
                    result = np.zeros((n, n))
                    diag_vals = coeff * np.power(x[indices], exp)
                    result[indices, indices] = diag_vals
                    return _sanitize_derivatives(result)

                return hess_power_sparse

    # Fast path: VectorUnarySum has diagonal Hessian
    # For sum(f(x)), H[i,i] = f''(x[i]), H[i,j] = 0 for i != j
    if isinstance(expr, VectorUnarySum):
        op = expr.op
        var_name_to_idx = {v.name: i for i, v in enumerate(variables)}
        vector_vars = expr.vector._variables
        indices = np.array(
            [var_name_to_idx[v.name] for v in vector_vars], dtype=np.intp
        )
        is_full = len(indices) == n and np.array_equal(indices, np.arange(n))

        # Second derivatives
        if op == "sin":
            # d²/dx² sin(x) = -sin(x)
            if is_full:

                def hess_sin(x):
                    return np.diag(-np.sin(x))

                return hess_sin
            else:

                def hess_sin_sparse(x):
                    result = np.zeros((n, n))
                    result[indices, indices] = -np.sin(x[indices])
                    return result

                return hess_sin_sparse

        elif op == "cos":
            # d²/dx² cos(x) = -cos(x)
            if is_full:

                def hess_cos(x):
                    return np.diag(-np.cos(x))

                return hess_cos
            else:

                def hess_cos_sparse(x):
                    result = np.zeros((n, n))
                    result[indices, indices] = -np.cos(x[indices])
                    return result

                return hess_cos_sparse

        elif op == "exp":
            # d²/dx² exp(x) = exp(x)
            if is_full:

                def hess_exp(x):
                    return np.diag(np.exp(x))

                return hess_exp
            else:

                def hess_exp_sparse(x):
                    result = np.zeros((n, n))
                    result[indices, indices] = np.exp(x[indices])
                    return result

                return hess_exp_sparse

        elif op == "log":
            # d²/dx² log(x) = -1/x²
            if is_full:

                def hess_log(x):
                    return np.diag(_sanitize_derivatives(-1.0 / (x**2)))

                return hess_log
            else:

                def hess_log_sparse(x):
                    result = np.zeros((n, n))
                    result[indices, indices] = -1.0 / (x[indices] ** 2)
                    return _sanitize_derivatives(result)

                return hess_log_sparse

        # Fall through to general path for other ops

    hessian_exprs = compute_hessian(expr, variables)

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
