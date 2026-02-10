"""Expression compiler for fast evaluation.

Compiles expression trees into optimized callables that minimize
Python overhead during repeated evaluations (e.g., in optimization loops).

Performance optimizations:
- Closure-based evaluation avoids dictionary lookups
- LRU cache prevents recompilation of identical expressions
- Iterative compilation for deep trees avoids recursion limits
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from optyx.core.errors import UnknownOperatorError, InvalidExpressionError
from optyx.core.expressions import NarySum, NaryProduct

# Large but finite value to replace infinities in gradients.
# This prevents solver crashes while maintaining gradient direction.
_LARGE_GRADIENT = 1e16

# Recursion threshold - use iterative for deep trees
_RECURSION_THRESHOLD = 400


def _sanitize_derivatives(arr: np.ndarray) -> np.ndarray:
    """Replace NaN and Inf values in derivative arrays.

    This handles singularities that occur at points like x=0 for:
    - abs(x): derivative is x/|x|, which is 0/0 = NaN at x=0
    - sqrt(x): derivative is 1/(2*sqrt(x)), which is Inf at x=0
    - log(x): derivative is 1/x, which is Inf at x=0

    The replacement strategy:
    - NaN → 0.0 (e.g., for abs(0), use subgradient 0)
    - +Inf → +1e16 (large but finite, preserves direction)
    - -Inf → -1e16 (large but finite, preserves direction)

    This allows solvers to continue without crashing, though users
    should avoid regions where these singularities occur if possible.

    Performance: For linear expressions (constant gradients), this check
    short-circuits and avoids the expensive nan_to_num call (3.2x speedup).
    """
    # Fast path: skip sanitization if all values are finite
    if np.all(np.isfinite(arr)):
        return arr
    return np.nan_to_num(arr, nan=0.0, posinf=_LARGE_GRADIENT, neginf=-_LARGE_GRADIENT)


if TYPE_CHECKING:
    from numpy.typing import NDArray

    from optyx.core.expressions import Expression, Variable
    from optyx.core.vectors import VectorPowerSum, VectorUnarySum


def compile_expression(
    expr: Expression,
    variables: list[Variable],
) -> Callable[[NDArray[np.floating]], NDArray[np.floating] | np.floating | float]:
    """Compile an expression tree into a fast callable.

    The returned function takes a 1D numpy array of variable values
    (in the order specified by `variables`) and returns the expression value.

    Args:
        expr: The expression to compile.
        variables: Ordered list of variables. The compiled function will
            expect values in this order.

    Returns:
        A callable that evaluates the expression given variable values as an array.

    Example:
        >>> x = Variable("x")
        >>> y = Variable("y")
        >>> expr = x**2 + y**2
        >>> f = compile_expression(expr, [x, y])
        >>> f(np.array([3.0, 4.0]))  # Returns 25.0
    """
    # Create mapping from variable name to array index
    var_indices = {var.name: i for i, var in enumerate(variables)}

    # Generate and cache the compiled function
    return _compile_cached(
        expr, tuple(var.name for var in variables), tuple(var_indices.items())
    )


@lru_cache(maxsize=1024)
def _compile_cached(
    expr: Expression,
    var_names: tuple[str, ...],
    var_indices_items: tuple[tuple[str, int], ...],
) -> Callable[[NDArray[np.floating]], NDArray[np.floating] | np.floating | float]:
    """Cached compilation of expressions.

    Uses LRU cache to avoid recompiling the same expression.
    Switches to iterative compilation for deep expression trees.
    """
    var_indices = dict(var_indices_items)

    # Check tree depth
    depth = _estimate_tree_depth(expr)
    if depth >= _RECURSION_THRESHOLD:
        eval_func = _build_evaluator_iterative(expr, var_indices)
    else:
        eval_func = _build_evaluator(expr, var_indices)
    return eval_func


def _estimate_tree_depth(expr: Expression) -> int:
    """Estimate depth of expression tree following left spine."""
    from optyx.core.expressions import BinaryOp, Constant, UnaryOp, Variable
    from optyx.core.vectors import (
        LinearCombination,
        VectorSum,
        DotProduct,
        VectorExpressionSum,
    )

    depth = 0
    current = expr
    while True:
        if isinstance(current, (Constant, Variable)):
            break
        elif isinstance(current, BinaryOp):
            depth += 1
            current = current.left
        elif isinstance(current, UnaryOp):
            depth += 1
            current = current.operand
        elif isinstance(current, (LinearCombination, VectorSum, VectorExpressionSum)):
            break  # These don't recurse deeply
        elif isinstance(current, DotProduct):
            depth += 1
            current = current.left
        else:
            break
    return depth


def _build_evaluator(
    expr: Expression,
    var_indices: dict[str, int],
) -> Callable[[NDArray[np.floating]], NDArray[np.floating] | np.floating | float]:
    """Recursively build an evaluator function for an expression.

    This approach avoids dictionary lookups during evaluation by
    pre-computing array indices and creating closures.
    """
    from optyx.core.expressions import BinaryOp, Constant, UnaryOp, Variable
    from optyx.core.parameters import Parameter
    from optyx.core.vectors import (
        DotProduct,
        L1Norm,
        L2Norm,
        LinearCombination,
        VectorSum,
        VectorVariable,
        ElementwisePower,
        VectorPowerSum,
        ElementwiseUnary,
        VectorUnarySum,
        VectorExpressionSum,
    )
    from optyx.core.matrices import QuadraticForm

    if isinstance(expr, Constant):
        value = expr.value
        return lambda x: value

    elif isinstance(expr, Parameter):
        # Parameters evaluate to their current value at call time
        # We capture the parameter object, not its value, for mutability
        param = expr
        return lambda x, p=param: p.value

    elif isinstance(expr, Variable):
        idx = var_indices[expr.name]
        return lambda x, i=idx: x[i]

    elif isinstance(expr, LinearCombination):
        # c @ x = c[0]*x[0] + c[1]*x[1] + ... - efficient numpy implementation
        coeffs = np.asarray(expr.coefficients)
        if isinstance(expr.vector, VectorVariable):
            indices = np.array([var_indices[v.name] for v in expr.vector._variables])
            return lambda x, c=coeffs, idx=indices: np.dot(c, x[idx])
        else:
            # VectorExpression/VectorBinaryOp - use vector evaluator
            vec_fn = _build_vector_evaluator(expr.vector, var_indices)
            return lambda x, c=coeffs, vf=vec_fn: np.dot(c, vf(x))

    elif isinstance(expr, VectorSum):
        # sum(x) = x[0] + x[1] + ... - efficient numpy implementation
        indices = np.array([var_indices[v.name] for v in expr.vector._variables])
        return lambda x, idx=indices: np.sum(x[idx])

    elif isinstance(expr, VectorExpressionSum):
        # sum(expr) where expr is a VectorExpression
        # Fast path for VectorBinaryOp: single numpy op + sum
        from optyx.core.vectors import VectorBinaryOp

        if isinstance(expr.expression, VectorBinaryOp):
            vec_fn = _build_vector_evaluator(expr.expression, var_indices)
            return lambda x, vf=vec_fn: float(np.sum(vf(x)))
        elem_fns = [
            _build_evaluator(e, var_indices) for e in expr.expression._expressions
        ]
        return lambda x, fns=elem_fns: float(sum(f(x) for f in fns))

    elif isinstance(expr, DotProduct):
        # x · y = x[0]*y[0] + x[1]*y[1] + ...
        left_fn = _build_vector_evaluator(expr.left, var_indices)
        right_fn = _build_vector_evaluator(expr.right, var_indices)
        return lambda x, lf=left_fn, rf=right_fn: np.dot(lf(x), rf(x))

    elif isinstance(expr, L2Norm):
        # ||x|| = sqrt(x[0]^2 + x[1]^2 + ...)
        vec_fn = _build_vector_evaluator(expr.vector, var_indices)
        return lambda x, vf=vec_fn: np.linalg.norm(vf(x))

    elif isinstance(expr, L1Norm):
        # ||x||_1 = |x[0]| + |x[1]| + ...
        vec_fn = _build_vector_evaluator(expr.vector, var_indices)
        return lambda x, vf=vec_fn: np.sum(np.abs(vf(x)))

    elif isinstance(expr, QuadraticForm):
        # x' @ Q @ x
        Q = expr.matrix
        vec_fn = _build_vector_evaluator(expr.vector, var_indices)
        return lambda x, vf=vec_fn, Q=Q: float(vf(x) @ Q @ vf(x))

    elif isinstance(expr, VectorPowerSum):
        # sum(x ** k) - efficient numpy implementation
        indices = np.array([var_indices[v.name] for v in expr.vector._variables])
        power = expr.power
        return lambda x, idx=indices, k=power: float(np.sum(x[idx] ** k))

    elif isinstance(expr, VectorUnarySum):
        # sum(f(x)) - efficient numpy implementation
        indices = np.array([var_indices[v.name] for v in expr.vector._variables])
        op = expr.op
        numpy_func = VectorUnarySum._NUMPY_FUNCS[op]
        return lambda x, idx=indices, f=numpy_func: float(np.sum(f(x[idx])))

    elif isinstance(expr, ElementwisePower):
        # x ** k element-wise - returns array
        indices = np.array([var_indices[v.name] for v in expr.vector._variables])
        power = expr.power
        return lambda x, idx=indices, k=power: x[idx] ** k

    elif isinstance(expr, ElementwiseUnary):
        # f(x) element-wise - returns array
        indices = np.array([var_indices[v.name] for v in expr.vector._variables])
        op = expr.op
        numpy_func = ElementwiseUnary._NUMPY_FUNCS[op]
        return lambda x, idx=indices, f=numpy_func: f(x[idx])

    elif isinstance(expr, BinaryOp):
        left_fn = _build_evaluator(expr.left, var_indices)
        right_fn = _build_evaluator(expr.right, var_indices)
        op = expr.op

        if op == "+":
            return lambda x, lf=left_fn, rf=right_fn: lf(x) + rf(x)
        elif op == "-":
            return lambda x, lf=left_fn, rf=right_fn: lf(x) - rf(x)
        elif op == "*":
            return lambda x, lf=left_fn, rf=right_fn: lf(x) * rf(x)
        elif op == "/":
            return lambda x, lf=left_fn, rf=right_fn: lf(x) / rf(x)
        elif op == "**":
            return lambda x, lf=left_fn, rf=right_fn: lf(x) ** rf(x)
        else:
            raise UnknownOperatorError(
                operator=op,
                context="expression compilation",
            )

    elif isinstance(expr, UnaryOp):
        operand_fn = _build_evaluator(expr.operand, var_indices)
        numpy_func = expr._numpy_func
        return lambda x, f=operand_fn, np_f=numpy_func: np_f(f(x))

    elif isinstance(expr, NarySum):
        term_fns = tuple(_build_evaluator(t, var_indices) for t in expr.terms)
        return lambda x, fns=term_fns: sum(fn(x) for fn in fns)

    elif isinstance(expr, NaryProduct):
        factor_fns = tuple(_build_evaluator(f, var_indices) for f in expr.factors)

        def _eval_product(x: NDArray, fns: tuple = factor_fns) -> float:
            result = 1.0
            for fn in fns:
                result = result * fn(x)
            return result

        return _eval_product

    else:
        raise InvalidExpressionError(
            expr_type=type(expr),
            context="expression compilation",
            suggestion="Use Variable, Constant, BinaryOp, or UnaryOp expressions.",
        )


def _build_vector_evaluator(
    vec: Any,
    var_indices: dict[str, int],
) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
    """Build an evaluator for a vector (returns array of values)."""
    from optyx.core.vectors import VectorBinaryOp, VectorExpression, VectorVariable

    if isinstance(vec, VectorVariable):
        indices = np.array([var_indices[v.name] for v in vec._variables])
        return lambda x, idx=indices: x[idx]
    elif isinstance(vec, VectorBinaryOp):
        # Single numpy op instead of N per-element evaluations
        left_fn = _build_vector_evaluator(vec.left, var_indices)
        right_fn = _build_vector_evaluator(vec.right, var_indices)
        np_op = vec._NUMPY_OPS[vec.op]
        return lambda x, lf=left_fn, rf=right_fn, op=np_op: op(lf(x), rf(x))
    elif isinstance(vec, VectorExpression):
        elem_fns = [_build_evaluator(e, var_indices) for e in vec._expressions]
        return lambda x, fns=elem_fns: np.array([f(x) for f in fns])
    elif isinstance(vec, (int, float)):
        val = np.array([float(vec)])
        return lambda x, v=val: v
    else:
        raise InvalidExpressionError(
            expr_type=type(vec),
            context="vector expression compilation",
            suggestion="Use VectorVariable or VectorExpression.",
        )


def _build_evaluator_iterative(
    expr: Expression,
    var_indices: dict[str, int],
) -> Callable[[NDArray[np.floating]], NDArray[np.floating] | np.floating | float]:
    """Build evaluator using iterative post-order traversal.

    Handles deep expression trees that would cause RecursionError.
    Uses explicit stack to build closures bottom-up.
    """
    from optyx.core.expressions import (
        BinaryOp,
        Constant,
        NaryProduct,
        NarySum,
        UnaryOp,
        Variable,
    )
    from optyx.core.parameters import Parameter
    from optyx.core.vectors import (
        DotProduct,
        L1Norm,
        L2Norm,
        LinearCombination,
        VectorSum,
        VectorVariable,
        VectorExpressionSum,
    )
    from optyx.core.matrices import QuadraticForm

    # Stack for iterative traversal: (expression, phase, children_fns)
    # phase 0: first visit, phase 1: children processed
    stack: list[tuple[Any, int, list]] = [(expr, 0, [])]
    result_stack: list[Callable] = []

    while stack:
        node, phase, children_fns = stack.pop()

        # Leaf nodes - return immediately
        if isinstance(node, Constant):
            value = node.value
            result_stack.append(lambda x, v=value: v)
            continue

        if isinstance(node, Parameter):
            param = node
            result_stack.append(lambda x, p=param: p.value)
            continue

        if isinstance(node, Variable):
            idx = var_indices[node.name]
            result_stack.append(lambda x, i=idx: x[i])
            continue

        # Vector expressions - O(n) but not recursive
        if isinstance(node, LinearCombination):
            coeffs = np.asarray(node.coefficients)
            if isinstance(node.vector, VectorVariable):
                indices = np.array(
                    [var_indices[v.name] for v in node.vector._variables]
                )
                result_stack.append(lambda x, c=coeffs, idx=indices: np.dot(c, x[idx]))
            else:
                # VectorExpression/VectorBinaryOp - use vector evaluator
                vec_fn = _build_vector_evaluator(node.vector, var_indices)
                result_stack.append(lambda x, c=coeffs, vf=vec_fn: np.dot(c, vf(x)))
            continue

        if isinstance(node, VectorSum):
            indices = np.array([var_indices[v.name] for v in node.vector._variables])
            result_stack.append(lambda x, idx=indices: np.sum(x[idx]))
            continue

        if isinstance(node, VectorExpressionSum):
            # sum(expr) where expr is a VectorExpression - build non-recursively
            # Fast path for VectorBinaryOp: single numpy op + sum
            from optyx.core.vectors import VectorBinaryOp

            if isinstance(node.expression, VectorBinaryOp):
                vec_fn = _build_vector_evaluator(node.expression, var_indices)
                result_stack.append(lambda x, vf=vec_fn: float(np.sum(vf(x))))
                continue
            elem_fns = []
            for e in node.expression._expressions:
                if isinstance(e, Variable):
                    idx = var_indices[e.name]
                    elem_fns.append(lambda x, i=idx: x[i])
                elif isinstance(e, Constant):
                    val = e.value
                    elem_fns.append(lambda x, v=val: v)
                else:
                    elem_fns.append(_build_evaluator(e, var_indices))
            result_stack.append(lambda x, fns=elem_fns: float(sum(f(x) for f in fns)))
            continue

        if isinstance(node, DotProduct):
            left_fn = _build_vector_evaluator(node.left, var_indices)
            right_fn = _build_vector_evaluator(node.right, var_indices)
            result_stack.append(lambda x, lf=left_fn, rf=right_fn: np.dot(lf(x), rf(x)))
            continue

        if isinstance(node, L2Norm):
            vec_fn = _build_vector_evaluator(node.vector, var_indices)
            result_stack.append(lambda x, vf=vec_fn: np.linalg.norm(vf(x)))
            continue

        if isinstance(node, L1Norm):
            vec_fn = _build_vector_evaluator(node.vector, var_indices)
            result_stack.append(lambda x, vf=vec_fn: np.sum(np.abs(vf(x))))
            continue

        if isinstance(node, QuadraticForm):
            Q = node.matrix
            vec_fn = _build_vector_evaluator(node.vector, var_indices)
            result_stack.append(lambda x, vf=vec_fn, Q=Q: float(vf(x) @ Q @ vf(x)))
            continue

        # Binary operation
        if isinstance(node, BinaryOp):
            if phase == 0:
                # First visit - push back with phase 1, then push children
                stack.append((node, 1, []))
                stack.append((node.right, 0, []))
                stack.append((node.left, 0, []))
            else:
                # Phase 1: children are processed, pop their results
                right_fn = result_stack.pop()
                left_fn = result_stack.pop()
                op = node.op

                if op == "+":
                    result_stack.append(
                        lambda x, lf=left_fn, rf=right_fn: lf(x) + rf(x)
                    )
                elif op == "-":
                    result_stack.append(
                        lambda x, lf=left_fn, rf=right_fn: lf(x) - rf(x)
                    )
                elif op == "*":
                    result_stack.append(
                        lambda x, lf=left_fn, rf=right_fn: lf(x) * rf(x)
                    )
                elif op == "/":
                    result_stack.append(
                        lambda x, lf=left_fn, rf=right_fn: lf(x) / rf(x)
                    )
                elif op == "**":
                    result_stack.append(
                        lambda x, lf=left_fn, rf=right_fn: lf(x) ** rf(x)
                    )
                else:
                    raise UnknownOperatorError(
                        operator=op,
                        context="iterative expression compilation",
                    )
            continue

        # Unary operation
        if isinstance(node, UnaryOp):
            if phase == 0:
                stack.append((node, 1, []))
                stack.append((node.operand, 0, []))
            else:
                operand_fn = result_stack.pop()
                numpy_func = node._numpy_func
                result_stack.append(lambda x, f=operand_fn, np_f=numpy_func: np_f(f(x)))
            continue

        # N-ary expressions - flat children, compile each directly
        if isinstance(node, NarySum):
            term_fns = tuple(_build_evaluator(t, var_indices) for t in node.terms)
            result_stack.append(lambda x, fns=term_fns: sum(fn(x) for fn in fns))
            continue

        if isinstance(node, NaryProduct):
            factor_fns = tuple(_build_evaluator(f, var_indices) for f in node.factors)

            def _eval_product_iter(x: NDArray, fns: tuple = factor_fns) -> float:
                result = 1.0
                for fn in fns:
                    result = result * fn(x)
                return result

            result_stack.append(_eval_product_iter)
            continue

        # Unknown type - try to evaluate directly
        raise InvalidExpressionError(
            expr_type=type(node),
            context="iterative expression compilation",
            suggestion="Use Variable, Constant, BinaryOp, or UnaryOp expressions.",
        )

    if not result_stack:
        raise InvalidExpressionError(
            expr_type=type(None),
            context="iterative expression compilation",
            suggestion="Check the expression tree structure - result stack was empty.",
        )
    return result_stack[-1]


def compile_to_dict_function(
    expr: Expression,
    variables: list[Variable],
) -> Callable[
    [dict[str, float | NDArray[np.floating]]],
    NDArray[np.floating] | np.floating | float,
]:
    """Compile an expression to a function that takes a dict of values.

    This is a convenience wrapper that accepts the same dict format
    as `expr.evaluate()` but with compiled performance.

    Args:
        expr: The expression to compile.
        variables: Ordered list of variables.

    Returns:
        A callable that takes a dict mapping variable names to values.
    """
    array_fn = compile_expression(expr, variables)
    var_names = [v.name for v in variables]

    def dict_fn(
        values: dict[str, float | NDArray[np.floating]],
    ) -> NDArray[np.floating] | np.floating | float:
        arr = np.array([values[name] for name in var_names])
        return array_fn(arr)

    return dict_fn


def compile_gradient(
    expr: Expression,
    variables: list[Variable],
) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
    """Compile the gradient of an expression using symbolic differentiation.

    Returns a function that computes the gradient vector at a given point.
    Uses symbolic differentiation via the autodiff module for exact gradients.

    For vectorized expression types (VectorPowerSum, VectorUnarySum),
    generates O(1) numpy-based gradient functions instead of n separate
    compiled expressions.

    Args:
        expr: The expression to differentiate.
        variables: Ordered list of variables.

    Returns:
        A callable that returns the gradient as a 1D array.

    Example:
        >>> x = Variable("x")
        >>> y = Variable("y")
        >>> expr = x**2 + y**2
        >>> grad_fn = compile_gradient(expr, [x, y])
        >>> grad_fn(np.array([3.0, 4.0]))  # Returns [6.0, 8.0]
    """
    from optyx.core.vectors import VectorPowerSum, VectorUnarySum

    # Fast path for VectorPowerSum: gradient is k * x^(k-1), vectorized
    if isinstance(expr, VectorPowerSum):
        return _compile_vectorized_power_gradient(expr, variables)

    # Fast path for VectorUnarySum: gradient is f'(x), vectorized
    if isinstance(expr, VectorUnarySum):
        return _compile_vectorized_unary_gradient(expr, variables)

    # General path: symbolic differentiation
    from optyx.core.autodiff import gradient

    # Compute symbolic gradient for each variable
    grad_exprs = [gradient(expr, var) for var in variables]

    # Compile each gradient expression
    grad_fns = [compile_expression(g, variables) for g in grad_exprs]

    def symbolic_gradient(x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute gradient using symbolic differentiation."""
        raw = np.array([fn(x) for fn in grad_fns])
        return _sanitize_derivatives(raw)

    return symbolic_gradient


def _compile_vectorized_power_gradient(
    expr: "VectorPowerSum",
    variables: list["Variable"],
) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
    """Compile O(1) gradient for VectorPowerSum.

    For sum(x**k), gradient w.r.t. x[i] is k * x[i] ** (k-1).
    This generates a single numpy operation instead of n separate functions.
    """
    k = expr.power
    n = len(variables)

    # Build index mapping: which positions in the gradient correspond to vector vars
    var_name_to_idx = {v.name: i for i, v in enumerate(variables)}
    vector_vars = expr.vector._variables
    indices = np.array([var_name_to_idx[v.name] for v in vector_vars], dtype=np.intp)

    # Check if vector variables form a contiguous block starting at 0
    if len(indices) == n and np.array_equal(indices, np.arange(n)):
        # All variables are the vector - simple case
        if k == 1:
            ones = np.ones(n)

            def grad_power_k1(x: NDArray[np.floating]) -> NDArray[np.floating]:
                return ones

            return grad_power_k1
        elif k == 2:

            def grad_power_k2(x: NDArray[np.floating]) -> NDArray[np.floating]:
                return 2.0 * x

            return grad_power_k2
        else:

            def grad_power_general(x: NDArray[np.floating]) -> NDArray[np.floating]:
                raw = k * np.power(x, k - 1)
                return _sanitize_derivatives(raw)

            return grad_power_general
    else:
        # Sparse case: only some variables are in the vector
        def grad_power_sparse(x: NDArray[np.floating]) -> NDArray[np.floating]:
            result = np.zeros(n)
            result[indices] = k * np.power(x[indices], k - 1)
            return _sanitize_derivatives(result)

        return grad_power_sparse


def _compile_vectorized_unary_gradient(
    expr: "VectorUnarySum",
    variables: list["Variable"],
) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
    """Compile O(1) gradient for VectorUnarySum.

    For sum(f(x)), gradient w.r.t. x[i] is f'(x[i]).
    This generates vectorized numpy operations instead of n separate functions.
    """
    op = expr.op
    n = len(variables)

    # Build index mapping
    var_name_to_idx = {v.name: i for i, v in enumerate(variables)}
    vector_vars = expr.vector._variables
    indices = np.array([var_name_to_idx[v.name] for v in vector_vars], dtype=np.intp)

    # Check if all variables are in the vector
    is_full = len(indices) == n and np.array_equal(indices, np.arange(n))

    # Select derivative function based on operation
    if op == "sin":
        # d/dx sin(x) = cos(x)
        if is_full:

            def grad_sin(x: NDArray[np.floating]) -> NDArray[np.floating]:
                return np.cos(x)

            return grad_sin
        else:

            def grad_sin_sparse(x: NDArray[np.floating]) -> NDArray[np.floating]:
                result = np.zeros(n)
                result[indices] = np.cos(x[indices])
                return result

            return grad_sin_sparse

    elif op == "cos":
        # d/dx cos(x) = -sin(x)
        if is_full:

            def grad_cos(x: NDArray[np.floating]) -> NDArray[np.floating]:
                return -np.sin(x)

            return grad_cos
        else:

            def grad_cos_sparse(x: NDArray[np.floating]) -> NDArray[np.floating]:
                result = np.zeros(n)
                result[indices] = -np.sin(x[indices])
                return result

            return grad_cos_sparse

    elif op == "exp":
        # d/dx exp(x) = exp(x)
        if is_full:

            def grad_exp(x: NDArray[np.floating]) -> NDArray[np.floating]:
                return np.exp(x)

            return grad_exp
        else:

            def grad_exp_sparse(x: NDArray[np.floating]) -> NDArray[np.floating]:
                result = np.zeros(n)
                result[indices] = np.exp(x[indices])
                return result

            return grad_exp_sparse

    elif op == "log":
        # d/dx log(x) = 1/x
        if is_full:

            def grad_log(x: NDArray[np.floating]) -> NDArray[np.floating]:
                raw = 1.0 / x
                return _sanitize_derivatives(raw)

            return grad_log
        else:

            def grad_log_sparse(x: NDArray[np.floating]) -> NDArray[np.floating]:
                result = np.zeros(n)
                result[indices] = 1.0 / x[indices]
                return _sanitize_derivatives(result)

            return grad_log_sparse

    elif op == "sqrt":
        # d/dx sqrt(x) = 1 / (2 * sqrt(x))
        if is_full:

            def grad_sqrt(x: NDArray[np.floating]) -> NDArray[np.floating]:
                raw = 0.5 / np.sqrt(x)
                return _sanitize_derivatives(raw)

            return grad_sqrt
        else:

            def grad_sqrt_sparse(x: NDArray[np.floating]) -> NDArray[np.floating]:
                result = np.zeros(n)
                result[indices] = 0.5 / np.sqrt(x[indices])
                return _sanitize_derivatives(result)

            return grad_sqrt_sparse

    elif op == "sinh":
        # d/dx sinh(x) = cosh(x)
        if is_full:

            def grad_sinh(x: NDArray[np.floating]) -> NDArray[np.floating]:
                return np.cosh(x)

            return grad_sinh
        else:

            def grad_sinh_sparse(x: NDArray[np.floating]) -> NDArray[np.floating]:
                result = np.zeros(n)
                result[indices] = np.cosh(x[indices])
                return result

            return grad_sinh_sparse

    elif op == "cosh":
        # d/dx cosh(x) = sinh(x)
        if is_full:

            def grad_cosh(x: NDArray[np.floating]) -> NDArray[np.floating]:
                return np.sinh(x)

            return grad_cosh
        else:

            def grad_cosh_sparse(x: NDArray[np.floating]) -> NDArray[np.floating]:
                result = np.zeros(n)
                result[indices] = np.sinh(x[indices])
                return result

            return grad_cosh_sparse

    elif op == "tanh":
        # d/dx tanh(x) = 1 - tanh(x)^2
        if is_full:

            def grad_tanh(x: NDArray[np.floating]) -> NDArray[np.floating]:
                return 1.0 - np.tanh(x) ** 2

            return grad_tanh
        else:

            def grad_tanh_sparse(x: NDArray[np.floating]) -> NDArray[np.floating]:
                result = np.zeros(n)
                result[indices] = 1.0 - np.tanh(x[indices]) ** 2
                return result

            return grad_tanh_sparse

    elif op == "tan":
        # d/dx tan(x) = 1 / cos(x)^2
        if is_full:

            def grad_tan(x: NDArray[np.floating]) -> NDArray[np.floating]:
                raw = 1.0 / np.cos(x) ** 2
                return _sanitize_derivatives(raw)

            return grad_tan
        else:

            def grad_tan_sparse(x: NDArray[np.floating]) -> NDArray[np.floating]:
                result = np.zeros(n)
                result[indices] = 1.0 / np.cos(x[indices]) ** 2
                return _sanitize_derivatives(result)

            return grad_tan_sparse

    elif op == "abs":
        # d/dx |x| = sign(x)
        if is_full:

            def grad_abs(x: NDArray[np.floating]) -> NDArray[np.floating]:
                return np.sign(x)

            return grad_abs
        else:

            def grad_abs_sparse(x: NDArray[np.floating]) -> NDArray[np.floating]:
                result = np.zeros(n)
                result[indices] = np.sign(x[indices])
                return result

            return grad_abs_sparse

    else:
        # Fallback to general symbolic differentiation
        from optyx.core.autodiff import gradient

        grad_exprs = [gradient(expr, var) for var in variables]
        grad_fns = [compile_expression(g, variables) for g in grad_exprs]

        def fallback_gradient(x: NDArray[np.floating]) -> NDArray[np.floating]:
            raw = np.array([fn(x) for fn in grad_fns])
            return _sanitize_derivatives(raw)

        return fallback_gradient


class CompiledExpression:
    """A compiled expression with both value and gradient evaluation.

    Provides a convenient interface for optimization solvers that need
    both objective function and gradient. Uses symbolic differentiation
    for exact gradient computation.
    """

    __slots__ = ("_expr", "_variables", "_value_fn", "_gradient_fn", "_var_names")

    def __init__(self, expr: Expression, variables: list[Variable]) -> None:
        self._expr = expr
        self._variables = variables
        self._var_names = [v.name for v in variables]
        self._value_fn = compile_expression(expr, variables)
        self._gradient_fn = compile_gradient(expr, variables)

    @property
    def n_variables(self) -> int:
        """Number of decision variables."""
        return len(self._variables)

    @property
    def variable_names(self) -> list[str]:
        """Names of decision variables in order."""
        return self._var_names.copy()

    def value(self, x: NDArray[np.floating]) -> float:
        """Evaluate the expression at point x."""
        result = self._value_fn(x)
        return float(np.asarray(result).item())

    def gradient(self, x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute the gradient at point x."""
        return self._gradient_fn(x)

    def value_and_gradient(
        self, x: NDArray[np.floating]
    ) -> tuple[float, NDArray[np.floating]]:
        """Compute both value and gradient at point x.

        Returns:
            A tuple of (objective_value, gradient_array).
        """
        return self.value(x), self.gradient(x)
