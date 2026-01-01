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
    from optyx.core.vectors import LinearCombination, VectorSum, DotProduct

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
        elif isinstance(current, (LinearCombination, VectorSum)):
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
            # VectorExpression - build evaluators for each element
            elem_fns = [
                _build_evaluator(e, var_indices) for e in expr.vector._expressions
            ]
            return lambda x, c=coeffs, fns=elem_fns: np.dot(
                c, np.array([f(x) for f in fns])
            )

    elif isinstance(expr, VectorSum):
        # sum(x) = x[0] + x[1] + ... - efficient numpy implementation
        indices = np.array([var_indices[v.name] for v in expr.vector._variables])
        return lambda x, idx=indices: np.sum(x[idx])

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
            raise ValueError(f"Unknown binary operator: {op}")

    elif isinstance(expr, UnaryOp):
        operand_fn = _build_evaluator(expr.operand, var_indices)
        numpy_func = expr._numpy_func
        return lambda x, f=operand_fn, np_f=numpy_func: np_f(f(x))

    else:
        raise TypeError(f"Unknown expression type: {type(expr)}")


def _build_vector_evaluator(
    vec: Any,
    var_indices: dict[str, int],
) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
    """Build an evaluator for a vector (returns array of values)."""
    from optyx.core.vectors import VectorExpression, VectorVariable

    if isinstance(vec, VectorVariable):
        indices = np.array([var_indices[v.name] for v in vec._variables])
        return lambda x, idx=indices: x[idx]
    elif isinstance(vec, VectorExpression):
        elem_fns = [_build_evaluator(e, var_indices) for e in vec._expressions]
        return lambda x, fns=elem_fns: np.array([f(x) for f in fns])
    else:
        raise TypeError(f"Unknown vector type: {type(vec)}")


def _build_evaluator_iterative(
    expr: Expression,
    var_indices: dict[str, int],
) -> Callable[[NDArray[np.floating]], NDArray[np.floating] | np.floating | float]:
    """Build evaluator using iterative post-order traversal.

    Handles deep expression trees that would cause RecursionError.
    Uses explicit stack to build closures bottom-up.
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
                # VectorExpression - build non-recursive
                elem_fns = []
                for e in node.vector._expressions:
                    if isinstance(e, Variable):
                        idx = var_indices[e.name]
                        elem_fns.append(lambda x, i=idx: x[i])
                    elif isinstance(e, Constant):
                        val = e.value
                        elem_fns.append(lambda x, v=val: v)
                    else:
                        # Fallback to recursive for complex elements
                        elem_fns.append(_build_evaluator(e, var_indices))
                result_stack.append(
                    lambda x, c=coeffs, fns=elem_fns: np.dot(
                        c, np.array([f(x) for f in fns])
                    )
                )
            continue

        if isinstance(node, VectorSum):
            indices = np.array([var_indices[v.name] for v in node.vector._variables])
            result_stack.append(lambda x, idx=indices: np.sum(x[idx]))
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
                    raise ValueError(f"Unknown binary operator: {op}")
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

        # Unknown type - try to evaluate directly
        raise TypeError(f"Unknown expression type: {type(node)}")

    if not result_stack:
        raise RuntimeError("Failed to build evaluator - empty result stack")
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
