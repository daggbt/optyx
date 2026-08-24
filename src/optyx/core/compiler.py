"""Expression compiler for fast evaluation.

Compiles expression trees into optimized callables that minimize
Python overhead during repeated evaluations (e.g., in optimization loops).

Performance optimizations:
- Closure-based evaluation avoids dictionary lookups
- LRU cache prevents recompilation of identical expressions
- Iterative compilation for deep trees avoids recursion limits
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, cast

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


class _ContiguousVectorCompileLayout:
    """Index resolver for a single contiguous VectorVariable layout."""

    __slots__ = ("vector_name", "vector_size", "_prefix", "_full_indices")

    def __init__(self, vector_name: str, vector_size: int) -> None:
        self.vector_name = vector_name
        self.vector_size = vector_size
        self._prefix = f"{vector_name}["
        self._full_indices = np.arange(vector_size, dtype=np.intp)

    def _parse_name_index(self, name: str) -> int | None:
        if not name.startswith(self._prefix) or not name.endswith("]"):
            return None

        try:
            index = int(name[len(self._prefix) : -1])
        except ValueError:
            return None

        if 0 <= index < self.vector_size:
            return index
        return None

    def contains_name(self, name: str) -> bool:
        return self._parse_name_index(name) is not None

    def index_of_name(self, name: str) -> int:
        index = self._parse_name_index(name)
        if index is None:
            raise KeyError(name)
        return index

    def matches_vector(self, vec: Any) -> bool:
        return (
            getattr(vec, "name", None) == self.vector_name
            and getattr(vec, "size", None) == self.vector_size
        )

    def vector_indices(self, vec: Any, *, allow_subset: bool = False) -> np.ndarray:
        if self.matches_vector(vec):
            return self._full_indices

        if allow_subset:
            return np.array(
                [
                    index
                    for name in vec._iter_variable_names()
                    if (index := self._parse_name_index(name)) is not None
                ],
                dtype=np.intp,
            )

        return np.fromiter(
            (self.index_of_name(name) for name in vec._iter_variable_names()),
            dtype=np.intp,
            count=vec.size,
        )


class ContiguousVectorVariables(Sequence[Any]):
    """Vector-backed variable sequence for single-vector compile fast paths."""

    __slots__ = ("source_vector", "_compile_layout")

    def __init__(self, source_vector: Any) -> None:
        self.source_vector = source_vector
        self._compile_layout = _ContiguousVectorCompileLayout(
            source_vector.name,
            source_vector.size,
        )

    def __len__(self) -> int:
        return self.source_vector.size

    def __getitem__(self, index: int | slice) -> Any:
        if isinstance(index, slice):
            return [
                self.source_vector._get_variable(i)
                for i in range(*index.indices(len(self)))
            ]

        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.source_vector._get_variable(index)

    def __iter__(self):
        for index in range(len(self)):
            yield self.source_vector._get_variable(index)

    def materialize(self) -> list[Any]:
        return list(self.source_vector._variables)


def _contiguous_compile_layout(
    variables: Any,
) -> _ContiguousVectorCompileLayout | None:
    if isinstance(variables, ContiguousVectorVariables):
        return variables._compile_layout
    return None


def _build_var_index_data(
    variables: Any,
) -> dict[str, int] | _ContiguousVectorCompileLayout:
    layout = _contiguous_compile_layout(variables)
    if layout is not None:
        return layout
    return {var.name: i for i, var in enumerate(variables)}


def _lookup_var_index(
    var_indices: dict[str, int] | _ContiguousVectorCompileLayout,
    name: str,
) -> int:
    if isinstance(var_indices, _ContiguousVectorCompileLayout):
        return var_indices.index_of_name(name)
    return var_indices[name]


def _has_var_name(
    var_indices: dict[str, int] | _ContiguousVectorCompileLayout,
    name: str,
) -> bool:
    if isinstance(var_indices, _ContiguousVectorCompileLayout):
        return var_indices.contains_name(name)
    return name in var_indices


def _iter_index_names(vec: Any):
    """Yield stable variable names for indexing without forcing materialization."""
    cache = getattr(vec, "_variable_cache", None)
    if cache is not None:
        for index, variable in enumerate(cache):
            if variable is not None:
                yield variable.name
            else:
                yield vec._name_at(index)
        return

    yield from vec._iter_variable_names()


def compile_expression(
    expr: Expression,
    variables: Any,
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
    layout = _contiguous_compile_layout(variables)
    if layout is not None:
        return _compile_contiguous_vector_cached(
            expr,
            layout.vector_name,
            layout.vector_size,
        )

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
    from optyx.core.optimizer import flatten_expression

    var_indices = dict(var_indices_items)
    optimized_expr = flatten_expression(expr)

    # Check tree depth
    depth = _estimate_tree_depth(optimized_expr)
    if depth >= _RECURSION_THRESHOLD:
        eval_func = _build_evaluator_iterative(optimized_expr, var_indices)
    else:
        eval_func = _build_evaluator(optimized_expr, var_indices)
    return eval_func


@lru_cache(maxsize=1024)
def _compile_contiguous_vector_cached(
    expr: Expression,
    vector_name: str,
    vector_size: int,
) -> Callable[[NDArray[np.floating]], NDArray[np.floating] | np.floating | float]:
    """Cached compilation for a single contiguous VectorVariable layout."""
    from optyx.core.optimizer import flatten_expression

    layout = _ContiguousVectorCompileLayout(vector_name, vector_size)
    optimized_expr = flatten_expression(expr)

    depth = _estimate_tree_depth(optimized_expr)
    if depth >= _RECURSION_THRESHOLD:
        eval_func = _build_evaluator_iterative(optimized_expr, layout)
    else:
        eval_func = _build_evaluator(optimized_expr, layout)
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


def _vector_indices(
    vec: Any,
    var_indices: dict[str, int] | _ContiguousVectorCompileLayout,
    *,
    allow_subset: bool = False,
) -> np.ndarray:
    """Resolve VectorVariable element indices without materializing scalars."""
    if isinstance(var_indices, _ContiguousVectorCompileLayout):
        return var_indices.vector_indices(vec, allow_subset=allow_subset)

    if allow_subset:
        return np.array(
            [
                _lookup_var_index(var_indices, name)
                for name in _iter_index_names(vec)
                if _has_var_name(var_indices, name)
            ],
            dtype=np.intp,
        )

    return np.fromiter(
        (_lookup_var_index(var_indices, name) for name in _iter_index_names(vec)),
        dtype=np.intp,
        count=vec.size,
    )


def _variables_match_vector_names(variables: Any, vec: Any) -> bool:
    """Check whether the variable ordering matches a VectorVariable exactly."""
    layout = _contiguous_compile_layout(variables)
    if layout is not None:
        return layout.matches_vector(vec)

    if len(variables) != vec.size:
        return False

    for variable, name in zip(variables, _iter_index_names(vec)):
        if variable.name != name:
            return False
    return True


def _try_build_nary_sum_fast_evaluator(
    expr: NarySum,
    var_indices: dict[str, int] | _ContiguousVectorCompileLayout,
) -> Callable[[NDArray[np.floating]], float] | None:
    """Build a vectorized evaluator for common loop-built sum patterns."""
    from optyx.core.expressions import BinaryOp, Constant, UnaryOp, Variable

    terms = expr.terms
    if not terms:
        return lambda x: 0.0

    if all(isinstance(term, Variable) for term in terms):
        variable_terms = cast("tuple[Variable, ...]", terms)
        indices = np.array(
            [_lookup_var_index(var_indices, term.name) for term in variable_terms],
            dtype=np.intp,
        )
        return lambda x, idx=indices: float(np.sum(x[idx]))

    power_indices: list[int] = []
    power_value: float | None = None
    for term in terms:
        if not isinstance(term, BinaryOp) or term.op != "**":
            power_indices = []
            break
        if not isinstance(term.left, Variable) or not isinstance(term.right, Constant):
            power_indices = []
            break
        exponent = term.right.value
        if isinstance(exponent, np.ndarray):
            power_indices = []
            break
        exponent_value = float(exponent)
        if power_value is None:
            power_value = exponent_value
        elif exponent_value != power_value:
            power_indices = []
            break
        power_indices.append(_lookup_var_index(var_indices, term.left.name))

    if power_indices and power_value is not None:
        indices = np.array(power_indices, dtype=np.intp)
        return lambda x, idx=indices, power=power_value: float(np.sum(x[idx] ** power))

    unary_indices: list[int] = []
    unary_op: str | None = None
    numpy_func: np.ufunc | None = None
    for term in terms:
        if not isinstance(term, UnaryOp) or not isinstance(term.operand, Variable):
            unary_indices = []
            break
        if unary_op is None:
            unary_op = term.op
            numpy_func = term._numpy_func
        elif term.op != unary_op:
            unary_indices = []
            break
        unary_indices.append(_lookup_var_index(var_indices, term.operand.name))

    if unary_indices and numpy_func is not None:
        indices = np.array(unary_indices, dtype=np.intp)
        return lambda x, idx=indices, np_f=numpy_func: float(np.sum(np_f(x[idx])))

    return None


def _build_evaluator(
    expr: Expression,
    var_indices: dict[str, int] | _ContiguousVectorCompileLayout,
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
        idx = _lookup_var_index(var_indices, expr.name)
        return lambda x, i=idx: x[i]

    elif isinstance(expr, LinearCombination):
        # c @ x = c[0]*x[0] + c[1]*x[1] + ... - efficient numpy implementation
        coeffs = np.asarray(expr.coefficients)
        if isinstance(expr.vector, VectorVariable):
            indices = _vector_indices(expr.vector, var_indices)
            return lambda x, c=coeffs, idx=indices: np.dot(c, x[idx])
        else:
            # VectorExpression/VectorBinaryOp - use vector evaluator
            vec_fn = _build_vector_evaluator(expr.vector, var_indices)
            return lambda x, c=coeffs, vf=vec_fn: np.dot(c, vf(x))

    elif isinstance(expr, VectorSum):
        # sum(x) = x[0] + x[1] + ... - efficient numpy implementation
        indices = _vector_indices(expr.vector, var_indices)
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
        indices = _vector_indices(expr.vector, var_indices)
        power = expr.power
        return lambda x, idx=indices, k=power: float(np.sum(x[idx] ** k))

    elif isinstance(expr, VectorUnarySum):
        # sum(f(x)) - efficient numpy implementation
        indices = _vector_indices(expr.vector, var_indices)
        op = expr.op
        numpy_func = VectorUnarySum._NUMPY_FUNCS[op]
        return lambda x, idx=indices, f=numpy_func: float(np.sum(f(x[idx])))

    elif isinstance(expr, ElementwisePower):
        # x ** k element-wise - returns array
        indices = _vector_indices(expr.vector, var_indices)
        power = expr.power
        return lambda x, idx=indices, k=power: x[idx] ** k

    elif isinstance(expr, ElementwiseUnary):
        # f(x) element-wise - returns array
        indices = _vector_indices(expr.vector, var_indices)
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
        fast_sum = _try_build_nary_sum_fast_evaluator(expr, var_indices)
        if fast_sum is not None:
            return fast_sum
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
    var_indices: dict[str, int] | _ContiguousVectorCompileLayout,
) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
    """Build an evaluator for a vector (returns array of values)."""
    from optyx.core.vectors import (
        ElementwisePower,
        VectorBinaryOp,
        VectorExpression,
        VectorVariable,
    )

    if isinstance(vec, VectorVariable):
        indices = _vector_indices(vec, var_indices)
        return lambda x, idx=indices: x[idx]
    elif isinstance(vec, VectorBinaryOp):
        # Single numpy op instead of N per-element evaluations
        left_fn = _build_vector_evaluator(vec.left, var_indices)
        right_fn = _build_vector_evaluator(vec.right, var_indices)
        np_op = vec._NUMPY_OPS[vec.op]
        return lambda x, lf=left_fn, rf=right_fn, op=np_op: op(lf(x), rf(x))
    elif isinstance(vec, ElementwisePower):
        base_fn = _build_vector_evaluator(vec.vector, var_indices)
        p = vec.power
        return lambda x, bf=base_fn, pw=p: bf(x) ** pw
    elif isinstance(vec, VectorExpression):
        elem_fns = [_build_evaluator(e, var_indices) for e in vec._expressions]
        n_elems = len(elem_fns)

        def vector_expr_eval(
            x: NDArray[np.floating], fns=elem_fns, n=n_elems
        ) -> NDArray[np.floating]:
            res = np.empty(n)
            for i, f in enumerate(fns):
                res[i] = f(x)
            return res

        return vector_expr_eval
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
    var_indices: dict[str, int] | _ContiguousVectorCompileLayout,
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
            idx = _lookup_var_index(var_indices, node.name)
            result_stack.append(lambda x, i=idx: x[i])
            continue

        # Vector expressions - O(n) but not recursive
        if isinstance(node, LinearCombination):
            coeffs = np.asarray(node.coefficients)
            if isinstance(node.vector, VectorVariable):
                indices = _vector_indices(node.vector, var_indices)
                result_stack.append(lambda x, c=coeffs, idx=indices: np.dot(c, x[idx]))
            else:
                # VectorExpression/VectorBinaryOp - use vector evaluator
                vec_fn = _build_vector_evaluator(node.vector, var_indices)
                result_stack.append(lambda x, c=coeffs, vf=vec_fn: np.dot(c, vf(x)))
            continue

        if isinstance(node, VectorSum):
            indices = _vector_indices(node.vector, var_indices)
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
                    idx = _lookup_var_index(var_indices, e.name)
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
    n_vars = len(var_names)

    def dict_fn(
        values: dict[str, float | NDArray[np.floating]],
    ) -> NDArray[np.floating] | np.floating | float:
        arr = np.empty(n_vars)
        for i, name in enumerate(var_names):
            arr[i] = values[name]
        return array_fn(arr)

    return dict_fn


def compile_vector_gradient(
    expr: Expression,
    variables: Any,
) -> Callable[[NDArray[np.floating]], NDArray[np.floating]] | None:
    """Attempt to compile a fast vector gradient O(1)."""
    from optyx.core.autodiff import detect_affine_gradient_pattern

    pattern = detect_affine_gradient_pattern(expr)
    if pattern is None:
        return None

    # Check if variables match exactly Pattern.vector
    if not _variables_match_vector_names(variables, pattern.vector):
        return None

    b = pattern.constant_term
    lt = pattern.linear_type

    # Fast path: use structured metadata to avoid O(n²) matrix operations
    if lt == "scaled_identity":
        scale = pattern.linear_scale
        if b is None:
            return lambda x, _s=scale: _s * x
        else:
            b_val = b
            return lambda x, _s=scale, _b=b_val: _s * x + _b

    if lt == "diagonal":
        diag = pattern.linear_diag
        if b is None:
            return lambda x, _d=diag: _d * x  # type: ignore
        else:
            b_val = b
            return lambda x, _d=diag, _b=b_val: _d * x + _b  # type: ignore

    A = pattern.linear_term

    # Cases
    if A is None and b is None:
        zeros = np.zeros(len(variables))
        return lambda x: zeros

    if A is None:
        b_val = b  # capture for closure
        return lambda x: b_val  # type: ignore

    if b is None:
        # Gradient is A @ x — A is a general matrix (O(n²) checks only for general)
        if lt != "general":
            # Unknown type, try diagonal detection
            A_diag = np.diagonal(A)
            A_is_diag = np.count_nonzero(A - np.diag(A_diag)) == 0

            if A_is_diag:
                if np.all(A_diag == A_diag[0]):
                    scale = float(A_diag[0])
                    return lambda x, _s=scale: _s * x
                return lambda x, _d=A_diag.copy(): _d * x

        def grad_Ax(x: NDArray[np.floating]) -> NDArray[np.floating]:
            return A @ x  # type: ignore

        return grad_Ax

    # b is not None, A is not None
    if lt != "general":
        A_diag = np.diagonal(A)
        A_is_diag = np.count_nonzero(A - np.diag(A_diag)) == 0

        if A_is_diag:
            if np.all(A_diag == A_diag[0]):
                scale = float(A_diag[0])
                b_val = b
                return lambda x, _s=scale, _b=b_val: _s * x + _b
            b_val = b
            d = A_diag.copy()
            return lambda x, _d=d, _b=b_val: _d * x + _b

    def grad_Ax_b(x: NDArray[np.floating]) -> NDArray[np.floating]:
        return A @ x + b

    return grad_Ax_b


def compile_gradient(
    expr: Expression,
    variables: Any,
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
    from optyx.core.optimizer import flatten_expression
    from optyx.core.expressions import NarySum  # noqa: F811

    expr = flatten_expression(expr)
    is_contiguous_single_vector = isinstance(variables, ContiguousVectorVariables)

    if isinstance(expr, NarySum) and not is_contiguous_single_vector:
        result = _compile_nary_sum_gradient_fast(expr, variables)
        if result is not None:
            return result

    # Fast path: Vector Gradient Pattern (Linear/Quadratic forms)
    vec_grad = compile_vector_gradient(expr, variables)
    if vec_grad is not None:
        return vec_grad

    from optyx.core.vectors import VectorPowerSum, VectorUnarySum, VectorBinaryOp

    # Fast path for VectorPowerSum: gradient is k * x^(k-1), vectorized
    if isinstance(expr, VectorPowerSum):
        return _compile_vectorized_power_gradient(expr, variables)

    # Fast path for VectorUnarySum: gradient is f'(x), vectorized
    if isinstance(expr, VectorUnarySum):
        return _compile_vectorized_unary_gradient(expr, variables)

    # Fast path for VectorExpressionSum(VectorBinaryOp): vectorized gradient
    from optyx.core.vectors import VectorExpressionSum

    if isinstance(expr, VectorExpressionSum) and isinstance(
        expr.expression, VectorBinaryOp
    ):
        result = _compile_vectorized_binary_op_sum_gradient(expr.expression, variables)
        if result is not None:
            return result

    if is_contiguous_single_vector:
        return compile_gradient(expr, variables.materialize())

    # General path: symbolic differentiation
    from optyx.core.autodiff import gradient

    # Compute symbolic gradient for each variable
    grad_exprs = [gradient(expr, var) for var in variables]

    # Compile each gradient expression
    grad_fns = [compile_expression(g, variables) for g in grad_exprs]
    n_grads = len(grad_fns)

    def symbolic_gradient(x: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute gradient using symbolic differentiation."""
        raw = np.empty(n_grads)
        for i, fn in enumerate(grad_fns):
            raw[i] = fn(x)
        return _sanitize_derivatives(raw)

    return symbolic_gradient


def compile_sparse_gradient(
    expr: "Expression",
    variables: list["Variable"],
) -> Callable[["NDArray[np.floating]"], Any]:
    """Compile a gradient that returns a sparse row vector (1×n csr_matrix).

    Uses sparsity analysis to only compute non-zero partial derivatives,
    returning a scipy.sparse.csr_matrix of shape (1, n) with O(nnz) memory.

    For constant gradients (linear expressions), returns a pre-built sparse
    matrix. For variable gradients, compiles only the non-zero columns.

    Args:
        expr: The expression to differentiate.
        variables: Ordered list of variables.

    Returns:
        A callable that returns the gradient as a (1, n) csr_matrix.
    """
    from scipy.sparse import csr_matrix
    from optyx.core.autodiff import analyze_gradient_sparsity, gradient as sym_gradient

    n = len(variables)
    sparsity = analyze_gradient_sparsity(expr, variables)

    # Constant gradient: return pre-built sparse matrix
    if sparsity.is_constant:
        if sparsity.nnz == 0:
            const_sparse = csr_matrix((1, n), dtype=np.float64)
        else:
            data = sparsity.constant_values
            indices = sparsity.nnz_indices.copy()
            indptr = np.array([0, len(indices)], dtype=np.int32)
            const_sparse = csr_matrix((data, indices, indptr), shape=(1, n))

        return lambda x, _m=const_sparse: _m

    nnz_idx = sparsity.nnz_indices

    if len(nnz_idx) == 0:
        zero_sparse = csr_matrix((1, n), dtype=np.float64)
        return lambda x, _m=zero_sparse: _m

    # Compile only the non-zero columns
    grad_exprs = [sym_gradient(expr, variables[j]) for j in nnz_idx]
    compiled_fns = [compile_expression(e, variables) for e in grad_exprs]

    def sparse_gradient(x: "NDArray[np.floating]") -> Any:
        data = np.array([f(x) for f in compiled_fns], dtype=np.float64)
        return csr_matrix((data, nnz_idx, np.array([0, len(nnz_idx)])), shape=(1, n))

    return sparse_gradient


def compile_sparse_gradient_dense_output(
    expr: "Expression",
    variables: list["Variable"],
) -> Callable[["NDArray[np.floating]"], NDArray[np.floating]]:
    """Compile a sparse-eval gradient that returns a dense vector.

    This is intended for solver frontends such as ``scipy.optimize.minimize``
    that require dense objective gradients. Only structurally non-zero partials
    are compiled and evaluated, then scattered into a dense 1D array.
    """
    from optyx.core.autodiff import analyze_gradient_sparsity, gradient as sym_gradient

    n = len(variables)
    sparsity = analyze_gradient_sparsity(expr, variables)

    if sparsity.is_constant:
        const_dense = np.zeros(n, dtype=np.float64)
        if sparsity.constant_values is not None:
            const_dense[sparsity.nnz_indices] = sparsity.constant_values
        return lambda x, _g=const_dense: _g

    nnz_idx = sparsity.nnz_indices

    if len(nnz_idx) == 0:
        zero_dense = np.zeros(n, dtype=np.float64)
        return lambda x, _g=zero_dense: _g

    grad_exprs = [sym_gradient(expr, variables[j]) for j in nnz_idx]
    compiled_fns = [compile_expression(e, variables) for e in grad_exprs]

    def dense_sparse_gradient(x: "NDArray[np.floating]") -> NDArray[np.floating]:
        result = np.zeros(n, dtype=np.float64)
        for idx, fn in zip(nnz_idx, compiled_fns):
            result[idx] = fn(x)
        return _sanitize_derivatives(result)

    return dense_sparse_gradient


# Default density threshold for switching between sparse and dense
_SPARSE_DENSITY_THRESHOLD = 0.5


def compile_gradient_with_sparsity(
    expr: "Expression",
    variables: list["Variable"],
    density_threshold: float = _SPARSE_DENSITY_THRESHOLD,
) -> Callable[["NDArray[np.floating]"], Any]:
    """Compile gradient, choosing sparse or dense format based on sparsity.

    Analyzes the expression's sparsity pattern and returns:
    - A sparse gradient (csr_matrix) if density <= threshold
    - A dense gradient (ndarray) if density > threshold

    Args:
        expr: The expression to differentiate.
        variables: Ordered list of variables.
        density_threshold: Density above which dense format is used (default 0.5).

    Returns:
        A callable returning either a (1, n) csr_matrix or (n,) ndarray.
    """
    from optyx.core.autodiff import analyze_gradient_sparsity

    sparsity = analyze_gradient_sparsity(expr, variables)

    if sparsity.density <= density_threshold:
        return compile_sparse_gradient(expr, variables)
    else:
        return compile_gradient(expr, variables)


def _compile_vectorized_power_gradient(
    expr: "VectorPowerSum",
    variables: Any,
) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
    """Compile O(1) gradient for VectorPowerSum.

    For sum(x**k), gradient w.r.t. x[i] is k * x[i] ** (k-1).
    This generates a single numpy operation instead of n separate functions.
    """
    k = expr.power
    n = len(variables)

    # Build index mapping: which positions in the gradient correspond to vector vars
    var_name_to_idx = _build_var_index_data(variables)
    indices = _vector_indices(expr.vector, var_name_to_idx)

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
    variables: Any,
) -> Callable[[NDArray[np.floating]], NDArray[np.floating]]:
    """Compile O(1) gradient for VectorUnarySum.

    For sum(f(x)), gradient w.r.t. x[i] is f'(x[i]).
    This generates vectorized numpy operations instead of n separate functions.
    """
    op = expr.op
    n = len(variables)

    # Build index mapping
    var_name_to_idx = _build_var_index_data(variables)
    indices = _vector_indices(expr.vector, var_name_to_idx)

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
        n_fns = len(grad_fns)

        def fallback_gradient(x: NDArray[np.floating]) -> NDArray[np.floating]:
            raw = np.empty(n_fns)
            for i, fn in enumerate(grad_fns):
                raw[i] = fn(x)
            return _sanitize_derivatives(raw)

        return fallback_gradient


def _compile_vectorized_binary_op_sum_gradient(
    vbo: Any,
    variables: Any,
) -> Callable[[NDArray[np.floating]], NDArray[np.floating]] | None:
    """Compile O(1) gradient for sum(VectorBinaryOp).

    Handles sum(left op right) where left/right are VectorVariable or scalar.
    Returns None if the pattern isn't recognized (falls back to symbolic).

    Derivative rules for sum(f(left, right)):
        sum(l + r): ∂/∂l_i = 1, ∂/∂r_i = 1
        sum(l - r): ∂/∂l_i = 1, ∂/∂r_i = -1
        sum(c * x): ∂/∂x_i = c
        sum(l * r): ∂/∂l_i = r_i, ∂/∂r_i = l_i  (needs runtime values)
        sum(l / r): ∂/∂l_i = 1/r_i, ∂/∂r_i = -l_i/r_i^2
    """
    from optyx.core.vectors import VectorBinaryOp, VectorVariable

    if not isinstance(vbo, VectorBinaryOp):
        return None

    op = vbo.op
    left = vbo.left
    right = vbo.right
    n = len(variables)
    var_name_to_idx = _build_var_index_data(variables)

    def _get_indices(
        vec: Any,
    ) -> np.ndarray | None:
        """Get variable indices for a vector operand, or None if scalar/const."""
        if isinstance(vec, VectorVariable):
            return _vector_indices(vec, var_name_to_idx, allow_subset=True)
        return None

    left_idx = _get_indices(left)
    right_idx = _get_indices(right)
    is_scalar_right = isinstance(right, (int, float))

    if op == "+":
        # sum(l + r): grad is 1 for each variable present in l or r
        def grad_add_sum(x: NDArray[np.floating]) -> NDArray[np.floating]:
            result = np.zeros(n)
            if left_idx is not None:
                result[left_idx] += 1.0
            if right_idx is not None:
                result[right_idx] += 1.0
            return result

        return grad_add_sum

    elif op == "-":
        # sum(l - r): grad is +1 for l vars, -1 for r vars
        def grad_sub_sum(x: NDArray[np.floating]) -> NDArray[np.floating]:
            result = np.zeros(n)
            if left_idx is not None:
                result[left_idx] += 1.0
            if right_idx is not None:
                result[right_idx] -= 1.0
            return result

        return grad_sub_sum

    elif op == "*":
        if is_scalar_right:
            # sum(x * c): grad w.r.t. x_i = c
            c = float(right)

            def grad_scalar_mul_sum(x: NDArray[np.floating]) -> NDArray[np.floating]:
                result = np.zeros(n)
                if left_idx is not None:
                    result[left_idx] = c
                return result

            return grad_scalar_mul_sum

        elif left_idx is not None and right_idx is not None:
            # sum(l * r): grad w.r.t. l_i = r_i, grad w.r.t. r_i = l_i
            li = left_idx
            ri = right_idx

            def grad_vec_mul_sum(x: NDArray[np.floating]) -> NDArray[np.floating]:
                result = np.zeros(n)
                result[li] += x[ri]
                result[ri] += x[li]
                return result

            return grad_vec_mul_sum

        return None  # Unrecognized mul pattern

    elif op == "/":
        if is_scalar_right:
            # sum(x / c): grad w.r.t. x_i = 1/c
            inv_c = 1.0 / float(right)

            def grad_scalar_div_sum(x: NDArray[np.floating]) -> NDArray[np.floating]:
                result = np.zeros(n)
                if left_idx is not None:
                    result[left_idx] = inv_c
                return result

            return grad_scalar_div_sum

        elif left_idx is not None and right_idx is not None:
            # sum(l / r): ∂/∂l_i = 1/r_i, ∂/∂r_i = -l_i/r_i^2
            li = left_idx
            ri = right_idx

            def grad_vec_div_sum(x: NDArray[np.floating]) -> NDArray[np.floating]:
                result = np.zeros(n)
                result[li] += 1.0 / x[ri]
                result[ri] -= x[li] / (x[ri] ** 2)
                return _sanitize_derivatives(result)

            return grad_vec_div_sum

        return None

    # Unrecognized op (e.g., **) — fall back
    return None


def _compile_nary_sum_gradient_fast(
    expr: Any,
    variables: list["Variable"],
) -> Callable[[NDArray[np.floating]], NDArray[np.floating]] | None:
    """Try to compile a fast gradient for common NarySum patterns.

    Handles common loop-built scalar patterns directly, and falls back to the
    existing mixed VectorBinaryOp strategy when only some terms are vectorized.
    """
    from optyx.core.expressions import BinaryOp, Constant, UnaryOp, Variable
    from optyx.core.vectors import VectorExpressionSum, VectorBinaryOp

    n = len(variables)
    var_name_to_idx = {v.name: i for i, v in enumerate(variables)}
    terms = expr.terms

    if terms and all(isinstance(term, Variable) for term in terms):
        indices = np.array(
            [var_name_to_idx[term.name] for term in terms], dtype=np.intp
        )
        if len(indices) == n and np.array_equal(indices, np.arange(n)):
            ones = np.ones(n)
            return lambda x, values=ones: values

        def grad_variable_sum(x: NDArray[np.floating]) -> NDArray[np.floating]:
            result = np.zeros(n)
            np.add.at(result, indices, 1.0)
            return result

        return grad_variable_sum

    if terms:
        power_indices: list[int] = []
        power_value: float | None = None
        for term in terms:
            if not isinstance(term, BinaryOp) or term.op != "**":
                power_indices = []
                break
            if not isinstance(term.left, Variable) or not isinstance(
                term.right, Constant
            ):
                power_indices = []
                break
            exponent = term.right.value
            if isinstance(exponent, np.ndarray):
                power_indices = []
                break
            exponent_value = float(exponent)
            if power_value is None:
                power_value = exponent_value
            elif exponent_value != power_value:
                power_indices = []
                break
            power_indices.append(var_name_to_idx[term.left.name])

        if power_indices and power_value is not None:
            indices = np.array(power_indices, dtype=np.intp)
            if len(indices) == n and np.array_equal(indices, np.arange(n)):
                if power_value == 1.0:
                    ones = np.ones(n)
                    return lambda x, values=ones: values
                if power_value == 2.0:
                    return lambda x: 2.0 * x

                def grad_full_power_sum(
                    x: NDArray[np.floating],
                ) -> NDArray[np.floating]:
                    raw = power_value * np.power(x, power_value - 1.0)
                    return _sanitize_derivatives(raw)

                return grad_full_power_sum

            def grad_sparse_power_sum(x: NDArray[np.floating]) -> NDArray[np.floating]:
                result = np.zeros(n)
                np.add.at(
                    result,
                    indices,
                    power_value * np.power(x[indices], power_value - 1.0),
                )
                return _sanitize_derivatives(result)

            return grad_sparse_power_sum

    if terms:
        unary_indices: list[int] = []
        unary_op: str | None = None
        for term in terms:
            if not isinstance(term, UnaryOp) or not isinstance(term.operand, Variable):
                unary_indices = []
                break
            if unary_op is None:
                unary_op = term.op
            elif term.op != unary_op:
                unary_indices = []
                break
            unary_indices.append(var_name_to_idx[term.operand.name])

        if unary_indices and unary_op is not None:
            indices = np.array(unary_indices, dtype=np.intp)

            def _eval_unary_derivative(
                values: NDArray[np.floating],
            ) -> NDArray[np.floating] | None:
                if unary_op == "sin":
                    return np.cos(values)
                if unary_op == "cos":
                    return -np.sin(values)
                if unary_op == "exp":
                    return np.exp(values)
                if unary_op == "log":
                    return 1.0 / values
                if unary_op == "sqrt":
                    return 1.0 / (2.0 * np.sqrt(values))
                if unary_op == "sinh":
                    return np.cosh(values)
                if unary_op == "cosh":
                    return np.sinh(values)
                if unary_op == "tanh":
                    return 1.0 - np.tanh(values) ** 2
                return None

            if len(indices) == n and np.array_equal(indices, np.arange(n)):

                def grad_full_unary_sum(
                    x: NDArray[np.floating],
                ) -> NDArray[np.floating]:
                    raw = _eval_unary_derivative(x)
                    if raw is None:
                        raise RuntimeError("unsupported unary sum gradient fast path")
                    return _sanitize_derivatives(raw)

                if _eval_unary_derivative(np.ones(1)) is not None:
                    return grad_full_unary_sum
            elif _eval_unary_derivative(np.ones(1)) is not None:

                def grad_sparse_unary_sum(
                    x: NDArray[np.floating],
                ) -> NDArray[np.floating]:
                    raw = _eval_unary_derivative(x[indices])
                    assert raw is not None
                    result = np.zeros(n)
                    np.add.at(result, indices, raw)
                    return _sanitize_derivatives(result)

                return grad_sparse_unary_sum

    fast_grads: list[Callable] = []
    slow_terms: list[Any] = []

    for term in expr.terms:
        if isinstance(term, VectorExpressionSum) and isinstance(
            term.expression, VectorBinaryOp
        ):
            fg = _compile_vectorized_binary_op_sum_gradient(term.expression, variables)
            if fg is not None:
                fast_grads.append(fg)
                continue
        slow_terms.append(term)

    if not fast_grads:
        return None  # No benefit, fall back entirely

    # Compile slow terms via symbolic differentiation
    from optyx.core.autodiff import gradient as sym_gradient

    slow_grad_fns: list[list[Callable]] = []
    for term in slow_terms:
        grad_exprs = [sym_gradient(term, var) for var in variables]
        compiled = [compile_expression(g, variables) for g in grad_exprs]
        slow_grad_fns.append(compiled)

    def nary_gradient(x: NDArray[np.floating]) -> NDArray[np.floating]:
        result = np.zeros(n)
        # Fast vectorized contributions
        for fg in fast_grads:
            result += fg(x)
        # Slow symbolic contributions
        temp = np.empty(n)
        for compiled in slow_grad_fns:
            for i, fn in enumerate(compiled):
                temp[i] = fn(x)
            result += temp
        return _sanitize_derivatives(result)

    return nary_gradient


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
