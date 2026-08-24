"""Problem analysis utilities.

Provides linear / quadratic detection and helpers to compute polynomial degree
of expression trees. These utilities are used to detect LP/QP problems for
fast-path solver selection.

Performance optimizations:
- Early termination: stops traversal immediately when non-polynomial detected
- Degree-bounded traversal: is_linear/is_quadratic stop when threshold exceeded
- Memoization: caches results for repeated sub-expressions (common in constraints)
- Iterative traversal: for deep expression trees (> 400 depth) to avoid recursion limit
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional, Sequence
import numbers

import numpy as np
from numpy.typing import NDArray

from optyx.core.expressions import (
    Expression,
    Constant,
    Variable,
    BinaryOp,
    UnaryOp,
    NarySum,
    NaryProduct,
)
from optyx.core.errors import NonLinearError, NoObjectiveError
from optyx.core.parameters import Parameter

# Recursion threshold - use iterative algorithm for trees deeper than this
_RECURSION_THRESHOLD = 400

if TYPE_CHECKING:
    from optyx.constraints import Constraint
    from optyx.problem import Problem


def compute_degree(expr: Expression) -> Optional[int]:
    """Compute the polynomial degree of an expression.

    Returns:
        - integer degree >= 0 if the expression is a polynomial
        - ``None`` if the expression is non-polynomial (e.g., sin, exp,
          division by variable, non-integer powers)

    Uses memoization for repeated sub-expressions.
    For deep expression trees (> 400 depth), uses iterative algorithm.
    """
    # Check tree depth and use iterative for deep trees
    depth = _estimate_tree_depth(expr)
    if depth >= _RECURSION_THRESHOLD:
        return _compute_degree_iterative(expr)
    return _compute_degree_cached(id(expr), expr)


def _estimate_tree_depth(expr: Expression, max_depth: int = 500) -> int:
    """Estimate the depth of an expression tree.

    Uses iterative traversal to check both left and right branches,
    avoiding RecursionError for any tree shape (left-skewed, right-skewed,
    or balanced).

    Args:
        expr: The expression to check.
        max_depth: Maximum depth to check before returning early.

    Returns:
        Estimated maximum depth of the tree.
    """
    from optyx.core.vectors import LinearCombination, VectorSum, DotProduct

    # Use explicit stack to avoid recursion
    stack: list[tuple[Any, int]] = [(expr, 0)]  # (node, current_depth)
    max_found = 0

    while stack and max_found < max_depth:
        current, depth = stack.pop()
        max_found = max(max_found, depth)

        if isinstance(current, (Constant, Parameter, Variable)):
            continue
        elif isinstance(current, BinaryOp):
            # Check both branches
            stack.append((current.left, depth + 1))
            stack.append((current.right, depth + 1))
        elif isinstance(current, UnaryOp):
            stack.append((current.operand, depth + 1))
        elif isinstance(current, NarySum):
            stack.extend((term, depth + 1) for term in current.terms)
        elif isinstance(current, NaryProduct):
            stack.extend((factor, depth + 1) for factor in current.factors)
        elif isinstance(current, (LinearCombination, VectorSum)):
            continue  # These don't recurse deeply
        elif isinstance(current, DotProduct):
            stack.append((current.left, depth + 1))
            stack.append((current.right, depth + 1))

    return max_found


def _compute_degree_iterative(expr: Expression) -> Optional[int]:
    """Compute degree iteratively using explicit stack.

    Handles deep expression trees that would cause RecursionError.
    """
    from optyx.core.matrices import QuadraticForm
    from optyx.core.vectors import (
        DotProduct,
        LinearCombination,
        VectorSum,
        VectorPowerSum,
        VectorUnarySum,
        ElementwisePower,
        ElementwiseUnary,
    )

    # Stack: (expression, phase, left_result, right_result)
    # phase 0: first visit, phase 1: left done, phase 2: both done
    stack: list[tuple[Expression, int, Optional[int], Optional[int]]] = [
        (expr, 0, None, None)
    ]
    result_stack: list[Optional[int]] = []

    while stack:
        node, phase, left_deg, right_deg = stack.pop()

        # Leaf nodes - return immediately
        if isinstance(node, Constant):
            result_stack.append(0)
            continue
        if isinstance(node, Parameter):
            result_stack.append(0)
            continue
        if isinstance(node, Variable):
            result_stack.append(1)
            continue

        # Flattened associative operations. Process all children in one stack
        # frame so wide expressions remain shallow and allocation-light.
        if isinstance(node, (NarySum, NaryProduct)):
            children = node.terms if isinstance(node, NarySum) else node.factors
            if phase == 0:
                stack.append((node, 1, None, None))
                for child in reversed(children):
                    stack.append((child, 0, None, None))
            else:
                count = len(children)
                child_degrees = result_stack[-count:] if count else []
                if count:
                    del result_stack[-count:]
                if any(degree is None for degree in child_degrees):
                    result_stack.append(None)
                else:
                    degrees = [degree for degree in child_degrees if degree is not None]
                    if isinstance(node, NarySum):
                        result_stack.append(max(degrees, default=0))
                    else:
                        result_stack.append(sum(degrees))
            continue

        # Vector expressions - these have known degrees
        if isinstance(node, LinearCombination):
            result_stack.append(1)
            continue
        if isinstance(node, VectorSum):
            result_stack.append(1)
            continue
        if isinstance(node, DotProduct):
            result_stack.append(2)
            continue
        if isinstance(node, QuadraticForm):
            result_stack.append(2)
            continue
        if isinstance(node, VectorPowerSum):
            # sum(x ** k) has degree k
            result_stack.append(int(node.power))
            continue
        if isinstance(node, VectorUnarySum):
            # sum(sin(x)), sum(exp(x)) etc. are non-polynomial
            result_stack.append(None)
            continue
        if isinstance(node, ElementwisePower):
            # x ** k has degree k
            result_stack.append(int(node.power))
            continue
        if isinstance(node, ElementwiseUnary):
            # sin(x), exp(x) etc. are non-polynomial
            result_stack.append(None)
            continue

        # Unary operations
        if isinstance(node, UnaryOp):
            if node.op == "neg":
                if phase == 0:
                    stack.append((node, 1, None, None))
                    stack.append((node.operand, 0, None, None))
                else:
                    result_stack.append(result_stack.pop())
            else:
                result_stack.append(None)
            continue

        # Binary operations
        if isinstance(node, BinaryOp):
            op = node.op

            if phase == 0:
                # First visit - process children
                stack.append((node, 1, None, None))
                stack.append((node.left, 0, None, None))
            elif phase == 1:
                # Left done
                left_result = result_stack.pop()

                # Early termination for power/division
                if op == "**":
                    if not isinstance(node.right, Constant):
                        result_stack.append(None)
                        continue
                    exp_val = node.right.value
                    if not isinstance(exp_val, numbers.Number):
                        result_stack.append(None)
                        continue
                    exp_float = float(exp_val)
                    if not exp_float.is_integer() or exp_float < 0:
                        result_stack.append(None)
                        continue
                    if left_result is None:
                        result_stack.append(None)
                    else:
                        result_stack.append(left_result * int(exp_float))
                    continue

                # Early termination on None for other ops
                if left_result is None:
                    result_stack.append(None)
                    continue

                # Need to process right child
                stack.append((node, 2, left_result, None))
                stack.append((node.right, 0, None, None))
            else:
                # Phase 2: both children done
                right_result = result_stack.pop()

                if right_result is None or left_deg is None:
                    result_stack.append(None)
                    continue

                if op in ("+", "-"):
                    result_stack.append(max(left_deg, right_result))
                elif op == "*":
                    result_stack.append(left_deg + right_result)
                elif op == "/":
                    result_stack.append(left_deg if right_result == 0 else None)
                else:
                    result_stack.append(None)
            continue

        # Unknown node type
        result_stack.append(None)

    return result_stack[-1] if result_stack else None


@lru_cache(maxsize=1024)
def _compute_degree_cached(expr_id: int, expr: Expression) -> Optional[int]:
    """Memoized degree computation keyed by expression object id."""
    return _compute_degree_impl(expr)


def _compute_degree_impl(expr: Expression) -> Optional[int]:
    """Core degree computation with early termination."""
    from optyx.core.matrices import QuadraticForm
    from optyx.core.vectors import (
        DotProduct,
        LinearCombination,
        VectorSum,
        VectorVariable,
        VectorPowerSum,
        VectorUnarySum,
        ElementwisePower,
        ElementwiseUnary,
    )

    # Fast path: leaf nodes (most common)
    if isinstance(expr, Constant):
        return 0
    if isinstance(expr, Parameter):
        return 0
    if isinstance(expr, Variable):
        return 1

    if isinstance(expr, NarySum):
        max_degree = 0
        for term in expr.terms:
            degree = _compute_degree_impl(term)
            if degree is None:
                return None
            max_degree = max(max_degree, degree)
        return max_degree

    if isinstance(expr, NaryProduct):
        total_degree = 0
        for factor in expr.factors:
            degree = _compute_degree_impl(factor)
            if degree is None:
                return None
            total_degree += degree
        return total_degree

    # Vector expressions
    if isinstance(expr, LinearCombination):
        # Check if vector contains variables (degree 1) or expressions
        if isinstance(expr.vector, VectorVariable):
            return 1
        # Check expressions in vector (VectorExpression case)
        if hasattr(expr.vector, "_expressions"):
            max_deg = 0
            for sub_expr in expr.vector._expressions:  # type: ignore[union-attr]
                d = _compute_degree_impl(sub_expr)
                if d is None:
                    return None
                max_deg = max(max_deg, d)
            return max_deg
        return 1  # Default for unknown vector types

    if isinstance(expr, VectorSum):
        if isinstance(expr.vector, VectorVariable):
            return 1
        if hasattr(expr.vector, "_expressions"):
            max_deg = 0
            for sub_expr in expr.vector._expressions:  # type: ignore[union-attr]
                d = _compute_degree_impl(sub_expr)
                if d is None:
                    return None
                max_deg = max(max_deg, d)
            return max_deg
        return 1  # Default for unknown vector types
    if isinstance(expr, DotProduct):
        # x · y could be quadratic if both are variables
        # For now, return 2 (quadratic) as worst case
        return 2
    if isinstance(expr, QuadraticForm):
        # xᵀAx is always quadratic
        return 2
    if isinstance(expr, VectorPowerSum):
        # sum(x ** k) has degree k
        return int(expr.power)
    if isinstance(expr, VectorUnarySum):
        # sum(sin(x)), sum(exp(x)) etc. are non-polynomial
        return None
    if isinstance(expr, ElementwisePower):
        # x ** k has degree k
        return int(expr.power)
    if isinstance(expr, ElementwiseUnary):
        # sin(x), exp(x) etc. are non-polynomial
        return None

    # Binary operations - early termination on None
    if isinstance(expr, BinaryOp):
        op = expr.op

        # Power operator - check exponent first (often invalid)
        if op == "**":
            if not isinstance(expr.right, Constant):
                return None
            exp_val = expr.right.value
            if not isinstance(exp_val, numbers.Number):
                return None
            exp_float = float(exp_val)
            if not exp_float.is_integer() or exp_float < 0:
                return None
            left_deg = _compute_degree_impl(expr.left)
            if left_deg is None:
                return None
            return left_deg * int(exp_float)

        # Division - check denominator type first
        if op == "/":
            right_deg = _compute_degree_impl(expr.right)
            if right_deg != 0:
                return None
            return _compute_degree_impl(expr.left)

        # Addition/Subtraction - early terminate if either side is None
        if op in ("+", "-"):
            left_deg = _compute_degree_impl(expr.left)
            if left_deg is None:
                return None
            right_deg = _compute_degree_impl(expr.right)
            if right_deg is None:
                return None
            return max(left_deg, right_deg)

        # Multiplication - polynomial degrees add
        if op == "*":
            left_deg = _compute_degree_impl(expr.left)
            if left_deg is None:
                return None
            right_deg = _compute_degree_impl(expr.right)
            if right_deg is None:
                return None
            return left_deg + right_deg

        # Unknown operator
        return None

    # Unary operations
    if isinstance(expr, UnaryOp):
        if expr.op == "neg":
            return _compute_degree_impl(expr.operand)
        return None

    # Unknown node type
    return None


def _check_degree_bounded(expr: Expression, max_degree: int) -> bool:
    """Check if expression degree is at most max_degree.

    Optimized traversal that terminates early when degree exceeds threshold.
    Returns False for non-polynomial expressions.
    """
    result = _check_degree_bounded_impl(expr, max_degree)
    return result is not None and result <= max_degree


def _check_degree_bounded_impl(expr: Expression, max_deg: int) -> Optional[int]:
    """Returns degree if <= max_deg, None if non-polynomial or exceeds bound."""
    # Leaf nodes
    if isinstance(expr, Constant):
        return 0
    if isinstance(expr, Parameter):
        return 0
    if isinstance(expr, Variable):
        return 1 if max_deg >= 1 else None

    if isinstance(expr, NarySum):
        result = 0
        for term in expr.terms:
            degree = _check_degree_bounded_impl(term, max_deg)
            if degree is None:
                return None
            result = max(result, degree)
        return result

    if isinstance(expr, NaryProduct):
        result = 0
        for factor in expr.factors:
            degree = _check_degree_bounded_impl(factor, max_deg - result)
            if degree is None:
                return None
            result += degree
        return result

    # Binary operations
    if isinstance(expr, BinaryOp):
        op = expr.op

        if op == "**":
            if not isinstance(expr.right, Constant):
                return None
            exp_val = expr.right.value
            if not isinstance(exp_val, numbers.Number):
                return None
            exp_float = float(exp_val)
            if not exp_float.is_integer() or exp_float < 0:
                return None
            exp_int = int(exp_float)
            # Early reject: if exponent alone exceeds max, base must be constant
            if exp_int > max_deg:
                left_deg = _check_degree_bounded_impl(expr.left, 0)
                if left_deg != 0:
                    return None
                return 0
            left_deg = _check_degree_bounded_impl(
                expr.left, max_deg // exp_int if exp_int else max_deg
            )
            if left_deg is None:
                return None
            result = left_deg * exp_int
            return result if result <= max_deg else None

        if op == "/":
            right_deg = _check_degree_bounded_impl(expr.right, 0)
            if right_deg != 0:
                return None
            return _check_degree_bounded_impl(expr.left, max_deg)

        if op in ("+", "-"):
            left_deg = _check_degree_bounded_impl(expr.left, max_deg)
            if left_deg is None:
                return None
            right_deg = _check_degree_bounded_impl(expr.right, max_deg)
            if right_deg is None:
                return None
            return max(left_deg, right_deg)

        if op == "*":
            left_deg = _check_degree_bounded_impl(expr.left, max_deg)
            if left_deg is None:
                return None
            # If left is non-constant, right must have degree such that sum <= max_deg
            remaining = max_deg - left_deg if left_deg > 0 else max_deg
            right_deg = _check_degree_bounded_impl(
                expr.right, remaining if left_deg > 0 else max_deg
            )
            if right_deg is None:
                return None
            result = left_deg + right_deg
            return result if result <= max_deg else None

        return None

    # Unary operations
    if isinstance(expr, UnaryOp):
        if expr.op == "neg":
            return _check_degree_bounded_impl(expr.operand, max_deg)
        return None

    return None


def is_linear(expr: Expression) -> bool:
    """Return True if expression is linear (degree ≤ 1).

    Constant expressions are considered linear (degree 0).
    Uses cached degree property on Expression for performance.
    """
    # Use cached degree property on Expression
    deg = expr.degree
    return deg is not None and deg <= 1


def is_quadratic(expr: Expression) -> bool:
    """Return True if expression is quadratic (degree ≤ 2).

    Returns False for non-polynomial expressions.
    Uses cached degree property on Expression for performance.
    """
    deg = expr.degree
    return deg is not None and deg <= 2


def clear_degree_cache() -> None:
    """Clear the memoization cache for degree computation.

    Call this if expressions are being reused across different problems
    and memory usage becomes a concern.
    """
    _compute_degree_cached.cache_clear()


def _get_scalar_constant_value(expr: Expression) -> float | None:
    """Return a live scalar value for a structurally constant expression."""
    if isinstance(expr, Constant):
        raw_value = expr.value
    elif isinstance(expr, Parameter):
        raw_value = expr.value
    elif isinstance(expr, UnaryOp) and expr.op == "neg":
        operand_value = _get_scalar_constant_value(expr.operand)
        return -operand_value if operand_value is not None else None
    elif isinstance(expr, BinaryOp):
        left_value = _get_scalar_constant_value(expr.left)
        if left_value is None:
            return None
        right_value = _get_scalar_constant_value(expr.right)
        if right_value is None:
            return None
        if expr.op == "+":
            return left_value + right_value
        if expr.op == "-":
            return left_value - right_value
        if expr.op == "*":
            return left_value * right_value
        if expr.op == "/":
            return left_value / right_value
        if expr.op == "**":
            return left_value**right_value
        return None
    elif isinstance(expr, NarySum):
        result = 0.0
        for term in expr.terms:
            value = _get_scalar_constant_value(term)
            if value is None:
                return None
            result += value
        return result
    elif isinstance(expr, NaryProduct):
        result = 1.0
        for factor in expr.factors:
            value = _get_scalar_constant_value(factor)
            if value is None:
                return None
            result *= value
        return result
    else:
        return None

    value = np.asarray(raw_value)
    if value.ndim != 0:
        raise NonLinearError(
            expression=repr(expr)[:100],
            context="scalar coefficient extraction",
            suggestion="Use a scalar Parameter or scalar constant as an LP coefficient.",
        )
    return float(value)


# =============================================================================
# Issue #31: LP Coefficient Extraction
# =============================================================================


def extract_linear_coefficient(expr: Expression, var: Variable) -> float:
    """Extract the linear coefficient for a variable from an expression.

    Walks the expression tree and accumulates the coefficient for the
    specified variable. Handles addition, subtraction, scalar multiplication,
    division by constant, and negation.

    Args:
        expr: A linear expression.
        var: The variable to extract the coefficient for.

    Returns:
        The coefficient of the variable in the expression.

    Examples:
        >>> x = Variable("x")
        >>> extract_linear_coefficient(3 * x, x)
        3.0
        >>> extract_linear_coefficient(x + x + x, x)
        3.0
        >>> extract_linear_coefficient(2*x + 3*x, x)
        5.0

    Raises:
        NonLinearError: If the expression is not linear.
    """
    if not is_linear(expr):
        raise NonLinearError(
            expression=repr(expr)[:100],
            context="coefficient extraction",
            suggestion="Ensure all variables appear linearly (no products of variables, powers, or transcendental functions).",
        )
    return _extract_coefficient_impl(expr, var)


def _extract_coefficient_impl(expr: Expression, var: Variable) -> float:
    """Recursive coefficient extraction."""
    from optyx.core.vectors import LinearCombination, VectorSum

    # Constant leaves contribute no variable coefficient.
    if isinstance(expr, (Constant, Parameter)):
        return 0.0

    # Variable - contributes 1 if same variable, 0 otherwise
    if isinstance(expr, Variable):
        return 1.0 if expr.name == var.name else 0.0

    # LinearCombination: c @ x - efficiently extract coefficient
    if isinstance(expr, LinearCombination):
        from optyx.core.vectors import VectorVariable

        if isinstance(expr.vector, VectorVariable):
            for i, v in enumerate(expr.vector._variables):
                if v.name == var.name:
                    return float(expr.coefficients[i])
            return 0.0
        else:
            # VectorExpression - sum coefficients from each element
            total = 0.0
            for i, elem in enumerate(expr.vector._expressions):
                total += float(expr.coefficients[i]) * _extract_coefficient_impl(
                    elem, var
                )
            return total

    # VectorSum: sum(x) - each variable has coefficient 1
    if isinstance(expr, VectorSum):
        for v in expr.vector._variables:
            if v.name == var.name:
                return 1.0
        return 0.0

    # Binary operations
    if isinstance(expr, BinaryOp):
        if expr.op == "+":
            return _extract_coefficient_impl(
                expr.left, var
            ) + _extract_coefficient_impl(expr.right, var)

        if expr.op == "-":
            return _extract_coefficient_impl(
                expr.left, var
            ) - _extract_coefficient_impl(expr.right, var)

        if expr.op == "*":
            # One side must be constant for linear expressions
            left_value = _get_scalar_constant_value(expr.left)
            if left_value is not None:
                return left_value * _extract_coefficient_impl(expr.right, var)
            right_value = _get_scalar_constant_value(expr.right)
            if right_value is not None:
                return _extract_coefficient_impl(expr.left, var) * right_value
            # For linear expressions, at least one side must be constant
            # This fallback handles edge cases where constants are nested
            return 0.0

        if expr.op == "/":
            # Division by constant
            right_value = _get_scalar_constant_value(expr.right)
            if right_value is not None:
                return _extract_coefficient_impl(expr.left, var) / right_value
            return 0.0

        if expr.op == "**":
            # x**0 = 1 (constant), x**1 = x
            if isinstance(expr.right, Constant):
                exp = int(expr.right.value)
                if exp == 0:
                    return 0.0  # Constant term
                if exp == 1:
                    return _extract_coefficient_impl(expr.left, var)
            return 0.0

    # Unary operations
    if isinstance(expr, UnaryOp):
        if expr.op == "neg":
            return -_extract_coefficient_impl(expr.operand, var)
        return 0.0

    if isinstance(expr, NarySum):
        return sum(_extract_coefficient_impl(term, var) for term in expr.terms)

    if isinstance(expr, NaryProduct):
        multiplier = 1.0
        variable_factor: Expression | None = None
        for factor in expr.factors:
            value = _get_scalar_constant_value(factor)
            if value is not None:
                multiplier *= value
            else:
                variable_factor = factor
        if variable_factor is not None:
            return multiplier * _extract_coefficient_impl(variable_factor, var)

    return 0.0


def extract_constant_term(expr: Expression) -> float:
    """Extract the constant term from a linear expression.

    Args:
        expr: A linear expression.

    Returns:
        The constant offset in the expression.

    Examples:
        >>> x = Variable("x")
        >>> extract_constant_term(2*x + 5)
        5.0
        >>> extract_constant_term(x - 3)
        -3.0

    Raises:
        NonLinearError: If the expression is not linear.
    """
    if not is_linear(expr):
        raise NonLinearError(
            expression=repr(expr)[:100],
            context="constant extraction",
            suggestion="Ensure all variables appear linearly.",
        )
    return _extract_constant_impl(expr)


def _extract_constant_impl(expr: Expression) -> float:
    """Recursive constant term extraction."""
    from optyx.core.vectors import LinearCombination, VectorSum

    if isinstance(expr, (Constant, Parameter)):
        constant_value = _get_scalar_constant_value(expr)
        assert constant_value is not None
        return constant_value

    if isinstance(expr, Variable):
        return 0.0

    # Vector expressions have no constant term (purely linear)
    if isinstance(expr, (LinearCombination, VectorSum)):
        return 0.0

    if isinstance(expr, BinaryOp):
        if expr.op == "+":
            return _extract_constant_impl(expr.left) + _extract_constant_impl(
                expr.right
            )

        if expr.op == "-":
            return _extract_constant_impl(expr.left) - _extract_constant_impl(
                expr.right
            )

        if expr.op == "*":
            # c * expr or expr * c
            left_value = _get_scalar_constant_value(expr.left)
            if left_value is not None:
                return left_value * _extract_constant_impl(expr.right)
            right_value = _get_scalar_constant_value(expr.right)
            if right_value is not None:
                return _extract_constant_impl(expr.left) * right_value
            return 0.0

        if expr.op == "/":
            right_value = _get_scalar_constant_value(expr.right)
            if right_value is not None:
                return _extract_constant_impl(expr.left) / right_value
            return 0.0

        if expr.op == "**":
            exponent_value = _get_scalar_constant_value(expr.right)
            if exponent_value is not None:
                exp = int(exponent_value)
                if exp == 0:
                    return 1.0  # x**0 = 1
                base_value = _get_scalar_constant_value(expr.left)
                if base_value is not None:
                    return base_value**exponent_value
            return 0.0

    if isinstance(expr, UnaryOp):
        if expr.op == "neg":
            return -_extract_constant_impl(expr.operand)
        return 0.0

    if isinstance(expr, NarySum):
        return sum(_extract_constant_impl(term) for term in expr.terms)

    if isinstance(expr, NaryProduct):
        result = 1.0
        for factor in expr.factors:
            result *= _extract_constant_impl(factor)
        return result

    return 0.0


@dataclass
class LPData:
    """Data structure containing extracted LP coefficients.

    Attributes:
        c: Objective function coefficients (n,)
        sense: 'min' or 'max'
        A_ub: Inequality constraint matrix (m_ub, n) or None.
            Can be dense ndarray or scipy.sparse matrix.
        b_ub: Inequality RHS vector (m_ub,) or None
        A_eq: Equality constraint matrix (m_eq, n) or None.
            Can be dense ndarray or scipy.sparse matrix.
        b_eq: Equality RHS vector (m_eq,) or None
        bounds: List of (lb, ub) tuples for each variable
        variables: List of variable names in order
    """

    c: NDArray[np.floating]
    sense: str
    A_ub: Any  # NDArray or scipy.sparse matrix or None
    b_ub: NDArray[np.floating] | None
    A_eq: Any  # NDArray or scipy.sparse matrix or None
    b_eq: NDArray[np.floating] | None
    bounds: list[tuple[float | None, float | None]]
    variables: list[str]
    parameter_versions: tuple[tuple[Parameter, int], ...] = ()
    objective_coefficient_signature: tuple[tuple[int, float], ...] = ()

    def parameters_are_current(self) -> bool:
        """Return whether cached numeric data matches all live parameters."""
        return all(
            parameter._version == version
            for parameter, version in self.parameter_versions
        )


def _collect_parameter_versions(problem: Problem) -> tuple[tuple[Parameter, int], ...]:
    """Collect referenced parameters once when numeric LP data is extracted."""
    from optyx.core.vectors import LinearCombination, VectorSum

    roots: list[Expression] = []
    if problem.objective is not None:
        roots.append(problem.objective)
    roots.extend(constraint.expr for constraint in problem.constraints)

    parameters: dict[int, Parameter] = {}
    stack = roots
    while stack:
        node = stack.pop()
        if isinstance(node, Parameter):
            parameters[id(node)] = node
        elif isinstance(node, BinaryOp):
            stack.extend((node.left, node.right))
        elif isinstance(node, UnaryOp):
            stack.append(node.operand)
        elif isinstance(node, NarySum):
            stack.extend(node.terms)
        elif isinstance(node, NaryProduct):
            stack.extend(node.factors)
        elif isinstance(node, (LinearCombination, VectorSum)):
            expressions = getattr(node.vector, "_expressions", None)
            if expressions is not None:
                stack.extend(expressions)

    return tuple((parameter, parameter._version) for parameter in parameters.values())


def extract_all_linear_coefficients(
    expr: Expression,
    var_index: dict[str, int],
    n: int,
) -> NDArray[np.floating]:
    """Extract all linear coefficients from an expression in a single pass.

    This is an O(n) operation that extracts coefficients for all variables
    at once, avoiding the O(n²) cost of calling extract_linear_coefficient
    n times.

    For common patterns like VectorSum(x) where x covers all variables,
    this uses O(1) numpy operations instead of Python loops.

    Args:
        expr: A linear expression.
        var_index: Mapping from variable name to index.
        n: Number of variables.

    Returns:
        Array of coefficients, one per variable.

    Raises:
        NonLinearError: If the expression is not linear.
    """
    from optyx.core.vectors import LinearCombination, VectorSum, VectorVariable

    if not is_linear(expr):
        raise NonLinearError(
            expression=repr(expr)[:100],
            context="batch coefficient extraction",
            suggestion="Ensure all variables appear linearly.",
        )

    # Fast path: VectorSum over VectorVariable covering all variables
    # This is O(1) using numpy instead of O(n) Python loop
    if isinstance(expr, VectorSum) and isinstance(expr.vector, VectorVariable):
        vec_n = expr.vector.size
        if vec_n == n:
            # Check if variables are in order (common case)
            first_name = expr.vector._name_at(0)
            first_idx = var_index.get(first_name, -1)
            if first_idx == 0:
                # All variables in order, return ones directly
                return np.ones(n, dtype=np.float64)

    # Fast path: LinearCombination over VectorVariable covering all variables
    if isinstance(expr, LinearCombination) and isinstance(expr.vector, VectorVariable):
        vec_n = expr.vector.size
        if vec_n == n:
            first_name = expr.vector._name_at(0)
            first_idx = var_index.get(first_name, -1)
            if first_idx == 0:
                # Variables in order, return coefficients directly
                return np.asarray(expr.coefficients, dtype=np.float64).copy()

    # Fast path: BinaryOp with VectorSum/LinearCombination (e.g., x.sum() - k)
    if isinstance(expr, BinaryOp):
        result = _try_extract_fast_binop(expr, var_index, n)
        if result is not None:
            return result

    # General case: O(n) recursive extraction
    result = np.zeros(n, dtype=np.float64)
    _extract_all_coefficients_impl(expr, var_index, result, 1.0)
    return result


def _try_extract_fast_binop(
    expr: BinaryOp,
    var_index: dict[str, int],
    n: int,
) -> NDArray[np.floating] | None:
    """Try to extract coefficients from BinaryOp using fast numpy paths.

    Returns None if fast path not applicable.
    """
    from optyx.core.vectors import LinearCombination, VectorSum, VectorVariable

    # Handle: VectorSum <= constant, VectorSum - constant, etc.
    if expr.op in ("+", "-", "<=", ">=", "=="):
        # Try left side as VectorSum
        if isinstance(expr.left, VectorSum) and isinstance(
            expr.left.vector, VectorVariable
        ):
            vec_n = expr.left.vector.size
            if vec_n == n:
                first_name = expr.left.vector._name_at(0)
                first_idx = var_index.get(first_name, -1)
                if (
                    first_idx == 0
                    and _get_scalar_constant_value(expr.right) is not None
                ):
                    return np.ones(n, dtype=np.float64)

        # Try left side as LinearCombination
        if isinstance(expr.left, LinearCombination) and isinstance(
            expr.left.vector, VectorVariable
        ):
            vec_n = expr.left.vector.size
            if vec_n == n:
                first_name = expr.left.vector._name_at(0)
                first_idx = var_index.get(first_name, -1)
                if (
                    first_idx == 0
                    and _get_scalar_constant_value(expr.right) is not None
                ):
                    return np.asarray(expr.left.coefficients, dtype=np.float64).copy()

    # Handle: constant * VectorSum, VectorSum * constant
    if expr.op == "*":
        left_value = _get_scalar_constant_value(expr.left)
        if left_value is not None:
            if isinstance(expr.right, VectorSum) and isinstance(
                expr.right.vector, VectorVariable
            ):
                vec_n = expr.right.vector.size
                if vec_n == n:
                    first_name = expr.right.vector._name_at(0)
                    first_idx = var_index.get(first_name, -1)
                    if first_idx == 0:
                        return np.full(n, left_value, dtype=np.float64)

        right_value = _get_scalar_constant_value(expr.right)
        if right_value is not None:
            if isinstance(expr.left, VectorSum) and isinstance(
                expr.left.vector, VectorVariable
            ):
                vec_n = expr.left.vector.size
                if vec_n == n:
                    first_name = expr.left.vector._name_at(0)
                    first_idx = var_index.get(first_name, -1)
                    if first_idx == 0:
                        return np.full(n, right_value, dtype=np.float64)

    return None


def _extract_all_coefficients_impl(
    expr: Expression,
    var_index: dict[str, int],
    result: NDArray[np.floating],
    multiplier: float,
) -> None:
    """Recursively extract all coefficients into result array.

    Args:
        expr: Expression to extract from.
        var_index: Mapping from variable name to index.
        result: Output array to accumulate coefficients into.
        multiplier: Current coefficient multiplier from parent expressions.
    """
    from optyx.core.vectors import LinearCombination, VectorSum, VectorVariable

    # Constant leaves have no variable coefficients.
    if isinstance(expr, (Constant, Parameter)):
        return

    # Variable - add coefficient at this variable's index
    if isinstance(expr, Variable):
        idx = var_index.get(expr.name)
        if idx is not None:
            result[idx] += multiplier
        return

    # VectorSum: sum(x) - each variable has coefficient 1 * multiplier
    if isinstance(expr, VectorSum):
        if isinstance(expr.vector, VectorVariable):
            vec_n = expr.vector.size
            first_idx = var_index.get(expr.vector._name_at(0), -1)
            if vec_n == result.size and first_idx == 0:
                result += multiplier
                return
        for name in expr.vector._iter_variable_names():
            idx = var_index.get(name)
            if idx is not None:
                result[idx] += multiplier
        return

    # LinearCombination: c @ x - coefficient is c[i] * multiplier
    if isinstance(expr, LinearCombination):
        if isinstance(expr.vector, VectorVariable):
            vec_n = expr.vector.size
            first_idx = var_index.get(expr.vector._name_at(0), -1)
            if vec_n == result.size and first_idx == 0:
                result += multiplier * np.asarray(expr.coefficients, dtype=np.float64)
                return
            for i, name in enumerate(expr.vector._iter_variable_names()):
                idx = var_index.get(name)
                if idx is not None:
                    result[idx] += float(expr.coefficients[i]) * multiplier
        else:
            # VectorExpression - recurse into each element
            for i, elem in enumerate(expr.vector._expressions):
                coeff = float(expr.coefficients[i]) * multiplier
                _extract_all_coefficients_impl(elem, var_index, result, coeff)
        return

    # Binary operations
    if isinstance(expr, BinaryOp):
        if expr.op == "+":
            _extract_all_coefficients_impl(expr.left, var_index, result, multiplier)
            _extract_all_coefficients_impl(expr.right, var_index, result, multiplier)
            return

        if expr.op == "-":
            _extract_all_coefficients_impl(expr.left, var_index, result, multiplier)
            _extract_all_coefficients_impl(expr.right, var_index, result, -multiplier)
            return

        if expr.op == "*":
            # One side must be constant for linear expressions
            left_value = _get_scalar_constant_value(expr.left)
            if left_value is not None:
                _extract_all_coefficients_impl(
                    expr.right, var_index, result, multiplier * left_value
                )
                return
            right_value = _get_scalar_constant_value(expr.right)
            if right_value is not None:
                _extract_all_coefficients_impl(
                    expr.left, var_index, result, multiplier * right_value
                )
                return
            # Both sides non-constant - no linear contribution
            return

        if expr.op == "/":
            # Division by constant
            right_value = _get_scalar_constant_value(expr.right)
            if right_value is not None:
                _extract_all_coefficients_impl(
                    expr.left, var_index, result, multiplier / right_value
                )
            return

        if expr.op == "**":
            # x**1 = x, x**0 = constant
            if isinstance(expr.right, Constant):
                exp = int(expr.right.value)
                if exp == 1:
                    _extract_all_coefficients_impl(
                        expr.left, var_index, result, multiplier
                    )
            return

    # Unary operations
    if isinstance(expr, UnaryOp):
        if expr.op == "neg":
            _extract_all_coefficients_impl(expr.operand, var_index, result, -multiplier)
        return

    if isinstance(expr, NarySum):
        for term in expr.terms:
            _extract_all_coefficients_impl(term, var_index, result, multiplier)
        return

    if isinstance(expr, NaryProduct):
        product_multiplier = multiplier
        variable_factor: Expression | None = None
        for factor in expr.factors:
            value = _get_scalar_constant_value(factor)
            if value is not None:
                product_multiplier *= value
            else:
                variable_factor = factor
        if variable_factor is not None:
            _extract_all_coefficients_impl(
                variable_factor, var_index, result, product_multiplier
            )


def _vstack(a: Any, b: Any) -> Any:
    """Vertically stack two matrices, preserving sparsity when possible."""
    from scipy import sparse as sp

    if sp.issparse(a) or sp.issparse(b):
        # Convert dense to sparse if needed for concatenation
        if not sp.issparse(a):
            a = sp.csr_matrix(a)
        if not sp.issparse(b):
            b = sp.csr_matrix(b)
        return sp.vstack([a, b], format="csr")
    return np.vstack([a, b])


class LinearProgramExtractor:
    """Extracts LP coefficients from a Problem for use with scipy.optimize.linprog.

    This class walks the expression trees of the objective and constraints,
    extracting the coefficient matrices needed for linear programming solvers.

    Example:
        >>> extractor = LinearProgramExtractor()
        >>> lp_data = extractor.extract(problem)
        >>> result = linprog(c=lp_data.c, A_ub=lp_data.A_ub, b_ub=lp_data.b_ub, ...)
    """

    def extract_objective(
        self, problem: Problem
    ) -> tuple[NDArray[np.floating], str, list[Variable]]:
        """Extract objective coefficients.

        Args:
            problem: The optimization problem.

        Returns:
            Tuple of (c, sense, variables) where:
            - c: coefficient array for each variable
            - sense: 'min' or 'max'
            - variables: ordered list of variables

        Raises:
            ValueError: If objective is not set or not linear.
        """
        if problem.objective is None:
            raise NoObjectiveError(
                suggestion="Call minimize() or maximize() on the problem first.",
            )

        if not is_linear(problem.objective):
            raise NonLinearError(
                expression=repr(problem.objective)[:100],
                context="LP extraction",
                suggestion="The objective must be linear for LP solvers. Use a QP solver for quadratic objectives.",
            )

        variables = problem.variables
        n = len(variables)

        # Build variable name to index mapping
        var_index = {var.name: i for i, var in enumerate(variables)}

        # Use batch extraction - O(n) instead of O(n²)
        c = extract_all_linear_coefficients(problem.objective, var_index, n)

        # Add Variable.obj contributions (linear objective coefficients set at creation)
        for i, var in enumerate(variables):
            if var.obj != 0.0:
                c[i] += var.obj

        sense = "min" if problem.sense == "minimize" else "max"
        return c, sense, variables

    def extract_constraints(
        self, problem: Problem, variables: Sequence[Variable]
    ) -> tuple[
        NDArray[np.floating] | None,
        NDArray[np.floating] | None,
        NDArray[np.floating] | None,
        NDArray[np.floating] | None,
    ]:
        """Extract constraint matrices.

        Args:
            problem: The optimization problem.
            variables: Ordered list of variables (from extract_objective).

        Returns:
            Tuple of (A_ub, b_ub, A_eq, b_eq) where:
            - A_ub: inequality constraint coefficient matrix
            - b_ub: inequality RHS vector
            - A_eq: equality constraint coefficient matrix
            - b_eq: equality RHS vector
            Returns None for matrices with no constraints of that type.

        Raises:
            ValueError: If any constraint is not linear.
        """
        n = len(variables)
        ub_rows: list[NDArray[np.floating]] = []
        ub_rhs: list[float] = []
        eq_rows: list[NDArray[np.floating]] = []
        eq_rhs: list[float] = []

        # Build variable name to index mapping for fast lookup
        var_index = {var.name: i for i, var in enumerate(variables)}

        for constraint in problem.constraints:
            if not is_linear(constraint.expr):
                raise NonLinearError(
                    expression=repr(constraint.expr)[:100],
                    context="LP constraint extraction",
                    suggestion="All constraints must be linear for LP solvers.",
                )

            # Use batch extraction - O(n) instead of O(n²)
            row = extract_all_linear_coefficients(constraint.expr, var_index, n)

            # RHS is the negative of the constant term
            # Constraint form: expr sense 0, where expr = Ax - b
            # So Ax <= b becomes Ax - b <= 0, meaning b = -constant_term
            rhs = -extract_constant_term(constraint.expr)

            if constraint.sense == "==":
                eq_rows.append(row)
                eq_rhs.append(rhs)
            elif constraint.sense == "<=":
                ub_rows.append(row)
                ub_rhs.append(rhs)
            elif constraint.sense == ">=":
                # a >= b becomes -a <= -b
                ub_rows.append(-row)
                ub_rhs.append(-rhs)

        A_ub = np.array(ub_rows, dtype=np.float64) if ub_rows else None
        b_ub = np.array(ub_rhs, dtype=np.float64) if ub_rhs else None
        A_eq = np.array(eq_rows, dtype=np.float64) if eq_rows else None
        b_eq = np.array(eq_rhs, dtype=np.float64) if eq_rhs else None

        # Merge structured matrix constraints collected via subject_to(A @ x <= b)
        A_ub, b_ub, A_eq, b_eq = self._merge_matrix_constraints(
            problem, variables, A_ub, b_ub, A_eq, b_eq
        )

        return A_ub, b_ub, A_eq, b_eq

    @staticmethod
    def _merge_matrix_constraints(
        problem: Problem,
        variables: Sequence[Variable],
        A_ub: Any,
        b_ub: NDArray[np.floating] | None,
        A_eq: Any,
        b_eq: NDArray[np.floating] | None,
    ) -> tuple[Any, NDArray[np.floating] | None, Any, NDArray[np.floating] | None]:
        """Merge matrix constraints into the extracted constraint matrices."""
        from scipy import sparse as sp

        if not problem._matrix_constraints:
            return A_ub, b_ub, A_eq, b_eq

        n = len(variables)
        var_index = {var.name: i for i, var in enumerate(variables)}

        for mc in problem._matrix_constraints:
            # Build column permutation: mc.variables may be a subset or
            # reordered relative to the full variable list
            mc_n = len(mc.variables)
            col_indices = np.array(
                [var_index[v.name] for v in mc.variables], dtype=np.intp
            )

            # Build the full-width matrix for this constraint block
            if sp.issparse(mc.A):
                if mc_n == n and np.array_equal(col_indices, np.arange(n)):
                    A_full = mc.A  # noqa: N806
                else:
                    # Permutation matrix P (mc_n x n): P[j, col_indices[j]] = 1
                    # A_full = mc.A @ P  ->  (m, mc_n) @ (mc_n, n) = (m, n)
                    P = sp.csr_matrix(  # noqa: N806
                        (np.ones(mc_n), (np.arange(mc_n), col_indices)),
                        shape=(mc_n, n),
                    )
                    A_full = (mc.A @ P).tocsr()  # noqa: N806
            else:
                # Dense path
                if mc_n == n and np.array_equal(col_indices, np.arange(n)):
                    # Variables already aligned — zero-copy
                    A_full = mc.A  # noqa: N806
                else:
                    A_full = np.zeros((mc.A.shape[0], n), dtype=np.float64)  # noqa: N806
                    A_full[:, col_indices] = mc.A

            b_block = mc.b

            if mc.sense == ">=":
                A_full = -A_full  # noqa: N806
                b_block = -b_block

            if mc.sense == "==":
                if A_eq is None:
                    A_eq = A_full
                    b_eq = b_block
                else:
                    A_eq = _vstack(A_eq, A_full)
                    b_eq = np.concatenate([b_eq, b_block])  # type: ignore[arg-type]
            else:
                # <= (including >= converted to <=)
                if A_ub is None:
                    A_ub = A_full
                    b_ub = b_block
                else:
                    A_ub = _vstack(A_ub, A_full)
                    b_ub = np.concatenate([b_ub, b_block])  # type: ignore[arg-type]

        return A_ub, b_ub, A_eq, b_eq

    def extract_bounds(
        self, variables: Sequence[Variable]
    ) -> list[tuple[float | None, float | None]]:
        """Extract variable bounds.

        Args:
            variables: Ordered list of variables.

        Returns:
            List of (lb, ub) tuples for each variable.
            Uses None for unbounded directions.
        """
        bounds: list[tuple[float | None, float | None]] = []
        for var in variables:
            lb = var.lb if var.lb is not None else None
            ub = var.ub if var.ub is not None else None
            bounds.append((lb, ub))
        return bounds

    def extract(self, problem: Problem) -> LPData:
        """Extract complete LP specification from a problem.

        Args:
            problem: The optimization problem.

        Returns:
            LPData containing all coefficients needed for linprog().

        Raises:
            ValueError: If problem is not a valid LP.
        """
        parameter_versions = _collect_parameter_versions(problem)
        source_vector = problem._single_vector_source()
        if source_vector is not None:
            n = source_vector.size
            names: list[str] = []
            bounds: list[tuple[float | None, float | None]] = []
            obj_terms = np.zeros(n, dtype=np.float64)
            var_index: dict[str, int] = {}

            for index, (name, bound_pair, _, obj_coeff) in enumerate(
                source_vector._iter_lp_metadata()
            ):
                names.append(name)
                bounds.append(bound_pair)
                var_index[name] = index
                if obj_coeff != 0.0:
                    obj_terms[index] = obj_coeff

            assert problem.objective is not None
            c = extract_all_linear_coefficients(problem.objective, var_index, n)
            if np.any(obj_terms):
                c = c + obj_terms

            ub_rows: list[NDArray[np.floating]] = []
            ub_rhs: list[float] = []
            eq_rows: list[NDArray[np.floating]] = []
            eq_rhs: list[float] = []

            for constraint in problem.constraints:
                if not is_linear(constraint.expr):
                    raise NonLinearError(
                        expression=repr(constraint.expr)[:100],
                        context="LP constraint extraction",
                        suggestion="All constraints must be linear for LP solvers.",
                    )

                row = extract_all_linear_coefficients(constraint.expr, var_index, n)
                rhs = -extract_constant_term(constraint.expr)

                if constraint.sense == "==":
                    eq_rows.append(row)
                    eq_rhs.append(rhs)
                elif constraint.sense == "<=":
                    ub_rows.append(row)
                    ub_rhs.append(rhs)
                elif constraint.sense == ">=":
                    ub_rows.append(-row)
                    ub_rhs.append(-rhs)

            A_ub = np.array(ub_rows, dtype=np.float64) if ub_rows else None
            b_ub = np.array(ub_rhs, dtype=np.float64) if ub_rhs else None
            A_eq = np.array(eq_rows, dtype=np.float64) if eq_rows else None
            b_eq = np.array(eq_rhs, dtype=np.float64) if eq_rhs else None

            return LPData(
                c=c,
                sense="min" if problem.sense == "minimize" else "max",
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                bounds=bounds,
                variables=names,
                parameter_versions=parameter_versions,
                objective_coefficient_signature=problem._objective_coefficient_signature(),
            )

        c, sense, variables = self.extract_objective(problem)
        A_ub, b_ub, A_eq, b_eq = self.extract_constraints(problem, variables)
        bounds = self.extract_bounds(variables)

        return LPData(
            c=c,
            sense=sense,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            variables=[v.name for v in variables],
            parameter_versions=parameter_versions,
            objective_coefficient_signature=problem._objective_coefficient_signature(),
        )


# =============================================================================
# Issue #106: Quadratic Coefficient Extraction
# =============================================================================


def extract_quadratic_coefficients(
    expr: Expression,
    variables: list[Variable],
) -> NDArray[np.floating]:
    """Extract the quadratic coefficient matrix from a quadratic expression.

    For an expression of the form x'Qx + c'x + d, returns the matrix Q
    such that the quadratic part is sum_{i,j} Q[i,j] * x_i * x_j.

    Args:
        expr: A quadratic expression.
        variables: List of variables in the desired ordering.

    Returns:
        Symmetric (n, n) matrix of quadratic coefficients.

    Raises:
        NonLinearError: If the expression is not quadratic.
    """
    from optyx.io import _is_at_most_quadratic, _collect_quadratic_coefficients

    if not _is_at_most_quadratic(expr):
        raise NonLinearError(
            expression=repr(expr)[:100],
            context="quadratic coefficient extraction",
            suggestion="Ensure the expression is at most quadratic.",
        )

    n = len(variables)
    var_index = {v.name: i for i, v in enumerate(variables)}
    Q = np.zeros((n, n), dtype=np.float64)
    _collect_quadratic_coefficients(expr, var_index, Q, 1.0)

    # Symmetrize: Q_sym = (Q + Q.T) / 2
    Q_sym = (Q + Q.T) / 2.0
    return Q_sym


# =============================================================================
# Issue #32: Constraint Helpers and Classification
# =============================================================================


def is_simple_bound(constraint: Constraint, variables: Sequence[Variable]) -> bool:
    """Check if a constraint represents a simple variable bound.

    A simple bound is a constraint involving only one variable and a constant,
    such as: x >= 0, x <= 10, x == 5.

    Args:
        constraint: The constraint to check.
        variables: List of all variables in the problem.

    Returns:
        True if the constraint is a simple bound on a single variable.

    Examples:
        >>> x = Variable("x")
        >>> y = Variable("y")
        >>> is_simple_bound(x >= 0, [x, y])  # True
        >>> is_simple_bound(x + y <= 10, [x, y])  # False
    """
    if not is_linear(constraint.expr):
        return False

    # Count non-zero coefficients
    nonzero_count = 0
    for var in variables:
        coef = extract_linear_coefficient(constraint.expr, var)
        if abs(coef) > 1e-10:
            nonzero_count += 1
            if nonzero_count > 1:
                return False

    return nonzero_count == 1


@dataclass
class ConstraintClassification:
    """Classification of constraints in a problem.

    Attributes:
        n_equality: Number of equality constraints
        n_inequality: Number of inequality constraints (<=, >=)
        n_simple_bounds: Number of constraints that are simple variable bounds
        n_general: Number of general constraints (not simple bounds)
        equality_indices: Indices of equality constraints
        inequality_indices: Indices of inequality constraints
        simple_bound_indices: Indices of simple bound constraints
    """

    n_equality: int
    n_inequality: int
    n_simple_bounds: int
    n_general: int
    equality_indices: list[int]
    inequality_indices: list[int]
    simple_bound_indices: list[int]


def classify_constraints(
    constraints: Sequence[Constraint], variables: Sequence[Variable]
) -> ConstraintClassification:
    """Classify constraints by type.

    Analyzes constraints and categorizes them as equality, inequality,
    simple bounds, or general constraints.

    Args:
        constraints: List of constraints to classify.
        variables: List of all variables in the problem.

    Returns:
        ConstraintClassification with counts and indices.

    Examples:
        >>> x = Variable("x")
        >>> y = Variable("y")
        >>> constraints = [x >= 0, x + y <= 10, x == y]
        >>> result = classify_constraints(constraints, [x, y])
        >>> result.n_simple_bounds
        1
        >>> result.n_equality
        1
    """
    equality_indices: list[int] = []
    inequality_indices: list[int] = []
    simple_bound_indices: list[int] = []

    for i, constraint in enumerate(constraints):
        if constraint.sense == "==":
            equality_indices.append(i)
        else:
            inequality_indices.append(i)

        if is_simple_bound(constraint, variables):
            simple_bound_indices.append(i)

    n_general = len(constraints) - len(simple_bound_indices)

    return ConstraintClassification(
        n_equality=len(equality_indices),
        n_inequality=len(inequality_indices),
        n_simple_bounds=len(simple_bound_indices),
        n_general=n_general,
        equality_indices=equality_indices,
        inequality_indices=inequality_indices,
        simple_bound_indices=simple_bound_indices,
    )
