"""Problem analysis utilities.

Provides linear / quadratic detection and helpers to compute polynomial degree
of expression trees. These utilities are used to detect LP/QP problems for
fast-path solver selection.

Performance optimizations:
- Early termination: stops traversal immediately when non-polynomial detected
- Degree-bounded traversal: is_linear/is_quadratic stop when threshold exceeded
- Memoization: caches results for repeated sub-expressions (common in constraints)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Optional, Sequence
import numbers

import numpy as np
from numpy.typing import NDArray

from optyx.core.expressions import Expression, Constant, Variable, BinaryOp, UnaryOp

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
    """
    return _compute_degree_cached(id(expr), expr)


@lru_cache(maxsize=1024)
def _compute_degree_cached(expr_id: int, expr: Expression) -> Optional[int]:
    """Memoized degree computation keyed by expression object id."""
    return _compute_degree_impl(expr)


def _compute_degree_impl(expr: Expression) -> Optional[int]:
    """Core degree computation with early termination."""
    # Fast path: leaf nodes (most common)
    if isinstance(expr, Constant):
        return 0
    if isinstance(expr, Variable):
        return 1

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
            if not isinstance(expr.right, Constant):
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

        # Multiplication - only allow scalar * polynomial
        if op == "*":
            left_deg = _compute_degree_impl(expr.left)
            if left_deg is None:
                return None
            right_deg = _compute_degree_impl(expr.right)
            if right_deg is None:
                return None
            # x*y (both degree >= 1) is non-polynomial for LP detection
            if left_deg > 0 and right_deg > 0:
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
    if isinstance(expr, Variable):
        return 1 if max_deg >= 1 else None

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
            if not isinstance(expr.right, Constant):
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
            # x*y is non-polynomial for LP detection
            if left_deg > 0 and right_deg > 0:
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
        ValueError: If the expression is not linear.
    """
    if not is_linear(expr):
        raise ValueError("Expression must be linear for coefficient extraction")
    return _extract_coefficient_impl(expr, var)


def _extract_coefficient_impl(expr: Expression, var: Variable) -> float:
    """Recursive coefficient extraction."""
    # Constant - contributes 0 to variable coefficient
    if isinstance(expr, Constant):
        return 0.0

    # Variable - contributes 1 if same variable, 0 otherwise
    if isinstance(expr, Variable):
        return 1.0 if expr.name == var.name else 0.0

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
            if isinstance(expr.left, Constant):
                return float(expr.left.value) * _extract_coefficient_impl(
                    expr.right, var
                )
            if isinstance(expr.right, Constant):
                return _extract_coefficient_impl(expr.left, var) * float(
                    expr.right.value
                )
            # For linear expressions, at least one side must be constant
            # This fallback handles edge cases where constants are nested
            return 0.0

        if expr.op == "/":
            # Division by constant
            if isinstance(expr.right, Constant):
                return _extract_coefficient_impl(expr.left, var) / float(
                    expr.right.value
                )
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
        ValueError: If the expression is not linear.
    """
    if not is_linear(expr):
        raise ValueError("Expression must be linear for constant extraction")
    return _extract_constant_impl(expr)


def _extract_constant_impl(expr: Expression) -> float:
    """Recursive constant term extraction."""
    if isinstance(expr, Constant):
        return float(expr.value)

    if isinstance(expr, Variable):
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
            if isinstance(expr.left, Constant):
                return float(expr.left.value) * _extract_constant_impl(expr.right)
            if isinstance(expr.right, Constant):
                return _extract_constant_impl(expr.left) * float(expr.right.value)
            return 0.0

        if expr.op == "/":
            if isinstance(expr.right, Constant):
                return _extract_constant_impl(expr.left) / float(expr.right.value)
            return 0.0

        if expr.op == "**":
            if isinstance(expr.right, Constant):
                exp = int(expr.right.value)
                if exp == 0:
                    return 1.0  # x**0 = 1
            return 0.0

    if isinstance(expr, UnaryOp):
        if expr.op == "neg":
            return -_extract_constant_impl(expr.operand)
        return 0.0

    return 0.0


@dataclass
class LPData:
    """Data structure containing extracted LP coefficients.

    Attributes:
        c: Objective function coefficients (n,)
        sense: 'min' or 'max'
        A_ub: Inequality constraint matrix (m_ub, n) or None
        b_ub: Inequality RHS vector (m_ub,) or None
        A_eq: Equality constraint matrix (m_eq, n) or None
        b_eq: Equality RHS vector (m_eq,) or None
        bounds: List of (lb, ub) tuples for each variable
        variables: List of variable names in order
    """

    c: NDArray[np.floating]
    sense: str
    A_ub: NDArray[np.floating] | None
    b_ub: NDArray[np.floating] | None
    A_eq: NDArray[np.floating] | None
    b_eq: NDArray[np.floating] | None
    bounds: list[tuple[float | None, float | None]]
    variables: list[str]


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
            raise ValueError("Problem has no objective function")

        if not is_linear(problem.objective):
            raise ValueError("Objective function is not linear")

        variables = problem.variables
        n = len(variables)
        c = np.zeros(n, dtype=np.float64)

        for i, var in enumerate(variables):
            c[i] = extract_linear_coefficient(problem.objective, var)

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

        for constraint in problem.constraints:
            if not is_linear(constraint.expr):
                raise ValueError(f"Constraint is not linear: {constraint}")

            # Extract coefficients for this constraint
            row = np.zeros(n, dtype=np.float64)
            for i, var in enumerate(variables):
                row[i] = extract_linear_coefficient(constraint.expr, var)

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
        )


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
