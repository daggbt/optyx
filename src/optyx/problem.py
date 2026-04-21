"""Problem class for defining optimization problems.

Provides a fluent API for building optimization problems:

    prob = Problem()
    prob.minimize(x**2 + y**2)
    prob.subject_to(x + y >= 1)
    solution = prob.solve()
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Literal, Iterable
from types import TracebackType

import numpy as np

from optyx.core.errors import (
    InvalidOperationError,
    ConstraintError,
    NoObjectiveError,
    UnsupportedOperationError,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from optyx.analysis import LPData
    from optyx.constraints import Constraint, MatrixConstraintBlock
    from optyx.core.expressions import Expression, Variable
    from optyx.core.vectors import VectorVariable
    from optyx.solution import Solution, SolverProgress


@dataclass
class _MatrixConstraint:
    """Stores a batch of linear constraints in matrix form: A @ x sense b."""

    A: Any  # NDArray or scipy.sparse matrix
    b: NDArray[np.floating]
    sense: Literal["<=", ">=", "=="]
    variables: list[Variable]  # The VectorVariable's individual variables


# Threshold for "small" problems where gradient-free methods are faster
SMALL_PROBLEM_THRESHOLD = 3

# Threshold for "large" problems where memory-efficient methods are preferred
LARGE_PROBLEM_THRESHOLD = 1000

# Thresholds for preferring trust-constr on large sparse NLPs
SPARSE_NLP_ROW_THRESHOLD = 32
SPARSE_NLP_VARIABLE_THRESHOLD = 64
SPARSE_NLP_DENSITY_THRESHOLD = 0.15
SPARSE_MATRIX_ENTRY_THRESHOLD = 4096

# Pre-compiled regex for natural sorting of variable names
_NUMBER_SPLIT_RE = re.compile(r"(\d+)")


def _natural_sort_key(var: Variable) -> tuple:
    """Generate a sort key for natural ordering of variable names.

    Handles variable names like 'x[0]', 'x[10]', 'A[1,2]' so they
    sort numerically rather than lexicographically.

    Examples:
        x[0], x[1], x[2], ..., x[10]  (not x[0], x[1], x[10], x[2])
        A[0,0], A[0,1], A[0,2], A[1,0], A[1,1], ...

    Returns:
        Tuple for sorting: (base_name, index1, index2, ...)
    """
    if hasattr(var, "_sort_key"):
        return var._sort_key

    name = var.name
    # Split into text and number parts
    parts = _NUMBER_SPLIT_RE.split(name)
    # Convert number parts to integers for proper numeric sorting
    return tuple(int(p) if p.isdigit() else p for p in parts)


def _try_get_single_vector_source(expr: "Expression") -> "VectorVariable | None":
    """Try to extract the single VectorVariable that an expression depends on.

    Returns the VectorVariable if the expression only uses variables from one
    VectorVariable, otherwise None. This enables O(1) variable extraction
    for common single-VectorVariable problems.

    Uses iterative traversal to handle deep expression trees without hitting
    Python's recursion limit.
    """
    from optyx.core.vectors import (
        VectorVariable,
        LinearCombination,
        VectorSum,
        VectorPowerSum,
        VectorUnarySum,
        VectorExpressionSum,
        DotProduct,
    )
    from optyx.core.expressions import BinaryOp, UnaryOp, Constant
    from optyx.core.parameters import Parameter

    # Iterative traversal using explicit stack
    stack: list[Expression] = [expr]
    found_source: VectorVariable | None = None

    while stack:
        current = stack.pop()

        # Skip constants and parameters - they don't contribute variables
        if isinstance(current, (Constant, Parameter)):
            continue

        # VectorSum - vector might be VectorVariable
        if isinstance(current, VectorSum):
            if isinstance(current.vector, VectorVariable):
                candidate = current.vector
                if found_source is None:
                    found_source = candidate
                elif found_source is not candidate:
                    return None  # Multiple different sources
            else:
                return None  # VectorExpression, not simple VectorVariable
            continue

        # LinearCombination (e.g., c @ x)
        if isinstance(current, LinearCombination):
            if isinstance(current.vector, VectorVariable):
                candidate = current.vector
                if found_source is None:
                    found_source = candidate
                elif found_source is not candidate:
                    return None
            else:
                return None  # VectorExpression
            continue

        # VectorPowerSum (sum(x ** k)) - vector is always VectorVariable
        if isinstance(current, VectorPowerSum):
            candidate = current.vector
            if found_source is None:
                found_source = candidate
            elif found_source is not candidate:
                return None
            continue

        # VectorUnarySum (sum(f(x))) - vector is always VectorVariable
        if isinstance(current, VectorUnarySum):
            candidate = current.vector
            if found_source is None:
                found_source = candidate
            elif found_source is not candidate:
                return None
            continue

        # DotProduct (x.dot(y)) - check if both sides are same VectorVariable
        if isinstance(current, DotProduct):
            left = current.left
            right = current.right
            is_left_vec = isinstance(left, VectorVariable)
            is_right_vec = isinstance(right, VectorVariable)

            if is_left_vec and is_right_vec:
                # Both are VectorVariables - must be the same
                if left is right:
                    candidate = left
                    if found_source is None:
                        found_source = candidate
                    elif found_source is not candidate:
                        return None
                else:
                    return None  # Two different VectorVariables
            elif is_left_vec or is_right_vec:
                # One is VectorVariable, one is VectorExpression
                return None  # Complex case, bail out
            else:
                # Both are VectorExpressions - too complex
                return None
            continue

        # VectorExpressionSum - push all element expressions to stack
        if isinstance(current, VectorExpressionSum):
            stack.extend(current.expression._expressions)
            continue

        # BinaryOp - push both children to stack
        if isinstance(current, BinaryOp):
            stack.append(current.left)
            stack.append(current.right)
            continue

        # UnaryOp - push operand to stack
        if isinstance(current, UnaryOp):
            stack.append(current.operand)
            continue

        # Any other type (e.g., scalar Variable) - not a vector source
        return None

    return found_source


class Problem:
    """An optimization problem with objective and constraints.

    Example:
        >>> x = Variable("x", lb=0)
        >>> y = Variable("y", lb=0)
        >>> prob = Problem()
        >>> prob.minimize(x**2 + y**2)
        >>> prob.subject_to(x + y >= 1)
        >>> solution = prob.solve()
        >>> print(solution.values)  # {'x': 0.5, 'y': 0.5}

    Note:
        The Problem class is not thread-safe. Compiled callables are cached
        per instance and reused across multiple solve() calls for performance.
        Any mutation (adding constraints, changing objective) invalidates the cache.
    """

    def __init__(self, name: str | None = None):
        """Create a new optimization problem.

        Args:
            name: Optional name for the problem.
        """
        self.name = name
        self._objective: Expression | None = None
        self._sense: Literal["minimize", "maximize"] = "minimize"
        self._constraints: list[Constraint] = []
        self._matrix_constraints: list[_MatrixConstraint] = []
        self._variables: list[Variable] | None = None  # Cached
        # Solver cache for compiled callables (reused across solve() calls)
        self._solver_cache: dict | None = None
        # LP data cache (reused across solve() calls for LP problems)
        self._lp_cache: LPData | None = None
        # Cached linearity check result (None = not computed, True/False = result)
        self._is_linear_cache: bool | None = None
        # Warm start: last solution array (used as x0 on re-solve)
        self._last_solution: NDArray[np.floating] | None = None

    def _invalidate_caches(self) -> None:
        """Invalidate all cached data when problem is modified."""
        self._variables = None
        self._solver_cache = None
        self._lp_cache = None
        self._is_linear_cache = None

    def _invalidate_constraint_caches(self) -> None:
        """Invalidate only constraint-related caches.

        Preserves objective/gradient compiled callables in the solver cache
        so they don't need to be recompiled when only constraints change.
        """
        self._variables = None
        self._lp_cache = None
        self._is_linear_cache = None
        # Remove only constraint keys from solver cache, keeping obj_fn/grad_fn/hess_fn
        if self._solver_cache is not None:
            for key in (
                "scipy_constraints",
                "linear_constraints",
                "sparse_constraint_jac_fn",
                "constraint_exprs",
                "constraint_fns",
                "constraint_senses",
                "constraint_variables",
            ):
                self._solver_cache.pop(key, None)

    def minimize(self, expr: Expression | float | int) -> Problem:
        """Set the objective function to minimize.

        Args:
            expr: Expression to minimize. Must be an optyx Expression,
                Variable, or numeric constant (int/float).

        Returns:
            Self for method chaining.

        Raises:
            InvalidOperationError: If expr is not a valid expression type.

        Example:
            >>> prob.minimize(x**2 + y**2)
            >>> prob.minimize(x + 2*y - 5)
        """
        self._objective = self._validate_expression(expr, "minimize")
        self._sense = "minimize"
        self._invalidate_caches()
        return self

    def maximize(self, expr: Expression | float | int) -> Problem:
        """Set the objective function to maximize.

        Args:
            expr: Expression to maximize. Must be an optyx Expression,
                Variable, or numeric constant (int/float).

        Returns:
            Self for method chaining.

        Raises:
            InvalidOperationError: If expr is not a valid expression type.

        Example:
            >>> prob.maximize(revenue - cost)
        """
        self._objective = self._validate_expression(expr, "maximize")
        self._sense = "maximize"
        self._invalidate_caches()
        return self

    def _append_matrix_constraint(self, block: MatrixConstraintBlock) -> None:
        from scipy import sparse as sp

        A = block.A
        b_arr = np.asarray(block.b, dtype=np.float64).ravel()

        if sp.issparse(A):
            m, n = A.shape
        else:
            A = np.asarray(A, dtype=np.float64)
            m, n = A.shape

        variables = list(block.variables)
        if n != len(variables):
            raise ValueError(f"A has {n} columns but x has {len(variables)} variables")
        if m != len(b_arr):
            raise ValueError(f"A has {m} rows but b has {len(b_arr)} elements")
        if block.sense not in ("<=", ">=", "=="):
            raise ValueError(f"sense must be '<=', '>=', or '==', got '{block.sense}'")

        self._matrix_constraints.append(
            _MatrixConstraint(
                A=A,
                b=b_arr,
                sense=block.sense,
                variables=variables,
            )
        )

    def subject_to(
        self,
        constraint: Constraint
        | MatrixConstraintBlock
        | Iterable[Constraint | MatrixConstraintBlock],
    ) -> Problem:
        """Add a constraint or list of constraints to the problem.

        Args:
            constraint: Constraint or iterable of constraints to add.
                Accepts lists, tuples, generators, etc.

        Returns:
            Self for method chaining.

        Raises:
            ConstraintError: If constraint is not a valid Constraint type.

        Example:
            >>> x = VectorVariable("x", 100)
            >>> prob.subject_to(x >= 0)  # Adds 100 constraints
            >>> prob.subject_to(x[i] >= 0 for i in range(10))  # Generator
        """
        from optyx.constraints import (
            Constraint as ConstraintType,
            MatrixConstraintBlock as MatrixConstraintBlockType,
        )

        if isinstance(constraint, ConstraintType):
            self._constraints.append(self._validate_constraint(constraint))
        elif isinstance(constraint, MatrixConstraintBlockType):
            self._append_matrix_constraint(constraint)
        elif isinstance(constraint, Iterable):
            for c in constraint:
                if isinstance(c, ConstraintType):
                    self._constraints.append(self._validate_constraint(c))
                elif isinstance(c, MatrixConstraintBlockType):
                    self._append_matrix_constraint(c)
                else:
                    self._constraints.append(self._validate_constraint(c))
        else:
            # Fallback
            from optyx.core.expressions import Expression

            if isinstance(constraint, Expression):
                reason = f"Expected Constraint, got Expression ({type(constraint).__name__}). Did you forget a comparison operator (==, <=, >=)?"
            else:
                reason = f"Expected Constraint or iterable of Constraints, got {type(constraint).__name__}"

            raise ConstraintError(
                message=reason,
                constraint_expr=str(constraint),
            )
        self._invalidate_constraint_caches()
        return self

    def remove_constraint(self, index_or_name: int | str) -> Problem:
        """Remove a constraint by index or name.

        Args:
            index_or_name: If int, removes the constraint at that index.
                If str, removes the first constraint with that name.

        Returns:
            Self for method chaining.

        Raises:
            IndexError: If integer index is out of range.
            KeyError: If no constraint with the given name is found.

        Example:
            >>> prob.subject_to((x + y <= 10).name == "cap")
            >>> prob.remove_constraint("cap")
            >>> prob.remove_constraint(0)  # Remove first constraint
        """
        if isinstance(index_or_name, int):
            idx = index_or_name
            if idx < 0 or idx >= len(self._constraints):
                raise IndexError(
                    f"Constraint index {idx} out of range "
                    f"(problem has {len(self._constraints)} constraints)"
                )
            self._constraints.pop(idx)
        elif isinstance(index_or_name, str):
            name = index_or_name
            for i, c in enumerate(self._constraints):
                if c.name == name:
                    self._constraints.pop(i)
                    break
            else:
                raise KeyError(
                    f"No constraint named '{name}' found. "
                    f"Named constraints: {[c.name for c in self._constraints if c.name]}"
                )
        else:
            raise TypeError(f"Expected int or str, got {type(index_or_name).__name__}")
        self._invalidate_constraint_caches()
        return self

    def __enter__(self) -> Problem:
        """Context manager support."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Context manager exit (no-op)."""
        pass

    def reset(self) -> None:
        """Reset the problem solver state (clears caches and warm start).

        Forces a complete re-analysis and re-compilation of the problem
        on the next solve() call. Also clears any stored warm start state,
        forcing a cold start on the next solve.
        """
        self._invalidate_caches()
        self._last_solution = None

    def _validate_expression(
        self, expr: Expression | float | int, context: str
    ) -> Expression:
        """Validate that expr is a valid Expression type.

        Args:
            expr: The expression to validate.
            context: Context for error message (e.g., "minimize").

        Returns:
            The expression if valid.

        Raises:
            InvalidOperationError: If expr is not a valid expression.
        """
        # Import here to avoid circular imports
        from optyx.core.expressions import Expression as ExprBase

        # Allow numeric constants (they can be used as trivial objectives)
        if isinstance(expr, (int, float)):
            from optyx.core.expressions import Constant

            return Constant(expr)

        # Check for Expression subclass
        if isinstance(expr, ExprBase):
            return expr

        # Invalid type
        raise InvalidOperationError(
            operation=context,
            operand_types=type(expr),
            reason=f"Expected an Expression or numeric value, got {type(expr).__name__}",
            suggestion=f"Use Variable, Expression, or numeric constant. "
            f"Example: prob.{context}(x**2 + y)",
        )

    def _validate_constraint(self, constraint: Constraint) -> Constraint:
        """Validate that constraint is a valid Constraint type.

        Args:
            constraint: The constraint to validate.

        Returns:
            The constraint if valid.

        Raises:
            ConstraintError: If constraint is not valid.
        """
        # Import here to avoid circular imports
        from optyx.constraints import Constraint as ConstraintType

        if isinstance(constraint, ConstraintType):
            return constraint

        # Common mistake: passing expression instead of constraint
        from optyx.core.expressions import Expression as ExprBase

        if isinstance(constraint, ExprBase):
            raise ConstraintError(
                message="Got an Expression instead of a Constraint. "
                "Did you forget a comparison operator?",
                constraint_expr=str(constraint),
            )

        # String or other invalid type
        raise ConstraintError(
            message=f"Expected a Constraint, got {type(constraint).__name__}",
            constraint_expr=str(constraint)
            if not isinstance(constraint, str)
            else f"'{constraint}'",
        )

    @property
    def objective(self) -> Expression | None:
        """The objective function expression."""
        return self._objective

    @property
    def sense(self) -> Literal["minimize", "maximize"]:
        """The optimization sense (minimize or maximize)."""
        return self._sense

    @property
    def constraints(self) -> list[Constraint]:
        """List of constraints."""
        return self._constraints.copy()

    @property
    def variables(self) -> list[Variable]:
        """All decision variables in the problem.

        Automatically extracted from objective and constraints.
        Sorted using natural ordering for consistent, deterministic results.

        Variable Ordering:
            - Variables are sorted by name using natural ordering
            - VectorVariable elements: x[0], x[1], ..., x[10] (numeric order)
            - MatrixVariable elements: A[0,0], A[0,1], ..., A[1,0] (row-major)
            - This ordering is used by the solver for flattening and is
              guaranteed to be deterministic across runs.
        """
        if self._variables is not None:
            return self._variables

        from optyx.core.expressions import get_all_variables

        # Fast path: check if objective is based on a single VectorVariable
        # In this case, we can skip the expensive set operations and sorting
        if self._objective is not None:
            source_vector = _try_get_single_vector_source(self._objective)
            if source_vector is not None:
                # Check if all constraints use the same VectorVariable
                all_same = True
                for constraint in self._constraints:
                    constraint_source = _try_get_single_vector_source(constraint.expr)
                    if (
                        constraint_source is None
                        or constraint_source is not source_vector
                    ):
                        all_same = False
                        break

                if all_same:
                    # All variables from one VectorVariable - already in order!
                    self._variables = list(source_vector._variables)
                    return self._variables

        # General case: collect from all expressions and sort
        all_vars: set[Variable] = set()

        if self._objective is not None:
            all_vars.update(get_all_variables(self._objective))

        for constraint in self._constraints:
            all_vars.update(constraint.get_variables())

        for mc in self._matrix_constraints:
            all_vars.update(mc.variables)

        self._variables = sorted(all_vars, key=_natural_sort_key)
        return self._variables

    @property
    def n_variables(self) -> int:
        """Number of decision variables."""
        return len(self.variables)

    @property
    def n_constraints(self) -> int:
        """Number of constraints."""
        n = len(self._constraints)
        for mc in self._matrix_constraints:
            n += mc.A.shape[0]
        return n

    def get_bounds(self) -> list[tuple[float | None, float | None]]:
        """Get variable bounds as a list of (lb, ub) tuples.

        Returns:
            List of bounds in variable order.
        """
        return [(v.lb, v.ub) for v in self.variables]

    def _is_linear_problem(self) -> bool:
        """Check if the problem is a linear program.

        Returns True if both the objective and all constraints are linear.
        Result is cached until problem is modified.
        """
        # Return cached result if available
        if self._is_linear_cache is not None:
            return self._is_linear_cache

        from optyx.analysis import is_linear

        if self._objective is None:
            self._is_linear_cache = False
            return False

        if not is_linear(self._objective):
            self._is_linear_cache = False
            return False

        for constraint in self._constraints:
            if not is_linear(constraint.expr):
                self._is_linear_cache = False
                return False

        # Matrix constraints are always linear by definition
        self._is_linear_cache = True
        return True

    def _only_simple_bounds(self) -> bool:
        """Check if all constraints are simple variable bounds.

        Simple bounds are constraints on a single variable like x >= 0 or x <= 10.
        """
        if self._matrix_constraints:
            return False

        if not self._constraints:
            return True

        from optyx.analysis import is_simple_bound

        return all(is_simple_bound(c, self.variables) for c in self._constraints)

    def _has_equality_constraints(self) -> bool:
        """Check if problem has any equality constraints."""
        return any(c.sense == "==" for c in self._constraints) or any(
            mc.sense == "==" for mc in self._matrix_constraints
        )

    def _has_general_constraints(self) -> bool:
        """Check if the problem has any non-bound constraints."""
        return bool(self._constraints or self._matrix_constraints)

    def _general_constraint_rows(self) -> int:
        """Return the total number of scalar and matrix constraint rows."""
        return len(self._constraints) + sum(
            mc.A.shape[0] for mc in self._matrix_constraints
        )

    def _prefer_trust_constr_for_sparse_constraints(self) -> bool:
        """Heuristic for large sparse constrained NLPs.

        trust-constr can exploit batched sparse Jacobians and sparse linear
        constraints better than SLSQP when the constrained NLP is large and the
        Jacobian structure is sparse.
        """
        from scipy import sparse as sp

        n = len(self.variables)
        total_rows = self._general_constraint_rows()

        if n == 0 or total_rows == 0:
            return False

        if self._matrix_constraints:
            total_entries = 0
            total_nnz = 0
            sparse_blocks = 0

            for mc in self._matrix_constraints:
                m, cols = mc.A.shape
                total_entries += m * cols
                if sp.issparse(mc.A):
                    sparse_blocks += 1
                    total_nnz += mc.A.nnz
                else:
                    total_nnz += int(np.count_nonzero(mc.A))

            density = total_nnz / total_entries if total_entries > 0 else 1.0

            if sparse_blocks > 0 and (
                n >= SPARSE_NLP_VARIABLE_THRESHOLD
                or total_rows >= SPARSE_NLP_ROW_THRESHOLD
            ):
                return True

            if (
                total_entries >= SPARSE_MATRIX_ENTRY_THRESHOLD
                and density <= SPARSE_NLP_DENSITY_THRESHOLD
            ):
                return True

        if (
            len(self._constraints) >= SPARSE_NLP_ROW_THRESHOLD
            and n >= SPARSE_NLP_VARIABLE_THRESHOLD
        ):
            total_support = sum(len(c.get_variables()) for c in self._constraints)
            density = total_support / (len(self._constraints) * n)
            if density <= SPARSE_NLP_DENSITY_THRESHOLD:
                return True

        return False

    def _auto_select_method(self) -> str:
        """Automatically select the best solver method for this problem.

        Decision tree:
        1. Linear problem → "linprog" (handled separately in solve())
        2. Unconstrained:
           - n > 1000 → "L-BFGS-B" (memory efficient for large problems)
           - else → "L-BFGS-B" (fast, handles bounds, good default)
        3. Large sparse constrained NLP → "trust-constr"
        4. Non-linear + constraints → "trust-constr" (robust for non-convex)
        5. Linear/quadratic + constraints → "SLSQP" (faster, with fallback)

        Note: If SLSQP produces a solution that violates constraints, the
        solver will automatically retry with trust-constr (see solve_scipy).
        """
        from optyx.analysis import compute_degree

        # Unconstrained - use L-BFGS-B (fast, memory-efficient, handles bounds)
        if not self._has_general_constraints():
            return "L-BFGS-B"

        if self._prefer_trust_constr_for_sparse_constraints():
            return "trust-constr"

        # Only variable bounds (no general constraints)
        # FIXME: L-BFGS-B does not support constraints passed via the 'constraints' argument.
        # Until we implement merging of simple bound constraints into the variable bounds,
        # we must avoid L-BFGS-B if there are any constraints in the list.
        # if self._only_simple_bounds():
        #     return "L-BFGS-B"

        # Check if objective is non-linear (degree > 2 or contains transcendental functions)
        obj = self.objective
        if obj is not None:
            degree = compute_degree(obj)
            # degree is None for transcendental functions (exp, log, etc.)
            # degree > 2 means higher-order polynomial
            # Both cases indicate non-linear that needs robust solver
            if degree is None or degree > 2:
                # Use trust-constr for non-linear objectives - more robust for non-convex
                return "trust-constr"

        # Check if any constraint is non-linear (degree > 2 or transcendental)
        for c in self._constraints:
            c_degree = compute_degree(c.expr)
            if c_degree is None or c_degree > 2:
                # Non-linear constraint requires robust solver
                return "trust-constr"

        # General constraints with linear/quadratic objective → SLSQP (with fallback)
        return "SLSQP"

    def solve(
        self,
        method: str = "auto",
        strict: bool = False,
        warm_start: bool = True,
        callback: Callable[[SolverProgress], bool | None] | None = None,
        time_limit: float | None = None,
        **kwargs,
    ) -> Solution:
        """Solve the optimization problem.

        Args:
            method: Solver method. Options:
                - "auto" (default): Automatically select the best method:
                    - Linear problems → linprog (HiGHS)
                    - Unconstrained → L-BFGS-B
                    - Bounds only → L-BFGS-B
                    - General constraints → SLSQP
                - "linprog": Force LP solver (scipy.optimize.linprog)
                - "highs": HiGHS LP solver (auto method selection)
                - "highs-ds": HiGHS dual simplex
                - "highs-ipm": HiGHS interior point method
                - "SLSQP": Sequential Least Squares Programming
                - "trust-constr": Trust-region constrained optimization
                - "L-BFGS-B": Limited-memory BFGS with bounds
                - "BFGS": Broyden-Fletcher-Goldfarb-Shanno
                - "Nelder-Mead": Derivative-free simplex method
            strict: If True, raise ValueError when the problem contains features
                that the solver cannot handle exactly (e.g., integer/binary
                variables with SciPy). If False (default), emit a warning and
                use the best available approximation.
            warm_start: If True (default), use the previous solution as the
                initial point for re-solving. Only applies to NLP methods.
                Call reset() to clear warm start state.
            callback: Optional function called at each solver iteration with a
                SolverProgress object. Return True to terminate early
                (solution will have SolverStatus.TERMINATED). Only applies
                to NLP methods (SciPy).
            time_limit: Maximum wall-clock time in seconds. If exceeded, the
                solver terminates early with SolverStatus.TERMINATED. Only
                applies to NLP methods (SciPy).
            **kwargs: Additional arguments passed to the solver.

        Returns:
            Solution object with results.

        Raises:
            NoObjectiveError: If no objective has been set.
            UnsupportedOperationError: If the problem is a nonlinear discrete
                model (MIQP/MINLP), which the current solver stack does not
                support.
        """
        if self._objective is None:
            raise NoObjectiveError(
                suggestion="Call minimize() or maximize() on the problem first.",
            )

        # Handle automatic method selection
        if method == "auto":
            if self._is_linear_problem():
                from optyx.solvers.lp_solver import solve_lp

                solution = solve_lp(self, strict=strict, **kwargs)
                self._store_solution(solution)
                return solution
            else:
                method = self._auto_select_method()

        # Check for MIQP/MINLP (nonlinear + integer vars) before NLP dispatch.
        # This fires for all NLP methods, not just "auto", to prevent silent
        # relaxation of integer/binary domains.
        _NLP_METHODS = {
            "SLSQP",
            "trust-constr",
            "L-BFGS-B",
            "BFGS",
            "CG",
            "Newton-CG",
            "Nelder-Mead",
            "Powell",
            "COBYLA",
            "TNC",
            "dogleg",
            "trust-ncg",
            "trust-exact",
            "trust-krylov",
        }
        if method in _NLP_METHODS:
            discrete_names = [
                v.name for v in self.variables if v.domain in ("integer", "binary")
            ]
            if discrete_names and not self._is_linear_problem():
                raise UnsupportedOperationError(
                    "MIQP/MINLP solve",
                    solver_name="SciPy/HiGHS",
                    problem_feature=(
                        "nonlinear objective or constraints with integer/binary "
                        f"variables {discrete_names}"
                    ),
                    suggestion=(
                        "Use the MILP solver for linear discrete models, or relax "
                        "integrality / switch to a dedicated MIQP or MINLP solver"
                    ),
                )

        # Handle explicit milp request
        if method == "milp":
            from optyx.solvers.lp_solver import solve_lp

            solution = solve_lp(self, strict=strict, **kwargs)
            self._store_solution(solution)
            return solution

        # Handle explicit linprog request
        if method == "linprog":
            from optyx.solvers.lp_solver import solve_lp

            solution = solve_lp(self, strict=strict, **kwargs)
            self._store_solution(solution)
            return solution

        # Route HiGHS methods to LP solver
        if method in ("highs", "highs-ds", "highs-ipm"):
            from optyx.solvers.lp_solver import solve_lp

            solution = solve_lp(self, method=method, strict=strict, **kwargs)
            self._store_solution(solution)
            return solution

        # Use scipy solver for NLP methods
        from optyx.solvers.scipy_solver import solve_scipy

        solution = solve_scipy(
            self,
            method=method,
            strict=strict,
            warm_start=warm_start,
            callback=callback,
            time_limit=time_limit,
            **kwargs,
        )
        self._store_solution(solution)
        return solution

    def _store_solution(self, solution: Solution) -> None:
        """Store solution values for warm starting subsequent solves."""
        if solution.values:
            variables = self.variables
            x = np.array([solution.values.get(v.name, 0.0) for v in variables])
            self._last_solution = x

    def __repr__(self) -> str:
        obj_str = "not set" if self._objective is None else f"{self._sense}"
        return (
            f"Problem(name={self.name!r}, "
            f"objective={obj_str}, "
            f"n_vars={self.n_variables}, "
            f"n_constraints={self.n_constraints})"
        )

    def write(self, filename: str) -> None:
        """Export the problem to LP file format.

        Writes the problem formulation to a human-readable .lp file,
        compatible with solvers like CPLEX, Gurobi, GLPK, and HiGHS.

        Supports linear and quadratic objectives, linear constraints,
        variable bounds, and integer/binary variable types.

        Args:
            filename: Path to the output .lp file.

        Raises:
            InvalidOperationError: If the problem contains nonlinear
                expressions that cannot be represented in LP format.
            NoObjectiveError: If no objective has been set.

        Example:
            >>> x = Variable("x", lb=0)
            >>> y = Variable("y", lb=0)
            >>> prob = Problem("example")
            >>> prob.minimize(2 * x + 3 * y)
            >>> prob.subject_to(x + y >= 1)
            >>> prob.write("example.lp")
        """
        from optyx.io import write_lp

        write_lp(self, filename)

    def to_lp(self) -> str:
        """Return the LP format string representation of the problem.

        Like write(), but returns the string instead of writing to a file.

        Returns:
            The LP format string.

        Raises:
            InvalidOperationError: If the problem contains nonlinear expressions.
        """
        from optyx.io import format_lp

        return format_lp(self)

    def summary(self) -> str:
        """Return a human-readable summary of the optimization problem.

        Provides an overview including problem name, variable counts
        (with breakdown by type), constraint counts, and objective sense.

        Returns:
            Multi-line string describing the problem structure.

        Example:
            >>> x = VectorVariable("x", 100, lb=0)
            >>> prob = Problem("portfolio")
            >>> prob.minimize(x.dot(x))
            >>> prob.subject_to(x.sum() == 1)
            >>> print(prob.summary())
            Optyx Problem: portfolio
              Variables: 100
              Constraints: 1 (0 equality, 1 inequality)
              Objective: minimize
        """
        # Count constraints by type
        n_eq = sum(1 for c in self._constraints if c.sense == "==")
        n_ineq = len(self._constraints) - n_eq

        # Build summary lines
        name_str = self.name or "Unnamed"
        lines = [
            f"Optyx Problem: {name_str}",
            f"  Variables: {self.n_variables}",
            f"  Constraints: {self.n_constraints} ({n_eq} equality, {n_ineq} inequality)",
        ]

        if self._objective is not None:
            lines.append(f"  Objective: {self._sense}")
        else:
            lines.append("  Objective: not set")

        return "\n".join(lines)
