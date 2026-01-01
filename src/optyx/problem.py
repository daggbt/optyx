"""Problem class for defining optimization problems.

Provides a fluent API for building optimization problems:

    prob = Problem()
    prob.minimize(x**2 + y**2)
    prob.subject_to(x + y >= 1)
    solution = prob.solve()
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from optyx.core.errors import InvalidOperationError, ConstraintError

if TYPE_CHECKING:
    from optyx.analysis import LPData
    from optyx.constraints import Constraint
    from optyx.core.expressions import Expression, Variable
    from optyx.solution import Solution


# Threshold for "small" problems where gradient-free methods are faster
SMALL_PROBLEM_THRESHOLD = 3

# Threshold for "large" problems where memory-efficient methods are preferred
LARGE_PROBLEM_THRESHOLD = 1000


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
    name = var.name
    # Split into text and number parts
    parts = re.split(r"(\d+)", name)
    # Convert number parts to integers for proper numeric sorting
    return tuple(int(p) if p.isdigit() else p for p in parts)


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
        self._variables: list[Variable] | None = None  # Cached
        # Solver cache for compiled callables (reused across solve() calls)
        self._solver_cache: dict | None = None
        # LP data cache (reused across solve() calls for LP problems)
        self._lp_cache: LPData | None = None
        # Cached linearity check result (None = not computed, True/False = result)
        self._is_linear_cache: bool | None = None

    def _invalidate_caches(self) -> None:
        """Invalidate all cached data when problem is modified."""
        self._variables = None
        self._solver_cache = None
        self._lp_cache = None
        self._is_linear_cache = None

    def minimize(self, expr: Expression) -> Problem:
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

    def maximize(self, expr: Expression) -> Problem:
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

    def subject_to(self, constraint: Constraint | list[Constraint]) -> Problem:
        """Add a constraint or list of constraints to the problem.

        Args:
            constraint: Constraint or list of constraints to add.
                Lists are typically produced by vectorized constraints
                like `x >= 0` on VectorVariable.

        Returns:
            Self for method chaining.

        Raises:
            ConstraintError: If constraint is not a valid Constraint type.

        Example:
            >>> x = VectorVariable("x", 100)
            >>> prob.subject_to(x >= 0)  # Adds 100 constraints
        """
        if isinstance(constraint, list):
            for c in constraint:
                self._constraints.append(self._validate_constraint(c))
        else:
            self._constraints.append(self._validate_constraint(constraint))
        self._invalidate_caches()
        return self

    def _validate_expression(self, expr: Expression, context: str) -> Expression:
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

        all_vars: set[Variable] = set()

        if self._objective is not None:
            all_vars.update(get_all_variables(self._objective))

        for constraint in self._constraints:
            all_vars.update(constraint.get_variables())

        self._variables = sorted(all_vars, key=_natural_sort_key)
        return self._variables

    @property
    def n_variables(self) -> int:
        """Number of decision variables."""
        return len(self.variables)

    @property
    def n_constraints(self) -> int:
        """Number of constraints."""
        return len(self._constraints)

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

        self._is_linear_cache = True
        return True

    def _only_simple_bounds(self) -> bool:
        """Check if all constraints are simple variable bounds.

        Simple bounds are constraints on a single variable like x >= 0 or x <= 10.
        """
        if not self._constraints:
            return True

        from optyx.analysis import is_simple_bound

        return all(is_simple_bound(c, self.variables) for c in self._constraints)

    def _has_equality_constraints(self) -> bool:
        """Check if problem has any equality constraints."""
        return any(c.sense == "==" for c in self._constraints)

    def _auto_select_method(self) -> str:
        """Automatically select the best solver method for this problem.

        Decision tree:
        1. Linear problem → "linprog" (handled separately in solve())
        2. Unconstrained:
           - n > 1000 → "L-BFGS-B" (memory efficient for large problems)
           - else → "L-BFGS-B" (fast, handles bounds, good default)
        3. Only simple bounds → "L-BFGS-B"
        4. Non-linear + constraints → "trust-constr" (robust for non-convex)
        5. Linear/quadratic + constraints → "SLSQP" (faster, with fallback)

        Note: If SLSQP produces a solution that violates constraints, the
        solver will automatically retry with trust-constr (see solve_scipy).
        """
        from optyx.analysis import compute_degree

        # Unconstrained - use L-BFGS-B (fast, memory-efficient, handles bounds)
        if not self._constraints:
            return "L-BFGS-B"

        # Only variable bounds (no general constraints)
        if self._only_simple_bounds():
            return "L-BFGS-B"

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
                - "SLSQP": Sequential Least Squares Programming
                - "trust-constr": Trust-region constrained optimization
                - "L-BFGS-B": Limited-memory BFGS with bounds
                - "BFGS": Broyden-Fletcher-Goldfarb-Shanno
                - "Nelder-Mead": Derivative-free simplex method
            strict: If True, raise ValueError when the problem contains features
                that the solver cannot handle exactly (e.g., integer/binary
                variables with SciPy). If False (default), emit a warning and
                use the best available approximation.
            **kwargs: Additional arguments passed to the solver.

        Returns:
            Solution object with results.

        Raises:
            ValueError: If no objective has been set, or if strict=True and
                the problem contains unsupported features.
        """
        if self._objective is None:
            raise ValueError("No objective set. Call minimize() or maximize() first.")

        # Handle automatic method selection
        if method == "auto":
            if self._is_linear_problem():
                from optyx.solvers.lp_solver import solve_lp

                return solve_lp(self, strict=strict, **kwargs)
            else:
                method = self._auto_select_method()

        # Handle explicit linprog request
        if method == "linprog":
            from optyx.solvers.lp_solver import solve_lp

            return solve_lp(self, strict=strict, **kwargs)

        # Use scipy solver for NLP methods
        from optyx.solvers.scipy_solver import solve_scipy

        return solve_scipy(self, method=method, strict=strict, **kwargs)

    def __repr__(self) -> str:
        obj_str = "not set" if self._objective is None else f"{self._sense}"
        return (
            f"Problem(name={self.name!r}, "
            f"objective={obj_str}, "
            f"n_vars={self.n_variables}, "
            f"n_constraints={self.n_constraints})"
        )

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
