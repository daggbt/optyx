"""Problem class for defining optimization problems.

Provides a fluent API for building optimization problems:

    prob = Problem()
    prob.minimize(x**2 + y**2)
    prob.subject_to(x + y >= 1)
    solution = prob.solve()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from optyx.analysis import LPData
    from optyx.constraints import Constraint
    from optyx.core.expressions import Expression, Variable
    from optyx.solution import Solution


# Threshold for "small" problems where gradient-free methods are faster
SMALL_PROBLEM_THRESHOLD = 3

# Threshold for "large" problems where memory-efficient methods are preferred
LARGE_PROBLEM_THRESHOLD = 1000


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
            expr: Expression to minimize.

        Returns:
            Self for method chaining.
        """
        self._objective = expr
        self._sense = "minimize"
        self._invalidate_caches()
        return self

    def maximize(self, expr: Expression) -> Problem:
        """Set the objective function to maximize.

        Args:
            expr: Expression to maximize.

        Returns:
            Self for method chaining.
        """
        self._objective = expr
        self._sense = "maximize"
        self._invalidate_caches()
        return self

    def subject_to(self, constraint: Constraint) -> Problem:
        """Add a constraint to the problem.

        Args:
            constraint: Constraint to add.

        Returns:
            Self for method chaining.
        """
        self._constraints.append(constraint)
        self._invalidate_caches()
        return self

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
        Sorted by name for consistent ordering.
        """
        if self._variables is not None:
            return self._variables

        all_vars: set[Variable] = set()

        if self._objective is not None:
            all_vars.update(self._objective.get_variables())

        for constraint in self._constraints:
            all_vars.update(constraint.get_variables())

        self._variables = sorted(all_vars, key=lambda v: v.name)
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
           - n ≤ 3 → "Nelder-Mead" (no gradient overhead)
           - n > 1000 → "L-BFGS-B" (memory efficient)
           - else → "BFGS" (fast with gradients)
        3. Only simple bounds → "L-BFGS-B"
        4. Has equality constraints → "trust-constr"
        5. Inequality only → "SLSQP"
        """
        n = len(self.variables)

        # Unconstrained
        if not self._constraints:
            if n <= SMALL_PROBLEM_THRESHOLD:
                return "Nelder-Mead"
            elif n > LARGE_PROBLEM_THRESHOLD:
                return "L-BFGS-B"
            else:
                return "BFGS"

        # Only variable bounds (no general constraints)
        if self._only_simple_bounds():
            return "L-BFGS-B"

        # Has equality constraints → use trust-constr (most robust)
        if self._has_equality_constraints():
            return "trust-constr"

        # Inequality-only constraints → use SLSQP (fast)
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
                    - Unconstrained, n ≤ 3 → Nelder-Mead
                    - Unconstrained, n > 1000 → L-BFGS-B
                    - Unconstrained, else → BFGS
                    - Bounds only → L-BFGS-B
                    - Equality constraints → trust-constr
                    - Inequality only → SLSQP
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
