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
    from optyx.constraints import Constraint
    from optyx.core.expressions import Expression, Variable
    from optyx.solution import Solution


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

    def minimize(self, expr: Expression) -> Problem:
        """Set the objective function to minimize.

        Args:
            expr: Expression to minimize.

        Returns:
            Self for method chaining.
        """
        self._objective = expr
        self._sense = "minimize"
        self._variables = None  # Invalidate cache
        self._solver_cache = None  # Invalidate solver cache
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
        self._variables = None  # Invalidate cache
        self._solver_cache = None  # Invalidate solver cache
        return self

    def subject_to(self, constraint: Constraint) -> Problem:
        """Add a constraint to the problem.

        Args:
            constraint: Constraint to add.

        Returns:
            Self for method chaining.
        """
        self._constraints.append(constraint)
        self._variables = None  # Invalidate cache
        self._solver_cache = None  # Invalidate solver cache
        return self
        self._variables = None  # Invalidate cache
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

    def solve(
        self,
        method: str = "SLSQP",
        strict: bool = False,
        **kwargs,
    ) -> Solution:
        """Solve the optimization problem.

        Args:
            method: Solver method. Options: "SLSQP", "trust-constr", "L-BFGS-B".
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
