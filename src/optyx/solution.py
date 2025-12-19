"""Solution classes for optimization results.

Provides structured representation of solver output including
status, objective value, variable values, and solver statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from optyx.core.expressions import Variable
else:
    from optyx.core.expressions import Variable


class SolverStatus(Enum):
    """Status of an optimization solve."""

    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    MAX_ITERATIONS = "max_iterations"
    FAILED = "failed"
    NOT_SOLVED = "not_solved"


@dataclass
class Solution:
    """Result of solving an optimization problem.

    Attributes:
        status: Solver termination status.
        objective_value: Optimal objective function value (None if not solved).
        values: Dictionary mapping variable names to optimal values.
        multipliers: Lagrange multipliers for constraints (if available).
        iterations: Number of solver iterations.
        message: Solver message or error description.
        solve_time: Time taken to solve (seconds).
    """

    status: SolverStatus
    objective_value: float | None = None
    values: dict[str, float] = field(default_factory=dict)
    multipliers: dict[str, float] | None = None
    iterations: int | None = None
    message: str = ""
    solve_time: float | None = None

    @property
    def is_optimal(self) -> bool:
        """Check if the solution is optimal."""
        return self.status == SolverStatus.OPTIMAL

    @property
    def is_feasible(self) -> bool:
        """Check if a feasible solution was found."""
        return self.status in (SolverStatus.OPTIMAL, SolverStatus.MAX_ITERATIONS)

    def __getitem__(self, var: Variable | str) -> float:
        """Get the optimal value of a variable.

        Args:
            var: Variable object or variable name.

        Returns:
            The optimal value.

        Raises:
            KeyError: If variable not found in solution.
        """
        if isinstance(var, Variable):
            name = var.name
        else:
            name = var
        return self.values[name]

    def get(self, var: Variable | str, default: float | None = None) -> float | None:
        """Get the optimal value of a variable with a default.

        Args:
            var: Variable object or variable name.
            default: Value to return if variable not found.

        Returns:
            The optimal value or default.
        """
        if isinstance(var, Variable):
            name = var.name
        else:
            name = var
        return self.values.get(name, default)

    def __repr__(self) -> str:
        if self.is_optimal:
            return (
                f"Solution(status={self.status.value}, "
                f"objective={self.objective_value:.6g}, "
                f"values={self.values})"
            )
        return f"Solution(status={self.status.value}, message='{self.message}')"
