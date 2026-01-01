"""Solution classes for optimization results.

Provides structured representation of solver output including
status, objective value, variable values, and solver statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from optyx.core.expressions import Variable
    from optyx.core.vectors import VectorVariable
    from optyx.core.matrices import MatrixVariable
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

    Example:
        >>> solution = problem.solve()
        >>> if solution.is_optimal:
        ...     print(f"Optimal value: {solution.objective_value}")
        ...     print(f"x = {solution['x']}")
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

    def __getitem__(
        self, var: Variable | VectorVariable | MatrixVariable | str
    ) -> float | NDArray[np.floating]:
        """Get the optimal value of a variable.

        For scalar Variable: returns float.
        For VectorVariable: returns 1D numpy array.
        For MatrixVariable: returns 2D numpy array.

        Args:
            var: Variable, VectorVariable, MatrixVariable, or variable name.

        Returns:
            The optimal value(s).

        Raises:
            KeyError: If variable not found in solution.

        Example:
            >>> x = Variable("x")
            >>> v = VectorVariable("v", 3)
            >>> A = MatrixVariable("A", 2, 2)
            >>> solution[x]  # float
            >>> solution[v]  # np.array([...]) shape (3,)
            >>> solution[A]  # np.array([[...]]) shape (2, 2)
        """
        # Import here to avoid circular imports
        from optyx.core.vectors import VectorVariable
        from optyx.core.matrices import MatrixVariable

        if isinstance(var, VectorVariable):
            return self._get_vector(var)
        elif isinstance(var, MatrixVariable):
            return self._get_matrix(var)
        elif isinstance(var, Variable):
            return self.values[var.name]
        else:
            # String name - return scalar
            return self.values[var]

    def _get_vector(self, vec: VectorVariable) -> NDArray[np.floating]:
        """Extract VectorVariable values as 1D numpy array.

        Args:
            vec: VectorVariable to extract.

        Returns:
            1D numpy array of values.

        Raises:
            KeyError: If any variable not found in solution.
        """
        result = np.zeros(vec.size)
        for i, v in enumerate(vec._variables):
            result[i] = self.values[v.name]
        return result

    def _get_matrix(self, mat: MatrixVariable) -> NDArray[np.floating]:
        """Extract MatrixVariable values as 2D numpy array.

        Values are arranged in row-major order matching the matrix structure.

        Args:
            mat: MatrixVariable to extract.

        Returns:
            2D numpy array of values.

        Raises:
            KeyError: If any variable not found in solution.
        """
        result = np.zeros((mat.rows, mat.cols))
        for i in range(mat.rows):
            for j in range(mat.cols):
                result[i, j] = self.values[mat[i, j].name]
        return result

    def get(
        self,
        var: Variable | VectorVariable | MatrixVariable | str,
        default: float | NDArray[np.floating] | None = None,
    ) -> float | NDArray[np.floating] | None:
        """Get the optimal value of a variable with a default.

        For scalar Variable: returns float.
        For VectorVariable: returns 1D numpy array.
        For MatrixVariable: returns 2D numpy array.

        Args:
            var: Variable, VectorVariable, MatrixVariable, or variable name.
            default: Value to return if variable not found.

        Returns:
            The optimal value(s) or default.
        """

        try:
            return self[var]
        except KeyError:
            return default

    def __repr__(self) -> str:
        if self.is_optimal:
            return (
                f"Solution(status={self.status.value}, "
                f"objective={self.objective_value:.6g}, "
                f"values={self.values})"
            )
        return f"Solution(status={self.status.value}, message='{self.message}')"
