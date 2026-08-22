"""Solution classes for optimization results.

Provides structured representation of solver output including
status, objective value, variable values, and solver statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable
import json
import os

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from optyx.core.expressions import Variable
    from optyx.core.vectors import VectorVariable
    from optyx.core.matrices import MatrixVariable
    from optyx.core.variable_dict import VariableDict
else:
    from optyx.core.expressions import Variable


class SolverStatus(Enum):
    """Status of an optimization solve."""

    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    MAX_ITERATIONS = "max_iterations"
    TERMINATED = "terminated"
    FAILED = "failed"
    NOT_SOLVED = "not_solved"


class LazyValuesDict(dict[str, float]):
    """Dictionary that materializes solution values on first access."""

    __slots__ = ("_loader", "_loaded")

    def __init__(
        self,
        initial: dict[str, float] | None = None,
        loader: Callable[[], dict[str, float]] | None = None,
    ) -> None:
        super().__init__(initial or {})
        self._loader = loader
        self._loaded = initial is not None or loader is None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        assert self._loader is not None
        super().update(self._loader())
        self._loader = None
        self._loaded = True

    def __getitem__(self, key: str) -> float:
        self._ensure_loaded()
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        self._ensure_loaded()
        return super().__contains__(key)

    def __iter__(self):
        self._ensure_loaded()
        return super().__iter__()

    def __len__(self) -> int:
        self._ensure_loaded()
        return super().__len__()

    def __bool__(self) -> bool:
        self._ensure_loaded()
        return super().__len__() > 0

    def __repr__(self) -> str:
        self._ensure_loaded()
        return super().__repr__()

    def __eq__(self, other: object) -> bool:
        self._ensure_loaded()
        return super().__eq__(other)

    def get(self, key: str, default: float | None = None) -> float | None:
        self._ensure_loaded()
        return super().get(key, default)

    def items(self):
        self._ensure_loaded()
        return super().items()

    def keys(self):
        self._ensure_loaded()
        return super().keys()

    def values(self):
        self._ensure_loaded()
        return super().values()

    def copy(self) -> dict[str, float]:
        self._ensure_loaded()
        return dict(self)


@dataclass
class SolverProgress:
    """Snapshot of solver state passed to user callbacks during optimization.

    Attributes:
        iteration: Current iteration number.
        objective_value: Current objective function value (in original sense).
        constraint_violation: Maximum constraint violation (0.0 if feasible).
        elapsed_time: Wall-clock time since solve started (seconds).
        x: Current variable values as a numpy array.
    """

    iteration: int
    objective_value: float
    constraint_violation: float
    elapsed_time: float
    x: NDArray[np.floating]


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
        constraint_violation: Maximum post-solve feasibility violation. ``None``
            means feasibility was not checked.
        feasibility_tolerance: Base absolute tolerance used for the feasibility
            check. ``None`` means feasibility was not checked.

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
    mip_gap: float | None = None
    best_bound: float | None = None
    constraint_violation: float | None = None
    feasibility_tolerance: float | None = None
    _raw_x: NDArray[np.floating] | None = field(default=None, repr=False, compare=False)
    _raw_layout_signature: tuple[Any, ...] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def is_optimal(self) -> bool:
        """Check if the solution is optimal."""
        return self.status == SolverStatus.OPTIMAL

    @property
    def is_feasible(self) -> bool:
        """Return whether post-solve validation confirmed feasibility."""
        return (
            self.feasibility_checked
            and self.constraint_violation is not None
            and self.feasibility_tolerance is not None
            and 0.0 <= self.constraint_violation <= self.feasibility_tolerance
        )

    @property
    def feasibility_checked(self) -> bool:
        """Return whether explicit post-solve feasibility evidence is available."""
        return (
            self.constraint_violation is not None
            and self.feasibility_tolerance is not None
        )

    def to_dict(self) -> dict:
        """Convert solution to dictionary."""
        return {
            "status": self.status.value,
            "objective_value": self.objective_value,
            "values": dict(self.values),
            "multipliers": self.multipliers,
            "iterations": self.iterations,
            "message": self.message,
            "solve_time": self.solve_time,
            "mip_gap": self.mip_gap,
            "best_bound": self.best_bound,
            "constraint_violation": self.constraint_violation,
            "feasibility_tolerance": self.feasibility_tolerance,
        }

    def to_json(self, path: str | None = None) -> str:
        """Convert solution to JSON string or save to file.

        Args:
            path: Optional file path to save JSON to.

        Returns:
            JSON string if path is None, otherwise empty string.
        """
        data = self.to_dict()
        if path:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
            return ""
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str_or_path: str) -> Solution:
        """Create solution from JSON string or file path.

        Args:
            json_str_or_path: JSON string or path to JSON file.

        Returns:
            Solution object.
        """
        if os.path.isfile(json_str_or_path):
            with open(json_str_or_path, "r") as f:
                data = json.load(f)
        else:
            data = json.loads(json_str_or_path)

        return cls(
            status=SolverStatus(data["status"]),
            objective_value=data.get("objective_value"),
            values=data.get("values", {}),
            multipliers=data.get("multipliers"),
            iterations=data.get("iterations"),
            message=data.get("message", ""),
            solve_time=data.get("solve_time"),
            mip_gap=data.get("mip_gap"),
            best_bound=data.get("best_bound"),
            constraint_violation=data.get("constraint_violation"),
            feasibility_tolerance=data.get("feasibility_tolerance"),
        )

    def print_vars(self) -> None:
        """Pretty-print variable values."""
        print(f"Status: {self.status.value}")
        if self.objective_value is not None:
            print(f"Objective: {self.objective_value:.6g}")
        print("Variables:")
        for name, value in sorted(self.values.items()):
            print(f"  {name}: {value:.6g}")

    def __getitem__(
        self, var: Variable | VectorVariable | MatrixVariable | str
    ) -> float | NDArray[np.floating] | dict[str, float]:
        """Get the optimal value of a variable.

        For scalar Variable: returns float.
        For VectorVariable: returns 1D numpy array.
        For MatrixVariable: returns 2D numpy array.
        For VariableDict: returns dict mapping keys to float values.

        Args:
            var: Variable, VectorVariable, MatrixVariable, VariableDict,
                or variable name.

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
        from optyx.core.variable_dict import VariableDict

        if isinstance(var, VariableDict):
            return self._get_variable_dict(var)
        elif isinstance(var, VectorVariable):
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

    def _get_variable_dict(self, vd: VariableDict) -> dict[str, float]:
        """Extract VariableDict values as a dict mapping keys to floats.

        Args:
            vd: VariableDict to extract.

        Returns:
            Dict mapping each key to its optimal value.

        Raises:
            KeyError: If any variable not found in solution.
        """
        return {key: self.values[var.name] for key, var in vd.items()}

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
        default: float | NDArray[np.floating] | dict[str, float] | None = None,
    ) -> float | NDArray[np.floating] | dict[str, float] | None:
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
