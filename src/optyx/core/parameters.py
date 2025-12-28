"""Parameter class for fast re-solves.

Parameters are constants that can change between solves without rebuilding
the problem structure. This enables fast re-optimization for scenarios like:
- Sensitivity analysis
- Rolling horizon optimization
- What-if scenarios
- Real-time optimization with changing inputs
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

import numpy as np

from optyx.core.expressions import Expression, Variable

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray


class Parameter(Expression):
    """An updatable constant for optimization problems.

    Unlike Constant, a Parameter's value can be changed between solves
    without rebuilding the problem structure. This enables fast re-solves
    when only numerical values change (not problem structure).

    Parameters participate in expression building just like variables,
    but their values are fixed during each solve. Changing a parameter
    value and re-solving uses cached problem structure.

    Args:
        name: Unique identifier for this parameter.
        value: Initial value (scalar or array).

    Example:
        >>> from optyx import Variable, Parameter, Problem
        >>>
        >>> x = Variable("x", lb=0)
        >>> price = Parameter("price", value=100)
        >>>
        >>> prob = Problem().maximize(price * x - x**2).subject_to(x <= 10)
        >>>
        >>> # Initial solve
        >>> sol1 = prob.solve()
        >>>
        >>> # Price changes - fast re-solve
        >>> price.set(120)
        >>> sol2 = prob.solve()  # Uses cached structure
    """

    __slots__ = ("name", "_value")

    def __init__(self, name: str, value: float | int | ArrayLike = 0.0) -> None:
        """Create a new parameter.

        Args:
            name: Unique identifier for this parameter.
            value: Initial value (default: 0.0).
        """
        self.name = name
        self._value: float | NDArray[np.floating] = (
            np.asarray(value) if not isinstance(value, (int, float)) else float(value)
        )

    @property
    def value(self) -> float | NDArray[np.floating]:
        """Get the current parameter value."""
        return self._value

    def set(self, value: float | int | ArrayLike) -> None:
        """Update the parameter value.

        This can be called between solves to change the parameter value
        without rebuilding the problem structure.

        Args:
            value: New value (scalar or array, must match original shape).

        Raises:
            ValueError: If array shape doesn't match original.

        Example:
            >>> price = Parameter("price", value=100)
            >>> price.set(120)  # Update for next solve
            >>> price.value
            120.0
        """
        new_value: float | NDArray[np.floating] = (
            np.asarray(value) if not isinstance(value, (int, float)) else float(value)
        )

        # Check shape compatibility for arrays
        if isinstance(self._value, np.ndarray) and isinstance(new_value, np.ndarray):
            if self._value.shape != new_value.shape:
                raise ValueError(
                    f"Shape mismatch: expected {self._value.shape}, "
                    f"got {new_value.shape}"
                )
        elif isinstance(self._value, np.ndarray) != isinstance(new_value, np.ndarray):
            # One is array, one is scalar - could be ok for 0-d arrays
            if isinstance(new_value, np.ndarray) and new_value.ndim > 0:
                raise ValueError(
                    f"Cannot change scalar parameter to array of shape {new_value.shape}"
                )

        self._value = new_value

    def evaluate(
        self, values: Mapping[str, ArrayLike | float]
    ) -> NDArray[np.floating] | float:
        """Evaluate the parameter (returns current value).

        Parameters evaluate to their stored value, not from the values dict.
        This allows parameters to be updated independently of solve calls.

        Args:
            values: Variable values (not used for parameters).

        Returns:
            The current parameter value.
        """
        return self._value

    def get_variables(self) -> set[Variable]:
        """Return all variables this expression depends on.

        Parameters are not variables - they return an empty set.
        """
        return set()

    def __hash__(self) -> int:
        return hash(("Parameter", self.name))

    def __eq__(self, other: object) -> bool:
        """Check equality by name."""
        if isinstance(other, Parameter):
            return self.name == other.name
        return False

    def __repr__(self) -> str:
        return f"Parameter('{self.name}', value={self._value})"


class VectorParameter:
    """A vector of parameters for array-valued constants.

    VectorParameter creates a collection of scalar Parameter instances
    that can be updated together, useful for time-varying vectors like
    demand forecasts or price curves.

    Args:
        name: Base name for the parameters. Elements are named "{name}[i]".
        size: Number of elements.
        values: Initial values (array-like or scalar for all elements).

    Example:
        >>> from optyx import VectorVariable, VectorParameter, dot
        >>>
        >>> # Time-varying prices
        >>> prices = VectorParameter("price", 24, values=[100]*24)
        >>> quantities = VectorVariable("q", 24, lb=0)
        >>>
        >>> revenue = dot(prices, quantities)
        >>>
        >>> # Update prices for next solve
        >>> prices.set([105, 110, 115, ...])  # New price forecast
    """

    __slots__ = ("name", "size", "_parameters")

    def __init__(
        self,
        name: str,
        size: int,
        values: ArrayLike | float | None = None,
    ) -> None:
        """Create a vector of parameters.

        Args:
            name: Base name for the parameters.
            size: Number of elements.
            values: Initial values (array or scalar, default: 0.0).
        """
        if size <= 0:
            raise ValueError(f"Size must be positive, got {size}")

        self.name = name
        self.size = size

        # Convert values to array
        if values is None:
            val_array = np.zeros(size)
        elif isinstance(values, (int, float)):
            val_array = np.full(size, values)
        else:
            val_array = np.asarray(values)
            if val_array.shape != (size,):
                raise ValueError(
                    f"Values shape {val_array.shape} doesn't match size ({size},)"
                )

        # Create individual parameters
        self._parameters: list[Parameter] = [
            Parameter(f"{name}[{i}]", val_array[i]) for i in range(size)
        ]

    def __getitem__(self, idx: int) -> Parameter:
        """Get a single parameter by index."""
        if idx < 0:
            idx = self.size + idx
        if idx < 0 or idx >= self.size:
            raise IndexError(f"Index {idx} out of range for size {self.size}")
        return self._parameters[idx]

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        return iter(self._parameters)

    def set(self, values: ArrayLike) -> None:
        """Update all parameter values.

        Args:
            values: New values (array-like of length size).

        Raises:
            ValueError: If values length doesn't match size.
        """
        val_array = np.asarray(values)
        if val_array.shape != (self.size,):
            raise ValueError(
                f"Values shape {val_array.shape} doesn't match size ({self.size},)"
            )

        for i, param in enumerate(self._parameters):
            param.set(val_array[i])

    def get_values(self) -> NDArray[np.floating]:
        """Get all current parameter values as array."""
        return np.array([p.value for p in self._parameters])

    def to_numpy(self) -> NDArray[np.floating]:
        """Get all current parameter values as numpy array.

        This is an alias for get_values() for consistency with
        VectorVariable.to_numpy().
        """
        return self.get_values()

    def __repr__(self) -> str:
        return f"VectorParameter('{self.name}', {self.size})"
