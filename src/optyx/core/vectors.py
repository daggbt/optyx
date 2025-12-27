"""Vector variables for optimization problems.

This module provides VectorVariable for representing vectors of decision variables,
enabling natural syntax like `x = VectorVariable("x", 100)` with indexing and slicing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, Literal, overload

from optyx.core.expressions import Variable

if TYPE_CHECKING:
    pass

# Type alias for variable domain
DomainType = Literal["continuous", "integer", "binary"]


class VectorVariable:
    """A vector of optimization variables.

    VectorVariable creates and manages a collection of scalar Variable instances,
    providing natural indexing, slicing, and iteration.

    Args:
        name: Base name for the vector. Elements are named "{name}[0]", "{name}[1]", etc.
        size: Number of elements in the vector.
        lb: Lower bound applied to all elements (None for unbounded).
        ub: Upper bound applied to all elements (None for unbounded).
        domain: Variable type for all elements - 'continuous', 'integer', or 'binary'.

    Example:
        >>> x = VectorVariable("x", 5, lb=0)
        >>> x[0]  # Variable named "x[0]" with lb=0
        >>> x[1:3]  # VectorVariable with elements x[1], x[2]
        >>> len(x)  # 5
        >>> for v in x: print(v.name)  # x[0], x[1], ..., x[4]
    """

    __slots__ = ("name", "size", "lb", "ub", "domain", "_variables")

    # Declare types for slots (helps type checkers)
    name: str
    size: int
    lb: float | None
    ub: float | None
    domain: DomainType
    _variables: list[Variable]

    def __init__(
        self,
        name: str,
        size: int,
        lb: float | None = None,
        ub: float | None = None,
        domain: DomainType = "continuous",
    ) -> None:
        if size <= 0:
            raise ValueError(f"Size must be positive, got {size}")

        self.name = name
        self.size = size
        self.lb = lb
        self.ub = ub
        self.domain = domain

        # Create individual variables
        self._variables: list[Variable] = [
            Variable(f"{name}[{i}]", lb=lb, ub=ub, domain=domain) for i in range(size)
        ]

    @overload
    def __getitem__(self, key: int) -> Variable: ...

    @overload
    def __getitem__(self, key: slice) -> VectorVariable: ...

    def __getitem__(self, key: int | slice) -> Variable | VectorVariable:
        """Index or slice the vector.

        Args:
            key: Integer index or slice object.

        Returns:
            Single Variable for integer index, VectorVariable for slice.

        Example:
            >>> x = VectorVariable("x", 10)
            >>> x[0]  # Variable("x[0]")
            >>> x[-1]  # Variable("x[9]")
            >>> x[2:5]  # VectorVariable with 3 elements
        """
        if isinstance(key, int):
            # Handle negative indices
            if key < 0:
                key = self.size + key
            if key < 0 or key >= self.size:
                raise IndexError(
                    f"Index {key} out of range for VectorVariable of size {self.size}"
                )
            return self._variables[key]

        elif isinstance(key, slice):
            # Get the sliced variables
            sliced_vars = self._variables[key]
            if len(sliced_vars) == 0:
                raise IndexError("Slice results in empty VectorVariable")

            # Create a new VectorVariable from the slice
            return VectorVariable._from_variables(
                name=f"{self.name}[{key.start or 0}:{key.stop or self.size}]",
                variables=sliced_vars,
                lb=self.lb,
                ub=self.ub,
                domain=self.domain,
            )

        else:
            raise TypeError(
                f"Indices must be integers or slices, not {type(key).__name__}"
            )

    @classmethod
    def _from_variables(
        cls,
        name: str,
        variables: list[Variable],
        lb: float | None = None,
        ub: float | None = None,
        domain: DomainType = "continuous",
    ) -> VectorVariable:
        """Create a VectorVariable from existing Variable instances.

        This is an internal constructor used for slicing.
        """
        # Create instance without calling __init__
        instance = object.__new__(cls)
        instance.name = name
        instance.size = len(variables)
        instance.lb = lb
        instance.ub = ub
        instance.domain = domain
        instance._variables = list(variables)  # Copy the list
        return instance

    def __len__(self) -> int:
        """Return the number of elements in the vector."""
        return self.size

    def __iter__(self) -> Iterator[Variable]:
        """Iterate over all variables in the vector."""
        return iter(self._variables)

    def get_variables(self) -> list[Variable]:
        """Return all variables in this vector.

        Returns:
            List of Variable instances in order.
        """
        return list(self._variables)

    def __repr__(self) -> str:
        bounds = ""
        if self.lb is not None or self.ub is not None:
            bounds = f", lb={self.lb}, ub={self.ub}"
        domain_str = "" if self.domain == "continuous" else f", domain='{self.domain}'"
        return f"VectorVariable('{self.name}', {self.size}{bounds}{domain_str})"
