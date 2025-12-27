"""Vector variables for optimization problems.

This module provides VectorVariable for representing vectors of decision variables,
enabling natural syntax like `x = VectorVariable("x", 100)` with indexing and slicing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Iterator, Literal, Mapping, overload

import numpy as np

from optyx.core.expressions import (
    Expression,
    Variable,
    Constant,
    BinaryOp,
    _ensure_expr,
)

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

# Type alias for variable domain
DomainType = Literal["continuous", "integer", "binary"]


class VectorSum(Expression):
    """Sum of all elements in a vector: sum(x) = x[0] + x[1] + ... + x[n-1].

    This is a scalar expression representing the sum of vector elements.

    Args:
        vector: The VectorVariable to sum.

    Example:
        >>> x = VectorVariable("x", 3)
        >>> s = VectorSum(x)
        >>> s.evaluate({"x[0]": 1, "x[1]": 2, "x[2]": 3})
        6.0
    """

    __slots__ = ("vector",)

    def __init__(self, vector: VectorVariable) -> None:
        self.vector = vector

    def evaluate(
        self, values: Mapping[str, ArrayLike | float]
    ) -> NDArray[np.floating] | float:
        """Evaluate the sum given variable values."""
        return sum(v.evaluate(values) for v in self.vector)  # type: ignore[return-value]

    def get_variables(self) -> set[Variable]:
        """Return all variables this expression depends on."""
        return set(self.vector._variables)

    def __repr__(self) -> str:
        return f"VectorSum({self.vector.name})"


class VectorExpression:
    """A vector of expressions (result of vector arithmetic).

    VectorExpression represents element-wise operations on vectors,
    such as `x + y` or `2 * x`.

    Args:
        expressions: List of scalar expressions, one per element.

    Example:
        >>> x = VectorVariable("x", 3)
        >>> y = VectorVariable("y", 3)
        >>> z = x + y  # VectorExpression with 3 elements
        >>> z[0].evaluate({"x[0]": 1, "y[0]": 2})
        3.0
    """

    __slots__ = ("_expressions", "size")

    def __init__(self, expressions: Sequence[Expression]) -> None:
        if len(expressions) == 0:
            raise ValueError("VectorExpression cannot be empty")
        self._expressions = list(expressions)
        self.size = len(expressions)

    def __getitem__(self, key: int) -> Expression:
        """Get a single expression by index."""
        if key < 0:
            key = self.size + key
        if key < 0 or key >= self.size:
            raise IndexError(
                f"Index {key} out of range for VectorExpression of size {self.size}"
            )
        return self._expressions[key]

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[Expression]:
        return iter(self._expressions)

    def evaluate(self, values: Mapping[str, ArrayLike | float]) -> list[float]:
        """Evaluate all expressions and return as list."""
        return [expr.evaluate(values) for expr in self._expressions]  # type: ignore[misc]

    def get_variables(self) -> set[Variable]:
        """Return all variables these expressions depend on."""
        result: set[Variable] = set()
        for expr in self._expressions:
            result.update(expr.get_variables())
        return result

    def __repr__(self) -> str:
        return f"VectorExpression(size={self.size})"

    # Arithmetic operations - return VectorExpression
    def __add__(
        self, other: VectorExpression | VectorVariable | float | int
    ) -> VectorExpression:
        """Element-wise addition."""
        return _vector_binary_op(self, other, "+")

    def __radd__(self, other: float | int) -> VectorExpression:
        return _vector_binary_op(self, other, "+")

    def __sub__(
        self, other: VectorExpression | VectorVariable | float | int
    ) -> VectorExpression:
        """Element-wise subtraction."""
        return _vector_binary_op(self, other, "-")

    def __rsub__(self, other: float | int) -> VectorExpression:
        # other - self
        return VectorExpression(
            [BinaryOp(_ensure_expr(other), expr, "-") for expr in self._expressions]
        )

    def __mul__(self, other: float | int) -> VectorExpression:
        """Scalar multiplication."""
        return _vector_binary_op(self, other, "*")

    def __rmul__(self, other: float | int) -> VectorExpression:
        return _vector_binary_op(self, other, "*")

    def __truediv__(self, other: float | int) -> VectorExpression:
        """Scalar division."""
        return _vector_binary_op(self, other, "/")

    def __neg__(self) -> VectorExpression:
        """Negate all elements."""
        return VectorExpression([-expr for expr in self._expressions])


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

    # Arithmetic operations - return VectorExpression
    def __add__(
        self, other: VectorVariable | VectorExpression | float | int
    ) -> VectorExpression:
        """Element-wise addition: x + y or x + scalar."""
        return _vector_binary_op(self, other, "+")

    def __radd__(self, other: float | int) -> VectorExpression:
        """Right addition for scalar + vector."""
        return _vector_binary_op(self, other, "+")

    def __sub__(
        self, other: VectorVariable | VectorExpression | float | int
    ) -> VectorExpression:
        """Element-wise subtraction: x - y or x - scalar."""
        return _vector_binary_op(self, other, "-")

    def __rsub__(self, other: float | int) -> VectorExpression:
        """Right subtraction: scalar - vector."""
        return VectorExpression(
            [BinaryOp(_ensure_expr(other), v, "-") for v in self._variables]
        )

    def __mul__(self, other: float | int) -> VectorExpression:
        """Scalar multiplication: x * 2."""
        return _vector_binary_op(self, other, "*")

    def __rmul__(self, other: float | int) -> VectorExpression:
        """Right scalar multiplication: 2 * x."""
        return _vector_binary_op(self, other, "*")

    def __truediv__(self, other: float | int) -> VectorExpression:
        """Scalar division: x / 2."""
        return _vector_binary_op(self, other, "/")

    def __neg__(self) -> VectorExpression:
        """Negate all elements: -x."""
        return VectorExpression([-v for v in self._variables])


def _vector_binary_op(
    left: VectorVariable | VectorExpression,
    right: VectorVariable | VectorExpression | float | int,
    op: Literal["+", "-", "*", "/"],
) -> VectorExpression:
    """Helper for element-wise binary operations on vectors.

    Args:
        left: Left operand (VectorVariable or VectorExpression).
        right: Right operand (vector or scalar).
        op: Operation to perform.

    Returns:
        VectorExpression with element-wise results.

    Raises:
        ValueError: If vector sizes don't match.
    """
    # Get expressions from left
    if isinstance(left, VectorVariable):
        left_exprs = list(left._variables)
    else:
        left_exprs = list(left._expressions)

    # Handle right operand
    if isinstance(right, (int, float)):
        # Scalar broadcast
        right_exprs = [Constant(right)] * len(left_exprs)
    elif isinstance(right, VectorVariable):
        if len(right) != len(left_exprs):
            raise ValueError(f"Vector size mismatch: {len(left_exprs)} vs {len(right)}")
        right_exprs = list(right._variables)
    elif isinstance(right, VectorExpression):
        if right.size != len(left_exprs):
            raise ValueError(f"Vector size mismatch: {len(left_exprs)} vs {right.size}")
        right_exprs = list(right._expressions)
    else:
        raise TypeError(f"Unsupported operand type: {type(right)}")

    # Create element-wise operations
    result_exprs = [
        BinaryOp(left_expr, right_expr, op)
        for left_expr, right_expr in zip(left_exprs, right_exprs)
    ]

    return VectorExpression(result_exprs)


def vector_sum(vector: VectorVariable | VectorExpression) -> VectorSum | Expression:
    """Sum all elements of a vector.

    Args:
        vector: VectorVariable or VectorExpression to sum.

    Returns:
        VectorSum expression for VectorVariable, or built expression for VectorExpression.

    Example:
        >>> x = VectorVariable("x", 3)
        >>> s = vector_sum(x)
        >>> s.evaluate({"x[0]": 1, "x[1]": 2, "x[2]": 3})
        6.0
    """
    if isinstance(vector, VectorVariable):
        return VectorSum(vector)
    elif isinstance(vector, VectorExpression):
        # Build sum expression from individual expressions
        if vector.size == 0:
            return Constant(0)
        result: Expression = vector._expressions[0]
        for expr in vector._expressions[1:]:
            result = result + expr
        return result
    else:
        raise TypeError(
            f"Expected VectorVariable or VectorExpression, got {type(vector)}"
        )
