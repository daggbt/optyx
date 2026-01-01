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
from optyx.core.errors import DimensionMismatchError

if TYPE_CHECKING:
    from optyx.constraints import Constraint
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


class DotProduct(Expression):
    """Dot product of two vectors: x · y = x[0]*y[0] + x[1]*y[1] + ... + x[n-1]*y[n-1].

    This is a scalar expression representing the inner product.

    Args:
        left: First vector.
        right: Second vector.

    Example:
        >>> x = VectorVariable("x", 3)
        >>> y = VectorVariable("y", 3)
        >>> d = DotProduct(x, y)
        >>> d.evaluate({"x[0]": 1, "x[1]": 2, "x[2]": 3, "y[0]": 4, "y[1]": 5, "y[2]": 6})
        32.0
    """

    __slots__ = ("left", "right")

    def __init__(
        self,
        left: VectorVariable | VectorExpression,
        right: VectorVariable | VectorExpression,
    ) -> None:
        left_size = (
            left.size
            if isinstance(left, (VectorVariable, VectorExpression))
            else len(left)
        )
        right_size = (
            right.size
            if isinstance(right, (VectorVariable, VectorExpression))
            else len(right)
        )
        if left_size != right_size:
            raise DimensionMismatchError(
                operation="dot product",
                left_shape=left_size,
                right_shape=right_size,
                suggestion="Vectors must have the same length.",
            )
        self.left = left
        self.right = right

    def evaluate(
        self, values: Mapping[str, ArrayLike | float]
    ) -> NDArray[np.floating] | float:
        """Evaluate the dot product given variable values."""
        left_vals = [v.evaluate(values) for v in self._iter_left()]
        right_vals = [v.evaluate(values) for v in self._iter_right()]
        return sum(lv * rv for lv, rv in zip(left_vals, right_vals))  # type: ignore[return-value]

    def _iter_left(self) -> Iterator[Expression]:
        """Iterate over left vector elements."""
        if isinstance(self.left, VectorVariable):
            return iter(self.left._variables)
        return iter(self.left._expressions)

    def _iter_right(self) -> Iterator[Expression]:
        """Iterate over right vector elements."""
        if isinstance(self.right, VectorVariable):
            return iter(self.right._variables)
        return iter(self.right._expressions)

    def get_variables(self) -> set[Variable]:
        """Return all variables this expression depends on."""
        result: set[Variable] = set()
        if isinstance(self.left, VectorVariable):
            result.update(self.left._variables)
        else:
            result.update(self.left.get_variables())
        if isinstance(self.right, VectorVariable):
            result.update(self.right._variables)
        else:
            result.update(self.right.get_variables())
        return result

    def __repr__(self) -> str:
        left_name = self.left.name if isinstance(self.left, VectorVariable) else "expr"
        right_name = (
            self.right.name if isinstance(self.right, VectorVariable) else "expr"
        )
        return f"DotProduct({left_name}, {right_name})"


class L2Norm(Expression):
    """L2 (Euclidean) norm of a vector: ||x|| = sqrt(x[0]^2 + x[1]^2 + ... + x[n-1]^2).

    This is a scalar expression representing the Euclidean length.

    Args:
        vector: The vector to compute the norm of.

    Example:
        >>> x = VectorVariable("x", 2)
        >>> n = L2Norm(x)
        >>> n.evaluate({"x[0]": 3, "x[1]": 4})
        5.0
    """

    __slots__ = ("vector",)

    def __init__(self, vector: VectorVariable | VectorExpression) -> None:
        self.vector = vector

    def evaluate(
        self, values: Mapping[str, ArrayLike | float]
    ) -> NDArray[np.floating] | float:
        """Evaluate the L2 norm given variable values."""
        vals = [v.evaluate(values) for v in self._iter_vector()]
        sum_sq = sum(v * v for v in vals)
        return np.sqrt(sum_sq)  # type: ignore[return-value]

    def _iter_vector(self) -> Iterator[Expression]:
        """Iterate over vector elements."""
        if isinstance(self.vector, VectorVariable):
            return iter(self.vector._variables)
        return iter(self.vector._expressions)

    def get_variables(self) -> set[Variable]:
        """Return all variables this expression depends on."""
        if isinstance(self.vector, VectorVariable):
            return set(self.vector._variables)
        return self.vector.get_variables()

    def __repr__(self) -> str:
        vec_name = (
            self.vector.name if isinstance(self.vector, VectorVariable) else "expr"
        )
        return f"L2Norm({vec_name})"


class L1Norm(Expression):
    """L1 (Manhattan) norm of a vector: ||x||_1 = |x[0]| + |x[1]| + ... + |x[n-1]|.

    This is a scalar expression representing the sum of absolute values.

    Args:
        vector: The vector to compute the norm of.

    Example:
        >>> x = VectorVariable("x", 3)
        >>> n = L1Norm(x)
        >>> n.evaluate({"x[0]": 1, "x[1]": -2, "x[2]": 3})
        6.0
    """

    __slots__ = ("vector",)

    def __init__(self, vector: VectorVariable | VectorExpression) -> None:
        self.vector = vector

    def evaluate(
        self, values: Mapping[str, ArrayLike | float]
    ) -> NDArray[np.floating] | float:
        """Evaluate the L1 norm given variable values."""
        vals = [v.evaluate(values) for v in self._iter_vector()]
        return sum(abs(v) for v in vals)  # type: ignore[return-value]

    def _iter_vector(self) -> Iterator[Expression]:
        """Iterate over vector elements."""
        if isinstance(self.vector, VectorVariable):
            return iter(self.vector._variables)
        return iter(self.vector._expressions)

    def get_variables(self) -> set[Variable]:
        """Return all variables this expression depends on."""
        if isinstance(self.vector, VectorVariable):
            return set(self.vector._variables)
        return self.vector.get_variables()

    def __repr__(self) -> str:
        vec_name = (
            self.vector.name if isinstance(self.vector, VectorVariable) else "expr"
        )
        return f"L1Norm({vec_name})"


class LinearCombination(Expression):
    """Linear combination of vector elements with constant coefficients.

    Represents: c[0]*x[0] + c[1]*x[1] + ... + c[n-1]*x[n-1]

    This enables efficient numpy integration: `coefficients @ vector`.

    Args:
        coefficients: NumPy array of constant coefficients.
        vector: VectorVariable or VectorExpression to combine.

    Example:
        >>> import numpy as np
        >>> returns = np.array([0.12, 0.08, 0.10])
        >>> weights = VectorVariable("w", 3)
        >>> portfolio_return = LinearCombination(returns, weights)
        >>> portfolio_return.evaluate({"w[0]": 0.5, "w[1]": 0.3, "w[2]": 0.2})
        0.084
    """

    __slots__ = ("coefficients", "vector")

    def __init__(
        self,
        coefficients: np.ndarray,
        vector: VectorVariable | VectorExpression,
    ) -> None:
        coefficients = np.asarray(coefficients)
        vec_size = vector.size if hasattr(vector, "size") else len(vector)
        if len(coefficients) != vec_size:
            raise DimensionMismatchError(
                operation="linear combination",
                left_shape=len(coefficients),
                right_shape=vec_size,
                suggestion="Coefficient array length must match vector size.",
            )
        self.coefficients = coefficients
        self.vector = vector

    def evaluate(
        self, values: Mapping[str, ArrayLike | float]
    ) -> NDArray[np.floating] | float:
        """Evaluate the linear combination given variable values."""
        vals = np.array([v.evaluate(values) for v in self._iter_vector()])
        return float(np.dot(self.coefficients, vals))

    def _iter_vector(self) -> Iterator[Expression]:
        """Iterate over vector elements."""
        if isinstance(self.vector, VectorVariable):
            return iter(self.vector._variables)
        return iter(self.vector._expressions)

    def get_variables(self) -> set[Variable]:
        """Return all variables this expression depends on."""
        if isinstance(self.vector, VectorVariable):
            return set(self.vector._variables)
        return self.vector.get_variables()

    def __repr__(self) -> str:
        vec_name = (
            self.vector.name if isinstance(self.vector, VectorVariable) else "expr"
        )
        return f"LinearCombination({len(self.coefficients)} coeffs, {vec_name})"


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

    # Tell NumPy to defer to Python's operators
    __array_ufunc__ = None

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

    def sum(self) -> Expression:
        """Sum of all elements."""
        if not self._expressions:
            return Constant(0.0)
        result: Expression = self._expressions[0]
        for expr in self._expressions[1:]:
            result = result + expr
        return result

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

    def __rtruediv__(self, other: float | int) -> VectorExpression:
        """Right scalar division."""
        return VectorExpression(
            [BinaryOp(_ensure_expr(other), expr, "/") for expr in self._expressions]
        )

    def __neg__(self) -> VectorExpression:
        """Negate all elements."""
        return VectorExpression([-expr for expr in self._expressions])

    def __pow__(self, other: float | int) -> VectorExpression:
        """Element-wise power."""
        return _vector_binary_op(self, other, "**")

    # Comparison operators - create lists of constraints
    def __le__(
        self, other: VectorExpression | VectorVariable | float | int
    ) -> list[Constraint]:
        """Element-wise <= constraint.

        Example:
            >>> x = VectorVariable("x", 3)
            >>> constraints = x <= 10  # 3 constraints: x[i] <= 10
        """
        return _vector_constraint(self, other, "<=")

    def __ge__(
        self, other: VectorExpression | VectorVariable | float | int
    ) -> list[Constraint]:
        """Element-wise >= constraint.

        Example:
            >>> x = VectorVariable("x", 3)
            >>> constraints = x >= 0  # 3 constraints: x[i] >= 0
        """
        return _vector_constraint(self, other, ">=")

    def eq(
        self, other: VectorExpression | VectorVariable | float | int
    ) -> list[Constraint]:
        """Element-wise == constraint.

        Example:
            >>> x = VectorVariable("x", 3)
            >>> y = VectorVariable("y", 3)
            >>> constraints = x.eq(y)  # 3 constraints: x[i] == y[i]
        """
        return _vector_constraint(self, other, "==")

    def dot(self, other: VectorExpression | VectorVariable) -> DotProduct:
        """Compute dot product with another vector.

        Args:
            other: Vector to compute dot product with.

        Returns:
            DotProduct expression (scalar).

        Example:
            >>> x = VectorVariable("x", 3)
            >>> y = VectorVariable("y", 3)
            >>> d = x.dot(y)
            >>> d.evaluate({"x[0]": 1, "x[1]": 2, "x[2]": 3, "y[0]": 4, "y[1]": 5, "y[2]": 6})
            32.0
        """
        return DotProduct(self, other)

    def __matmul__(
        self, other: VectorExpression | VectorVariable | np.ndarray | list
    ) -> DotProduct | LinearCombination:
        """Matrix multiplication operator for dot product.

        For VectorExpression @ VectorVariable: returns DotProduct (same as .dot())
        For VectorExpression @ array: returns LinearCombination

        Args:
            other: Vector or array to compute dot product with.

        Returns:
            DotProduct or LinearCombination expression (scalar).

        Example:
            >>> x = VectorVariable("x", 3)
            >>> y = VectorVariable("y", 3)
            >>> expr = x + 1
            >>> d = expr @ y  # Dot product of (x + 1) with y
        """
        if isinstance(other, (VectorVariable, VectorExpression)):
            return DotProduct(self, other)
        elif isinstance(other, (np.ndarray, list)):
            arr = np.asarray(other)
            if arr.ndim != 1:
                raise ValueError(f"Expected 1D array, got shape {arr.shape}")
            return LinearCombination(arr, self)
        else:
            return NotImplemented

    def __rmatmul__(self, other: np.ndarray | list) -> LinearCombination:
        """Right matrix multiplication: array @ vector_expr.

        Args:
            other: Array to compute dot product with.

        Returns:
            LinearCombination expression (scalar).
        """
        arr = np.asarray(other)
        if arr.ndim != 1:
            return NotImplemented
        return LinearCombination(arr, self)


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

    # Tell NumPy to defer to Python's operators (enables numpy_array @ vector)
    __array_ufunc__ = None

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

    def __rtruediv__(self, other: float | int) -> VectorExpression:
        """Right scalar division: 1 / x."""
        return VectorExpression(
            [BinaryOp(_ensure_expr(other), v, "/") for v in self._variables]
        )

    def __neg__(self) -> VectorExpression:
        """Negate all elements: -x."""
        return VectorExpression([-v for v in self._variables])

    def __pow__(self, other: float | int) -> VectorExpression:
        """Element-wise power: x ** 2."""
        return _vector_binary_op(self, other, "**")

    # Comparison operators - create lists of constraints
    def __le__(
        self, other: VectorVariable | VectorExpression | float | int
    ) -> list[Constraint]:
        """Element-wise <= constraint.

        Example:
            >>> x = VectorVariable("x", 3)
            >>> constraints = x <= 10  # 3 constraints: x[i] <= 10
        """
        return _vector_constraint(self, other, "<=")

    def __ge__(
        self, other: VectorVariable | VectorExpression | float | int
    ) -> list[Constraint]:
        """Element-wise >= constraint.

        Example:
            >>> x = VectorVariable("x", 3)
            >>> constraints = x >= 0  # 3 constraints: x[i] >= 0
        """
        return _vector_constraint(self, other, ">=")

    def eq(
        self, other: VectorVariable | VectorExpression | float | int
    ) -> list[Constraint]:
        """Element-wise == constraint.

        Example:
            >>> x = VectorVariable("x", 3)
            >>> y = VectorVariable("y", 3)
            >>> constraints = x.eq(y)  # 3 constraints: x[i] == y[i]
        """
        return _vector_constraint(self, other, "==")

    def dot(self, other: VectorVariable | VectorExpression) -> DotProduct:
        """Compute dot product with another vector.

        Args:
            other: Vector to compute dot product with.

        Returns:
            DotProduct expression (scalar).

        Example:
            >>> x = VectorVariable("x", 3)
            >>> y = VectorVariable("y", 3)
            >>> d = x.dot(y)
            >>> d.evaluate({"x[0]": 1, "x[1]": 2, "x[2]": 3, "y[0]": 4, "y[1]": 5, "y[2]": 6})
            32.0
        """
        return DotProduct(self, other)

    def __matmul__(
        self, other: VectorVariable | VectorExpression | np.ndarray | list
    ) -> DotProduct | LinearCombination:
        """Matrix multiplication operator for dot product.

        For VectorVariable @ VectorVariable: returns DotProduct (same as .dot())
        For VectorVariable @ array: returns LinearCombination (coefficients @ vector)

        Args:
            other: Vector or array to compute dot product with.

        Returns:
            DotProduct or LinearCombination expression (scalar).

        Example:
            >>> x = VectorVariable("x", 3)
            >>> y = VectorVariable("y", 3)
            >>> d = x @ y  # Same as x.dot(y)
            >>> coeffs = np.array([1, 2, 3])
            >>> lc = x @ coeffs  # Same as coeffs @ x
        """
        if isinstance(other, (VectorVariable, VectorExpression)):
            return DotProduct(self, other)
        elif isinstance(other, (np.ndarray, list)):
            # VectorVariable @ array is the same as array @ VectorVariable
            arr = np.asarray(other)
            if arr.ndim != 1:
                raise ValueError(f"Expected 1D array, got shape {arr.shape}")
            return LinearCombination(arr, self)
        else:
            return NotImplemented

    def sum(self) -> VectorSum:
        """Compute sum of all elements in the vector.

        Returns:
            VectorSum expression (scalar).

        Example:
            >>> x = VectorVariable("x", 3)
            >>> s = x.sum()
            >>> s.evaluate({"x[0]": 1, "x[1]": 2, "x[2]": 3})
            6.0
        """
        return VectorSum(self)

    def norm(self, ord: int = 2) -> L2Norm | L1Norm:
        """Compute the norm of this vector.

        Args:
            ord: Order of the norm. 2 for L2 (Euclidean), 1 for L1 (Manhattan).

        Returns:
            L2Norm or L1Norm expression (scalar).

        Example:
            >>> x = VectorVariable("x", 2)
            >>> n = x.norm()  # L2 norm
            >>> n.evaluate({"x[0]": 3, "x[1]": 4})
            5.0
            >>> n1 = x.norm(1)  # L1 norm
            >>> n1.evaluate({"x[0]": 3, "x[1]": -4})
            7.0
        """
        if ord == 2:
            return L2Norm(self)
        elif ord == 1:
            return L1Norm(self)
        else:
            raise ValueError(f"Unsupported norm order: {ord}. Use 1 or 2.")

    def __rmatmul__(self, other: np.ndarray) -> LinearCombination:
        """Enable numpy_array @ vector syntax for linear combinations.

        Args:
            other: NumPy array of coefficients.

        Returns:
            LinearCombination expression.

        Example:
            >>> import numpy as np
            >>> returns = np.array([0.12, 0.08, 0.10])
            >>> weights = VectorVariable("w", 3, lb=0, ub=1)
            >>> portfolio_return = returns @ weights
        """
        return LinearCombination(np.asarray(other), self)

    def to_numpy(self, solution: Mapping[str, float]) -> np.ndarray:
        """Extract solution values as a NumPy array.

        Args:
            solution: Dictionary mapping variable names to values.

        Returns:
            NumPy array of solution values in order.

        Example:
            >>> x = VectorVariable("x", 3)
            >>> solution = {"x[0]": 1.0, "x[1]": 2.0, "x[2]": 3.0}
            >>> x.to_numpy(solution)
            array([1., 2., 3.])
        """
        return np.array([solution[v.name] for v in self._variables])

    @classmethod
    def from_numpy(
        cls,
        name: str,
        array: np.ndarray,
        lb: float | None = None,
        ub: float | None = None,
        domain: DomainType = "continuous",
    ) -> VectorVariable:
        """Create a VectorVariable with size inferred from a NumPy array.

        The array values are not stored - this is just a convenience
        method to create a vector matching the array's shape.

        Args:
            name: Base name for the vector variables.
            array: NumPy array to match size (1D array expected).
            lb: Lower bound for all variables.
            ub: Upper bound for all variables.
            domain: Variable domain type.

        Returns:
            VectorVariable with size matching the array.

        Example:
            >>> import numpy as np
            >>> data = np.array([1.0, 2.0, 3.0, 4.0])
            >>> x = VectorVariable.from_numpy("x", data, lb=0)
            >>> len(x)
            4
        """
        array = np.asarray(array)
        if array.ndim != 1:
            raise ValueError(f"Expected 1D array, got {array.ndim}D")
        return cls(name, len(array), lb=lb, ub=ub, domain=domain)


def _vector_constraint(
    left: VectorVariable | VectorExpression,
    right: VectorVariable | VectorExpression | float | int | np.ndarray | list,
    sense: Literal["<=", ">=", "=="],
) -> list[Constraint]:
    """Create element-wise constraints for vectors.

    Args:
        left: Left operand (VectorVariable or VectorExpression).
        right: Right operand (vector, scalar, numpy array, or list).
        sense: Constraint sense (<=, >=, or ==).

    Returns:
        List of Constraint objects, one per element.

    Raises:
        ValueError: If vector sizes don't match.
    """
    from optyx.constraints import _make_constraint

    # Get expressions from left
    if isinstance(left, VectorVariable):
        left_exprs: list[Expression] = list(left._variables)
    else:
        left_exprs = list(left._expressions)

    # Handle right operand
    if isinstance(right, (int, float)):
        # Scalar broadcast - create constraints directly
        return [_make_constraint(expr, sense, right) for expr in left_exprs]
    elif isinstance(right, VectorVariable):
        if len(right) != len(left_exprs):
            raise DimensionMismatchError(
                operation=f"vector constraint ({sense})",
                left_shape=len(left_exprs),
                right_shape=len(right),
            )
        right_exprs: list[Expression] = list(right._variables)
    elif isinstance(right, VectorExpression):
        if right.size != len(left_exprs):
            raise DimensionMismatchError(
                operation=f"vector constraint ({sense})",
                left_shape=len(left_exprs),
                right_shape=right.size,
            )
        right_exprs = list(right._expressions)
    elif isinstance(right, (np.ndarray, list)):
        # Handle numpy arrays and lists
        right_arr = np.asarray(right)
        if right_arr.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {right_arr.shape}")
        if len(right_arr) != len(left_exprs):
            raise ValueError(
                f"Vector size mismatch: {len(left_exprs)} vs {len(right_arr)}"
            )
        # Create constraints with scalar values from array
        return [
            _make_constraint(left_expr, sense, float(val))
            for left_expr, val in zip(left_exprs, right_arr)
        ]
    else:
        raise TypeError(f"Unsupported operand type: {type(right)}")

    # Create element-wise constraints
    return [
        _make_constraint(left_expr, sense, right_expr)
        for left_expr, right_expr in zip(left_exprs, right_exprs)
    ]


def _vector_binary_op(
    left: VectorVariable | VectorExpression,
    right: VectorVariable | VectorExpression | float | int,
    op: Literal["+", "-", "*", "/", "**"],
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
    elif isinstance(right, (np.ndarray, list)):
        arr = np.asarray(right)
        if arr.ndim != 1:
            raise ValueError(f"Expected 1D array, got shape {arr.shape}")
        if len(arr) != len(left_exprs):
            raise ValueError(f"Vector size mismatch: {len(left_exprs)} vs {len(arr)}")
        right_exprs = [Constant(val) for val in arr]
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


def norm(vector: VectorVariable | VectorExpression, ord: int = 2) -> L2Norm | L1Norm:
    """Compute the norm of a vector.

    Args:
        vector: VectorVariable or VectorExpression to compute norm of.
        ord: Order of the norm. 2 for L2 (Euclidean), 1 for L1 (Manhattan).

    Returns:
        L2Norm or L1Norm expression.

    Example:
        >>> x = VectorVariable("x", 2)
        >>> n = norm(x)
        >>> n.evaluate({"x[0]": 3, "x[1]": 4})
        5.0
        >>> n1 = norm(x, ord=1)
        >>> n1.evaluate({"x[0]": 3, "x[1]": -4})
        7.0
    """
    if ord == 2:
        return L2Norm(vector)
    elif ord == 1:
        return L1Norm(vector)
    else:
        raise ValueError(f"Unsupported norm order: {ord}. Use 1 or 2.")
