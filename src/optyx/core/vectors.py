"""Vector variables for optimization problems.

This module provides VectorVariable for representing vectors of decision variables,
enabling natural syntax like `x = VectorVariable("x", 100)` with indexing and slicing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Iterator, Literal, Mapping, cast, overload

import numpy as np

from optyx.core.expressions import (
    Expression,
    Variable,
    Constant,
    BinaryOp,
)
from optyx.core.errors import (
    DimensionMismatchError,
    EmptyContainerError,
    InvalidSizeError,
    InvalidOperationError,
    WrongDimensionalityError,
)

if TYPE_CHECKING:
    from optyx.constraints import Constraint
    from optyx.core.matrices import MatrixVectorProduct
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
        self._hash = None

    def evaluate(
        self, values: Mapping[str, ArrayLike | float]
    ) -> NDArray[np.floating] | float:
        """Evaluate the sum given variable values."""
        return float(
            np.sum(
                np.fromiter(
                    (v.evaluate(values) for v in self.vector),
                    dtype=float,
                    count=self.vector.size,
                )
            )
        )

    def get_variables(self) -> set[Variable]:
        """Return all variables this expression depends on."""
        return set(self.vector._variables)

    def jacobian_row(self, variables: list[Variable]) -> list[Expression] | None:
        """Return Jacobian row in O(1) if variables match vector variables.

        For VectorSum(x), the gradient is 1 for each x[i], 0 otherwise.
        This is O(1) to construct vs O(n) individual gradient calls.

        Returns:
            List of Constant expressions, or None if optimization not applicable.
        """
        # Build a lookup for fast variable matching
        my_vars = set(self.vector._variables)
        result: list[Expression] = []
        for var in variables:
            if var in my_vars:
                result.append(Constant(1.0))
            else:
                result.append(Constant(0.0))
        return result

    def __repr__(self) -> str:
        return f"VectorSum({self.vector.name})"


class VectorExpressionSum(Expression):
    """Sum of all elements in a VectorExpression.

    This is a scalar expression representing sum(expr) for a VectorExpression.
    Unlike the nested BinaryOp approach, this stores a flat reference to avoid
    RecursionError on large vectors.

    Args:
        expression: The VectorExpression to sum.

    Example:
        >>> x = VectorVariable("x", 3)
        >>> y = VectorVariable("y", 3)
        >>> s = VectorExpressionSum(x - y)
        >>> s.evaluate({"x[0]": 4, "x[1]": 5, "x[2]": 6, "y[0]": 1, "y[1]": 2, "y[2]": 3})
        9.0
    """

    __slots__ = ("expression",)

    def __init__(self, expression: VectorExpression) -> None:
        self.expression = expression
        self._hash = None

    @property
    def size(self) -> int:
        """Size of the underlying vector expression."""
        return self.expression.size

    def evaluate(
        self, values: Mapping[str, ArrayLike | float]
    ) -> NDArray[np.floating] | float:
        """Evaluate the sum given variable values.

        For VectorBinaryOp, uses its vectorized evaluate() + np.sum()
        to avoid triggering O(N) materialization of scalar expressions.
        """
        # Use type name check to avoid forward reference issue
        # (VectorBinaryOp is defined later in this module)
        if type(self.expression).__name__ == "VectorBinaryOp":
            return float(np.sum(np.array(self.expression.evaluate(values))))
        return sum(e.evaluate(values) for e in self.expression._expressions)  # type: ignore[return-value]

    def get_variables(self) -> set[Variable]:
        """Return all variables this expression depends on."""
        return self.expression.get_variables()

    def jacobian_row(self, variables: list[Variable]) -> list[Expression] | None:
        """Return Jacobian row.

        For VectorExpressionSum, gradient w.r.t. each variable depends on
        the underlying expressions. We delegate to the autodiff system.
        """
        return None  # Let autodiff handle the general case

    def __repr__(self) -> str:
        return f"VectorExpressionSum(size={self.expression.size})"


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
        self._hash = None

    def evaluate(
        self, values: Mapping[str, ArrayLike | float]
    ) -> NDArray[np.floating] | float:
        """Evaluate the dot product given variable values."""
        size = self.left.size
        left_vals = np.fromiter(
            (v.evaluate(values) for v in self._iter_left()),
            dtype=float,
            count=size,
        )
        right_vals = np.fromiter(
            (v.evaluate(values) for v in self._iter_right()),
            dtype=float,
            count=size,
        )
        return float(np.dot(left_vals, right_vals))

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

    def jacobian_row(self, variables: list[Variable]) -> list[Expression] | None:
        """Return Jacobian row in O(n) for special cases.

        For DotProduct(x, x) (same VectorVariable), gradient is 2*x[i].
        For DotProduct(x, y) with two different VectorVariables, gradient
        is y[i] for x[i] and x[i] for y[i].

        Returns:
            List of expressions, or None if optimization not applicable.
        """
        # Only optimize for VectorVariable cases
        if not isinstance(self.left, VectorVariable):
            return None
        if not isinstance(self.right, VectorVariable):
            return None

        left_vars = self.left._variables
        right_vars = self.right._variables

        # Case 1: x.dot(x) -> gradient is 2*x[i]
        if self.left is self.right:
            var_to_elem: dict[Variable, Expression] = {
                v: BinaryOp(Constant(2.0), v, "*") for v in left_vars
            }
            return [var_to_elem.get(v, Constant(0.0)) for v in variables]

        # Case 2: x.dot(y) -> gradient is y[i] w.r.t. x[i], x[i] w.r.t. y[i]
        left_lookup = {left_vars[i]: right_vars[i] for i in range(len(left_vars))}
        right_lookup = {right_vars[i]: left_vars[i] for i in range(len(right_vars))}

        result: list[Expression] = []
        for var in variables:
            if var in left_lookup:
                result.append(left_lookup[var])
            elif var in right_lookup:
                result.append(right_lookup[var])
            else:
                result.append(Constant(0.0))
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
        self._hash = None

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
        self._hash = None

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
        self._hash = None

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

    def jacobian_row(self, variables: list[Variable]) -> list[Expression] | None:
        """Return Jacobian row in O(n) - coefficients are the gradients.

        For LinearCombination(c, x), gradient is c[i] for x[i], 0 otherwise.
        This is O(n) but avoids n separate gradient() calls.

        Returns:
            List of Constant expressions, or None if optimization not applicable.
        """
        # Only optimize for VectorVariable case
        if not isinstance(self.vector, VectorVariable):
            return None

        # Map each variable to its coefficient
        var_to_coeff: dict[Variable, float] = {}
        for i, var in enumerate(self.vector._variables):
            var_to_coeff[var] = float(self.coefficients[i])

        return [Constant(var_to_coeff.get(v, 0.0)) for v in variables]

    def __repr__(self) -> str:
        vec_name = (
            self.vector.name if isinstance(self.vector, VectorVariable) else "expr"
        )
        return f"LinearCombination({len(self.coefficients)} coeffs, {vec_name})"


# =============================================================================
# Vectorized NLP Expression Types
# =============================================================================
# These expression types enable O(1) numpy-based evaluation and gradient
# computation for common non-linear patterns, avoiding O(n) expression tree
# traversal.


class ElementwisePower(Expression):
    """Element-wise power of a vector: x[i] ** k for each element.

    This is a vector expression representing x ** k element-wise.
    Enables O(1) evaluation using numpy instead of n separate BinaryOp nodes.

    Args:
        vector: The VectorVariable to raise to power.
        power: The exponent (constant).

    Example:
        >>> x = VectorVariable("x", 3)
        >>> x_sq = ElementwisePower(x, 2)
        >>> x_sq.evaluate({"x[0]": 1, "x[1]": 2, "x[2]": 3})
        array([1., 4., 9.])
    """

    __slots__ = ("vector", "power")

    def __init__(self, vector: VectorVariable, power: float | int) -> None:
        self.vector = vector
        self.power = float(power)
        self._hash = None

    @property
    def size(self) -> int:
        """Number of elements."""
        return self.vector.size

    def evaluate(self, values: Mapping[str, ArrayLike | float]) -> NDArray[np.floating]:
        """Evaluate element-wise power using numpy."""
        vals = np.array([v.evaluate(values) for v in self.vector._variables])
        return vals**self.power

    def get_variables(self) -> set[Variable]:
        """Return all variables this expression depends on."""
        return set(self.vector._variables)

    def sum(self) -> "VectorPowerSum":
        """Sum of all powered elements: sum(x ** k).

        Returns VectorPowerSum for O(1) evaluation instead of nested BinaryOps.
        """
        return VectorPowerSum(self.vector, self.power)

    def __iter__(self) -> Iterator[Expression]:
        """Iterate over element-wise power expressions."""
        for var in self.vector._variables:
            yield BinaryOp(var, Constant(self.power), "**")

    def __getitem__(self, idx: int) -> Expression:
        """Get single element: x[i] ** k."""
        return BinaryOp(self.vector._variables[idx], Constant(self.power), "**")

    def __repr__(self) -> str:
        return f"ElementwisePower({self.vector.name}, {self.power})"


class VectorPowerSum(Expression):
    """Sum of element-wise powers: sum(x ** k) = x[0]**k + x[1]**k + ... + x[n-1]**k.

    This is a scalar expression representing sum(x ** k).
    Enables O(1) evaluation and gradient computation using numpy.

    Args:
        vector: The VectorVariable to raise to power and sum.
        power: The exponent (constant).

    Example:
        >>> x = VectorVariable("x", 3)
        >>> s = VectorPowerSum(x, 2)
        >>> s.evaluate({"x[0]": 1, "x[1]": 2, "x[2]": 3})
        14.0  # 1^2 + 2^2 + 3^2
    """

    __slots__ = ("vector", "power")

    def __init__(self, vector: VectorVariable, power: float | int) -> None:
        self.vector = vector
        self.power = float(power)
        self._hash = None

    def evaluate(self, values: Mapping[str, ArrayLike | float]) -> float:
        """Evaluate sum of powers using numpy."""
        vals = np.array([v.evaluate(values) for v in self.vector._variables])
        return float(np.sum(vals**self.power))

    def get_variables(self) -> set[Variable]:
        """Return all variables this expression depends on."""
        return set(self.vector._variables)

    def jacobian_row(self, variables: list[Variable]) -> list[Expression] | None:
        """Return Jacobian row in O(n).

        For VectorPowerSum(x, k), gradient w.r.t. x[i] is k * x[i] ** (k-1).

        Returns:
            List of expressions, or None if optimization not applicable.
        """
        k = self.power
        my_vars = set(self.vector._variables)
        result: list[Expression] = []

        for var in variables:
            if var in my_vars:
                # Gradient: k * x[i] ** (k-1)
                if k == 1:
                    result.append(Constant(1.0))
                elif k == 2:
                    # 2 * x[i]
                    result.append(BinaryOp(Constant(2.0), var, "*"))
                else:
                    # k * x[i] ** (k-1)
                    power_term = BinaryOp(var, Constant(k - 1), "**")
                    result.append(BinaryOp(Constant(k), power_term, "*"))
            else:
                result.append(Constant(0.0))

        return result

    def __repr__(self) -> str:
        return f"VectorPowerSum({self.vector.name}, {self.power})"


class ElementwiseUnary(Expression):
    """Element-wise unary operation on a vector: f(x[i]) for each element.

    This is a vector expression representing f(x) element-wise.
    Enables O(1) evaluation using numpy instead of n separate UnaryOp nodes.

    Args:
        vector: The VectorVariable to apply function to.
        op: The operation name ('sin', 'cos', 'exp', 'log', 'abs', 'sqrt').

    Example:
        >>> x = VectorVariable("x", 3)
        >>> sin_x = ElementwiseUnary(x, "sin")
        >>> sin_x.evaluate({"x[0]": 0, "x[1]": np.pi/2, "x[2]": np.pi})
        array([0., 1., 0.])
    """

    __slots__ = ("vector", "op")

    # Mapping from op name to numpy function
    _NUMPY_FUNCS = {
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "abs": np.abs,
        "sqrt": np.sqrt,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
    }

    def __init__(self, vector: VectorVariable, op: str) -> None:
        if op not in self._NUMPY_FUNCS:
            raise InvalidOperationError(
                operation=op,
                operand_types="VectorVariable",
                reason=f"Unsupported operation. Supported: {list(self._NUMPY_FUNCS.keys())}",
            )
        self.vector = vector
        self.op = op
        self._hash = None

    @property
    def size(self) -> int:
        """Number of elements."""
        return self.vector.size

    def evaluate(self, values: Mapping[str, ArrayLike | float]) -> NDArray[np.floating]:
        """Evaluate element-wise function using numpy."""
        vals = np.array([v.evaluate(values) for v in self.vector._variables])
        return self._NUMPY_FUNCS[self.op](vals)

    def get_variables(self) -> set[Variable]:
        """Return all variables this expression depends on."""
        return set(self.vector._variables)

    def sum(self) -> "VectorUnarySum":
        """Sum of all function values: sum(f(x)).

        Returns VectorUnarySum for O(1) evaluation.
        """
        return VectorUnarySum(self.vector, self.op)

    def __iter__(self) -> Iterator[Expression]:
        """Iterate over element-wise unary expressions."""
        from optyx.core.expressions import UnaryOp

        for var in self.vector._variables:
            yield UnaryOp(var, self.op)

    def __getitem__(self, idx: int) -> Expression:
        """Get single element: f(x[i])."""
        from optyx.core.expressions import UnaryOp

        return UnaryOp(self.vector._variables[idx], self.op)

    def __repr__(self) -> str:
        return f"ElementwiseUnary({self.op}, {self.vector.name})"


class VectorUnarySum(Expression):
    """Sum of element-wise unary function: sum(f(x)) = f(x[0]) + f(x[1]) + ... + f(x[n-1]).

    This is a scalar expression representing sum(f(x)).
    Enables O(1) evaluation and gradient computation using numpy.

    Args:
        vector: The VectorVariable to apply function to and sum.
        op: The operation name ('sin', 'cos', 'exp', 'log', etc.).

    Example:
        >>> x = VectorVariable("x", 3)
        >>> s = VectorUnarySum(x, "sin")
        >>> s.evaluate({"x[0]": 0, "x[1]": np.pi/2, "x[2]": np.pi})
        1.0  # sin(0) + sin(pi/2) + sin(pi) ≈ 0 + 1 + 0
    """

    __slots__ = ("vector", "op")

    # Mapping from op name to numpy function
    _NUMPY_FUNCS = {
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "abs": np.abs,
        "sqrt": np.sqrt,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "tanh": np.tanh,
    }

    # Mapping from op to derivative op
    _DERIVATIVE_OPS = {
        "sin": "cos",  # d/dx sin(x) = cos(x)
        "cos": "-sin",  # d/dx cos(x) = -sin(x)
        "exp": "exp",  # d/dx exp(x) = exp(x)
        "log": "1/x",  # d/dx log(x) = 1/x
        "sqrt": "1/2sqrt",  # d/dx sqrt(x) = 1/(2*sqrt(x))
        "sinh": "cosh",  # d/dx sinh(x) = cosh(x)
        "cosh": "sinh",  # d/dx cosh(x) = sinh(x)
        "tanh": "1-tanh2",  # d/dx tanh(x) = 1 - tanh(x)^2
    }

    def __init__(self, vector: VectorVariable, op: str) -> None:
        if op not in self._NUMPY_FUNCS:
            raise InvalidOperationError(
                operation=op,
                operand_types="VectorVariable",
                reason=f"Unsupported operation. Supported: {list(self._NUMPY_FUNCS.keys())}",
            )
        self.vector = vector
        self.op = op
        self._hash = None

    def evaluate(self, values: Mapping[str, ArrayLike | float]) -> float:
        """Evaluate sum of function values using numpy."""
        vals = np.array([v.evaluate(values) for v in self.vector._variables])
        return float(np.sum(self._NUMPY_FUNCS[self.op](vals)))

    def get_variables(self) -> set[Variable]:
        """Return all variables this expression depends on."""
        return set(self.vector._variables)

    def jacobian_row(self, variables: list[Variable]) -> list[Expression] | None:
        """Return Jacobian row in O(n).

        Returns gradient expressions based on the function's derivative.

        Returns:
            List of expressions, or None if optimization not applicable.
        """
        from optyx.core.expressions import UnaryOp

        my_vars = set(self.vector._variables)
        result: list[Expression] = []

        for var in variables:
            if var in my_vars:
                # Compute derivative based on op
                if self.op == "sin":
                    # d/dx sin(x) = cos(x)
                    result.append(UnaryOp(var, "cos"))
                elif self.op == "cos":
                    # d/dx cos(x) = -sin(x)
                    result.append(BinaryOp(Constant(-1.0), UnaryOp(var, "sin"), "*"))
                elif self.op == "exp":
                    # d/dx exp(x) = exp(x)
                    result.append(UnaryOp(var, "exp"))
                elif self.op == "log":
                    # d/dx log(x) = 1/x
                    result.append(BinaryOp(Constant(1.0), var, "/"))
                elif self.op == "sqrt":
                    # d/dx sqrt(x) = 1/(2*sqrt(x))
                    sqrt_x = UnaryOp(var, "sqrt")
                    two_sqrt = BinaryOp(Constant(2.0), sqrt_x, "*")
                    result.append(BinaryOp(Constant(1.0), two_sqrt, "/"))
                elif self.op == "sinh":
                    # d/dx sinh(x) = cosh(x)
                    result.append(UnaryOp(var, "cosh"))
                elif self.op == "cosh":
                    # d/dx cosh(x) = sinh(x)
                    result.append(UnaryOp(var, "sinh"))
                elif self.op == "tanh":
                    # d/dx tanh(x) = 1 - tanh(x)^2
                    tanh_x = UnaryOp(var, "tanh")
                    tanh_sq = BinaryOp(tanh_x, Constant(2.0), "**")
                    result.append(BinaryOp(Constant(1.0), tanh_sq, "-"))
                elif self.op == "tan":
                    # d/dx tan(x) = 1/cos(x)^2 = sec(x)^2
                    cos_x = UnaryOp(var, "cos")
                    cos_sq = BinaryOp(cos_x, Constant(2.0), "**")
                    result.append(BinaryOp(Constant(1.0), cos_sq, "/"))
                elif self.op == "abs":
                    # d/dx |x| = sign(x) - not smooth, approximate as x/|x|
                    abs_x = UnaryOp(var, "abs")
                    result.append(BinaryOp(var, abs_x, "/"))
                else:
                    # Fallback: return None to use general autodiff
                    return None
            else:
                result.append(Constant(0.0))

        return result

    def __repr__(self) -> str:
        return f"VectorUnarySum({self.op}, {self.vector.name})"


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

    __slots__ = ("_stored_expressions", "size")

    # Tell NumPy to defer to Python's operators
    __array_ufunc__ = None

    def __init__(self, expressions: Sequence[Expression]) -> None:
        if len(expressions) == 0:
            raise EmptyContainerError(
                container_type="VectorExpression",
                operation="initialization",
            )
        self._stored_expressions = list(expressions)
        self.size = len(expressions)

    @property
    def _expressions(self) -> Sequence[Expression]:
        return self._stored_expressions

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

    def sum(self) -> VectorExpressionSum:
        """Sum of all elements.

        Returns:
            VectorExpressionSum expression (scalar).

        Example:
            >>> x = VectorVariable("x", 3)
            >>> y = VectorVariable("y", 3)
            >>> s = (x - y).sum()  # VectorExpressionSum, not nested BinaryOps
        """
        return VectorExpressionSum(self)

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

    def __rsub__(self, other: float | int) -> VectorBinaryOp:
        # other - self: wrap scalar as left, self as right
        return VectorBinaryOp(
            VectorExpression([Constant(float(other))] * self.size),
            self,
            "-",
            self.size,
        )

    def __mul__(self, other: float | int) -> VectorExpression:
        """Scalar multiplication."""
        return _vector_binary_op(self, other, "*")

    def __rmul__(self, other: float | int) -> VectorExpression:
        return _vector_binary_op(self, other, "*")

    def __truediv__(self, other: float | int) -> VectorExpression:
        """Scalar division."""
        return _vector_binary_op(self, other, "/")

    def __rtruediv__(self, other: float | int) -> VectorBinaryOp:
        """Right scalar division."""
        return VectorBinaryOp(
            VectorExpression([Constant(float(other))] * self.size),
            self,
            "/",
            self.size,
        )

    def __neg__(self) -> VectorBinaryOp:
        """Negate all elements: -expr = (-1) * expr."""
        return _vector_binary_op(self, -1, "*")

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

    def between(
        self,
        lb: float | int | np.ndarray | list | VectorExpression | VectorVariable,
        ub: float | int | np.ndarray | list | VectorExpression | VectorVariable,
    ) -> list[Constraint]:
        """Create element-wise range constraints: lb <= self <= ub.

        Returns:
            List of constraints covering both bounds.
        """
        return _vector_constraint(self, lb, ">=") + _vector_constraint(self, ub, "<=")

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
                raise WrongDimensionalityError(
                    context="dot product",
                    expected_ndim=1,
                    got_ndim=arr.ndim,
                )
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


class VectorBinaryOp(VectorExpression):
    """Element-wise binary operation between two vectors as a single node.

    Instead of materializing N individual BinaryOp nodes, this stores the
    operation symbolically and evaluates using a single NumPy call.

    The ``_expressions`` list is created lazily on first access so that code
    which iterates over individual elements (e.g. gradient computation) still
    works, while the common compilation path avoids O(N) materialization.

    Args:
        left: Left operand vector.
        right: Right operand vector (or broadcast scalar).
        op: Binary operation (``+``, ``-``, ``*``, ``/``, ``**``).
        size: Number of elements.

    Example:
        >>> x = VectorVariable("x", 3)
        >>> y = VectorVariable("y", 3)
        >>> z = x + y  # VectorBinaryOp, single node
        >>> z.evaluate({"x[0]": 1, "x[1]": 2, "x[2]": 3, "y[0]": 4, "y[1]": 5, "y[2]": 6})
        [5.0, 7.0, 9.0]
    """

    __slots__ = ("left", "right", "op", "_materialized")

    _NUMPY_OPS: dict[str, np.ufunc] = {
        "+": np.add,
        "-": np.subtract,
        "*": np.multiply,
        "/": np.divide,
        "**": np.power,
    }

    def __init__(
        self,
        left: VectorVariable | VectorExpression,
        right: VectorVariable | VectorExpression | float | int,
        op: str,
        size: int,
    ) -> None:
        # Store vector-level operands for fast numpy compilation
        self.left = left
        self.right = right
        self.op = op
        self.size = size
        # Lazy materialization: _expressions created only when accessed
        self._materialized: list[BinaryOp] | None = None

    @property
    def _expressions(self) -> list[BinaryOp]:  # type: ignore[override]
        """Lazily materialize per-element BinaryOp nodes.

        Only called when code needs individual scalar expressions
        (e.g., gradient fallback, iteration). The compiler fast path
        and evaluate() never trigger this.
        """
        if self._materialized is None:
            left_exprs = _iter_vector_exprs(self.left)
            right_exprs = _iter_vector_exprs(self.right, broadcast_size=self.size)
            op = cast("Literal['+', '-', '*', '/', '**']", self.op)
            self._materialized = [
                BinaryOp(le, re, op) for le, re in zip(left_exprs, right_exprs)
            ]
        return self._materialized

    @_expressions.setter
    def _expressions(self, value: list[BinaryOp]) -> None:
        """Allow direct assignment for compatibility."""
        self._materialized = value

    # -- fast evaluate (no materialization) ----------------------------------

    def evaluate(self, values: Mapping[str, ArrayLike | float]) -> list[float]:
        """Evaluate using numpy — O(1) NumPy calls, no per-element overhead."""
        left_vals = _eval_vector(self.left, values)
        right_vals = _eval_vector(self.right, values)
        result = self._NUMPY_OPS[self.op](left_vals, right_vals)
        return [float(v) for v in result]

    def get_variables(self) -> set[Variable]:
        """Return all variables from both operands."""
        result: set[Variable] = set()
        if isinstance(self.left, VectorVariable):
            result.update(self.left._variables)
        elif isinstance(self.left, VectorExpression) and hasattr(
            self.left, "get_variables"
        ):
            result.update(self.left.get_variables())
        if isinstance(self.right, VectorVariable):
            result.update(self.right._variables)
        elif isinstance(self.right, (VectorVariable, VectorExpression)) and hasattr(
            self.right, "get_variables"
        ):
            result.update(self.right.get_variables())  # type: ignore[union-attr]
        return result

    def __repr__(self) -> str:
        left_name = self.left.name if isinstance(self.left, VectorVariable) else "expr"
        right_name = (
            self.right.name if isinstance(self.right, VectorVariable) else "expr"
        )
        return f"VectorBinaryOp({left_name} {self.op} {right_name}, size={self.size})"


def _iter_vector_exprs(
    vec: VectorVariable | VectorExpression | float | int,
    broadcast_size: int | None = None,
) -> list[Expression]:
    """Extract a list of scalar expressions from a vector operand."""
    if isinstance(vec, VectorVariable):
        return list(vec._variables)
    if isinstance(vec, VectorExpression):
        return list(
            vec._expressions
        )  # triggers lazy materialization for VectorBinaryOp
    if isinstance(vec, (int, float)):
        return [Constant(vec)] * (broadcast_size or 1)
    return [Constant(vec)] * (broadcast_size or 1)


def _eval_vector(
    vec: VectorVariable | VectorExpression | float | int,
    values: Mapping[str, ArrayLike | float],
) -> NDArray[np.floating]:
    """Evaluate a vector operand to a numpy array."""
    if isinstance(vec, VectorVariable):
        return np.array([v.evaluate(values) for v in vec._variables])
    if isinstance(vec, VectorBinaryOp):
        # Recursive numpy evaluation — no materialization
        left_vals = _eval_vector(vec.left, values)
        right_vals = _eval_vector(vec.right, values)
        return VectorBinaryOp._NUMPY_OPS[vec.op](left_vals, right_vals)
    if isinstance(vec, VectorExpression):
        return np.array([e.evaluate(values) for e in vec._expressions])
    # scalar
    return np.array([float(vec)])


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
    lb: float | Sequence[float] | NDArray | None
    ub: float | Sequence[float] | NDArray | None
    domain: DomainType
    _variables: list[Variable]

    def __init__(
        self,
        name: str,
        size: int,
        lb: float | Sequence[float] | NDArray | None = None,
        ub: float | Sequence[float] | NDArray | None = None,
        domain: DomainType = "continuous",
    ) -> None:
        if size <= 0:
            raise InvalidSizeError(
                entity=name,
                size=size,
                reason="must be positive",
            )

        self.name = name
        self.size = size
        self.lb = lb
        self.ub = ub
        self.domain = domain

        # Helper to get bound for index i
        def get_bound(
            b: float | Sequence[float] | NDArray | None, i: int, param_name: str
        ) -> float | None:
            if b is None:
                return None
            if isinstance(b, (int, float, np.number)):
                return float(b)
            if hasattr(b, "__len__") and hasattr(b, "__getitem__"):
                if len(b) != size:
                    raise InvalidSizeError(
                        entity=f"{param_name} for {name}",
                        size=len(b),
                        reason=f"must match vector size {size}",
                    )
                return float(b[i])
            # Fallback
            if hasattr(b, "__float__"):
                return float(b)  # type: ignore
            return None

        # Create individual variables
        self._variables: list[Variable] = []
        for i in range(size):
            val_l = get_bound(lb, i, "lb")
            val_u = get_bound(ub, i, "ub")
            self._variables.append(
                Variable(f"{name}[{i}]", lb=val_l, ub=val_u, domain=domain)
            )

    @overload
    def __getitem__(self, key: int) -> Variable: ...

    @overload
    def __getitem__(self, key: slice) -> VectorVariable: ...

    def __getitem__(
        self, key: int | slice | Sequence[int] | NDArray
    ) -> Variable | VectorVariable:
        """Index or slice the vector.

        Args:
            key: Integer index, slice object, list of indices, or boolean array.

        Returns:
            Single Variable for integer index, VectorVariable for slice or list/array.

        Example:
            >>> x = VectorVariable("x", 10)
            >>> x[0]  # Variable("x[0]")
            >>> x[-1]  # Variable("x[9]")
            >>> x[2:5]  # VectorVariable with 3 elements
            >>> x[[0, 2, 4]]  # VectorVariable with elements 0, 2, 4
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
                lb=self.lb if isinstance(self.lb, (int, float, type(None))) else None,
                ub=self.ub if isinstance(self.ub, (int, float, type(None))) else None,
                domain=self.domain,
            )

        elif isinstance(key, (list, tuple, np.ndarray)):
            # Fancy indexing
            if isinstance(key, (list, tuple)):
                key = np.array(key)

            # Type ignore because we checked type above but type checker might not infer
            indices: NDArray = key  # type: ignore

            if indices.dtype == bool:
                if len(indices) != self.size:
                    raise IndexError(f"Boolean index has wrong length: {len(indices)}")
                indices = np.where(indices)[0]

            selected_vars = []
            for idx in indices:
                # Handle negative indices manually
                i = int(idx)
                if i < 0:
                    i = self.size + i
                if i < 0 or i >= self.size:
                    raise IndexError(f"Index {i} out of range")
                selected_vars.append(self._variables[i])

            if len(selected_vars) == 0:
                raise IndexError("Fancy indexing results in empty VectorVariable")

            return VectorVariable._from_variables(
                name=f"{self.name}[fancy]",
                variables=selected_vars,
                lb=self.lb if isinstance(self.lb, (int, float, type(None))) else None,
                ub=self.ub if isinstance(self.ub, (int, float, type(None))) else None,
                domain=self.domain,
            )

        else:
            # Original error fallback
            raise TypeError(f"Invalid index type: {type(key)}")
            raise InvalidOperationError(
                operation="vector indexing",
                operand_types=(type(key).__name__,),
                suggestion="Use integers for single elements or slices for subvectors.",
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

    @property
    def T(self) -> None:
        """Transpose is not supported for vectors.

        Raises:
            TypeError: Always, with guidance on alternatives.

        Note:
            Unlike NumPy where 1D arrays have a trivial transpose,
            Optyx vectors don't support .T because:

            1. For dot products, use: ``x.dot(y)``
            2. For quadratic forms xᵀQx, use: ``x.dot(Q @ x)``
            3. For linear combinations cᵀx, use: ``c @ x``

        Example:
            >>> x = VectorVariable("x", 3)
            >>> x.T  # Raises InvalidOperationError with helpful message
        """
        raise InvalidOperationError(
            operation="transpose",
            operand_types=("VectorVariable",),
            suggestion="Use x.dot(y) for dot products, x.dot(Q @ x) for quadratic forms, or c @ x for linear combinations.",
        )

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

    def __rsub__(self, other: float | int) -> VectorBinaryOp:
        """Right subtraction: scalar - vector."""
        return VectorBinaryOp(
            VectorExpression([Constant(float(other))] * self.size),
            self,
            "-",
            self.size,
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

    def __rtruediv__(self, other: float | int) -> VectorBinaryOp:
        """Right scalar division: 1 / x."""
        return VectorBinaryOp(
            VectorExpression([Constant(float(other))] * self.size),
            self,
            "/",
            self.size,
        )

    def __neg__(self) -> VectorBinaryOp:
        """Negate all elements: -x."""
        return _vector_binary_op(self, -1, "*")

    def __pow__(self, other: float | int) -> ElementwisePower:
        """Element-wise power: x ** 2.

        Returns ElementwisePower for O(1) numpy-based evaluation.
        """
        return ElementwisePower(self, other)

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

    def between(
        self,
        lb: float | int | np.ndarray | list | VectorExpression | VectorVariable,
        ub: float | int | np.ndarray | list | VectorExpression | VectorVariable,
    ) -> list[Constraint]:
        """Create element-wise range constraints: lb <= self <= ub.

        Returns:
            List of constraints covering both bounds.
        """
        return _vector_constraint(self, lb, ">=") + _vector_constraint(self, ub, "<=")

    def dot(self, other: VectorVariable | VectorExpression) -> DotProduct | Expression:
        """Compute dot product with another vector.

        Special case: if ``other`` is a ``MatrixVectorProduct`` of ``A @ self``,
        this returns a ``QuadraticForm`` for optimized gradient computation.
        That is, ``x.dot(A @ x)`` returns ``QuadraticForm(x, A)`` which has an
        O(1) gradient rule instead of the general O(n) DotProduct gradient.

        Args:
            other: Vector to compute dot product with.

        Returns:
            DotProduct expression (scalar), or QuadraticForm if other is A @ self.

        Example:
            >>> x = VectorVariable("x", 3)
            >>> y = VectorVariable("y", 3)
            >>> d = x.dot(y)
            >>> d.evaluate({"x[0]": 1, "x[1]": 2, "x[2]": 3, "y[0]": 4, "y[1]": 5, "y[2]": 6})
            32.0

            >>> # Quadratic form optimization: x.dot(Q @ x) -> QuadraticForm
            >>> import numpy as np
            >>> Q = np.eye(3)
            >>> qf = x.dot(Q @ x)  # Returns QuadraticForm, not DotProduct
        """
        # Check for quadratic form pattern: x.dot(A @ x)
        from optyx.core.matrices import MatrixVectorProduct, QuadraticForm

        if isinstance(other, MatrixVectorProduct):
            # Check if the MatrixVectorProduct's vector is self
            if isinstance(other.vector, VectorVariable):
                if other.vector is self or other.vector.name == self.name:
                    # This is x.dot(A @ x) - return QuadraticForm for O(1) gradient
                    return QuadraticForm(self, other.matrix)

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
        # Import here to avoid circular import
        from optyx.core.matrices import MatrixVariable

        if isinstance(other, (VectorVariable, VectorExpression)):
            return DotProduct(self, other)
        elif isinstance(other, MatrixVariable):
            raise InvalidOperationError(
                operation="matrix multiplication",
                operand_types=("VectorVariable", "MatrixVariable"),
                suggestion="Use MatrixVariable @ VectorVariable instead.",
            )
        elif isinstance(other, (np.ndarray, list)):
            # VectorVariable @ array is the same as array @ VectorVariable
            arr = np.asarray(other)
            if arr.ndim != 1:
                raise WrongDimensionalityError(
                    context="dot product",
                    expected_ndim=1,
                    got_ndim=arr.ndim,
                )
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
            raise InvalidOperationError(
                operation="norm",
                operand_types=(f"ord={ord}",),
                suggestion="Supported norm orders: 1 (L1/Manhattan) or 2 (L2/Euclidean).",
            )

    def __rmatmul__(self, other: np.ndarray) -> LinearCombination | MatrixVectorProduct:
        """Enable numpy_array @ vector syntax.

        For 1D arrays: returns LinearCombination (dot product with coefficients).
        For 2D arrays: returns MatrixVectorProduct (matrix-vector multiplication).

        Args:
            other: NumPy array (1D for linear combination, 2D for matrix-vector).

        Returns:
            LinearCombination for 1D arrays, MatrixVectorProduct for 2D arrays.

        Example:
            >>> import numpy as np
            >>> # 1D: Linear combination (scalar result)
            >>> c = np.array([0.12, 0.08, 0.10])
            >>> x = VectorVariable("x", 3)
            >>> expr = c @ x  # LinearCombination

            >>> # 2D: Matrix-vector product (vector result)
            >>> Q = np.array([[1, 2], [3, 4]])
            >>> y = VectorVariable("y", 2)
            >>> Qy = Q @ y  # MatrixVectorProduct
            >>> x.dot(Q @ x)  # Quadratic form: x · (Qx) = xᵀQx
        """
        arr = np.asarray(other)
        if arr.ndim == 1:
            return LinearCombination(arr, self)
        elif arr.ndim == 2:
            from optyx.core.matrices import MatrixVectorProduct

            return MatrixVectorProduct(arr, self)
        else:
            raise WrongDimensionalityError(
                context="array @ vector",
                expected_ndim=2,
                got_ndim=arr.ndim,
            )

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
            raise WrongDimensionalityError(
                context="VectorVariable.from_numpy",
                expected_ndim=1,
                got_ndim=array.ndim,
            )
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
            raise WrongDimensionalityError(
                context=f"vector constraint ({sense})",
                expected_ndim=1,
                got_ndim=right_arr.ndim,
            )
        if len(right_arr) != len(left_exprs):
            raise DimensionMismatchError(
                operation=f"vector constraint ({sense})",
                left_shape=len(left_exprs),
                right_shape=len(right_arr),
            )
        # Create constraints with scalar values from array
        return [
            _make_constraint(left_expr, sense, float(val))
            for left_expr, val in zip(left_exprs, right_arr)
        ]
    else:
        raise InvalidOperationError(
            operation=f"vector constraint ({sense})",
            operand_types=(type(right).__name__,),
            suggestion="Use VectorVariable, VectorExpression, scalar, numpy array, or list.",
        )

    # Create element-wise constraints
    return [
        _make_constraint(left_expr, sense, right_expr)
        for left_expr, right_expr in zip(left_exprs, right_exprs)
    ]


def _vector_binary_op(
    left: VectorVariable | VectorExpression,
    right: VectorVariable | VectorExpression | float | int,
    op: Literal["+", "-", "*", "/", "**"],
) -> VectorBinaryOp:
    """Helper for element-wise binary operations on vectors.

    Returns a single VectorBinaryOp node instead of N individual BinaryOps.
    The per-element BinaryOps are materialized lazily only when needed.

    Args:
        left: Left operand (VectorVariable or VectorExpression).
        right: Right operand (vector or scalar).
        op: Operation to perform.

    Returns:
        VectorBinaryOp with element-wise operation.

    Raises:
        DimensionMismatchError: If vector sizes don't match.
    """
    # Determine left size
    if isinstance(left, VectorVariable):
        left_size = left.size
    elif isinstance(left, (VectorExpression, ElementwisePower)):
        left_size = left.size
    else:
        left_size = left.size

    # Validate right operand and determine size
    if isinstance(right, (int, float)):
        pass  # scalar broadcast, always valid
    elif isinstance(right, VectorVariable):
        if right.size != left_size:
            raise DimensionMismatchError(
                operation=f"vector {op}",
                left_shape=left_size,
                right_shape=right.size,
            )
    elif isinstance(right, (VectorExpression, ElementwisePower)):
        if right.size != left_size:
            raise DimensionMismatchError(
                operation=f"vector {op}",
                left_shape=left_size,
                right_shape=right.size,
            )
    elif isinstance(right, (np.ndarray, list)):
        arr = np.asarray(right)
        if arr.ndim != 1:
            raise WrongDimensionalityError(
                context=f"vector {op}",
                expected_ndim=1,
                got_ndim=arr.ndim,
            )
        if len(arr) != left_size:
            raise DimensionMismatchError(
                operation=f"vector {op}",
                left_shape=left_size,
                right_shape=len(arr),
            )
        # Wrap numpy array as a VectorExpression of Constants
        right = VectorExpression([Constant(float(val)) for val in arr])
    else:
        raise InvalidOperationError(
            operation=f"vector {op}",
            operand_types=(type(right).__name__,),
            suggestion="Use VectorVariable, VectorExpression, scalar, numpy array, or list.",
        )

    return VectorBinaryOp(left, right, op, left_size)


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
        raise InvalidOperationError(
            operation="vector_sum",
            operand_types=(type(vector).__name__,),
            suggestion="Use VectorVariable or VectorExpression.",
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
        raise InvalidOperationError(
            operation="norm",
            operand_types=(f"ord={ord}",),
            suggestion="Supported norm orders: 1 (L1/Manhattan) or 2 (L2/Euclidean).",
        )
