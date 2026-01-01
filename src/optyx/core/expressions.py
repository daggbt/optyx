"""Symbolic expression system with operator overloading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal
from typing import Mapping

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray
    from optyx.constraints import Constraint


class Expression(ABC):
    """Abstract base class for all symbolic expressions.

    Expressions form a tree structure that can be evaluated given variable values.
    All arithmetic operators are overloaded to build expression trees automatically.

    Attributes:
        _hash: Cached hash value for the expression.
        _degree: Cached polynomial degree (None if not computed, -1 if non-polynomial).
    """

    __slots__ = ("_hash", "_degree")

    @abstractmethod
    def evaluate(
        self, values: Mapping[str, ArrayLike | float]
    ) -> NDArray[np.floating] | float:
        """Evaluate the expression given variable values.

        Args:
            values: Dictionary mapping variable names to their values.

        Returns:
            The numerical result of evaluating the expression.
        """
        pass

    @abstractmethod
    def get_variables(self) -> set[Variable]:
        """Return all variables this expression depends on."""
        pass

    def __hash__(self) -> int:
        if not hasattr(self, "_hash") or self._hash is None:
            self._hash = id(self)
        return self._hash

    def __eq__(self, other: object) -> bool:
        # Identity equality for expressions (not value equality)
        return self is other

    # Arithmetic operators - build expression trees

    def __add__(self, other: Expression | float | int) -> BinaryOp:
        return BinaryOp(self, _ensure_expr(other), "+")

    def __radd__(self, other: float | int) -> BinaryOp:
        return BinaryOp(_ensure_expr(other), self, "+")

    def __sub__(self, other: Expression | float | int) -> BinaryOp:
        return BinaryOp(self, _ensure_expr(other), "-")

    def __rsub__(self, other: float | int) -> BinaryOp:
        return BinaryOp(_ensure_expr(other), self, "-")

    def __mul__(self, other: Expression | float | int) -> BinaryOp:
        return BinaryOp(self, _ensure_expr(other), "*")

    def __rmul__(self, other: float | int) -> BinaryOp:
        return BinaryOp(_ensure_expr(other), self, "*")

    def __truediv__(self, other: Expression | float | int) -> BinaryOp:
        return BinaryOp(self, _ensure_expr(other), "/")

    def __rtruediv__(self, other: float | int) -> BinaryOp:
        return BinaryOp(_ensure_expr(other), self, "/")

    def __pow__(self, other: Expression | float | int) -> BinaryOp:
        return BinaryOp(self, _ensure_expr(other), "**")

    def __rpow__(self, other: float | int) -> BinaryOp:
        return BinaryOp(_ensure_expr(other), self, "**")

    def __neg__(self) -> UnaryOp:
        return UnaryOp(self, "neg")

    def __pos__(self) -> Expression:
        return self

    # Comparison operators - create constraints

    def __le__(self, other: Expression | float | int) -> Constraint:
        """Create a <= constraint: self <= other."""
        from optyx.constraints import _make_constraint

        return _make_constraint(self, "<=", other)

    def __ge__(self, other: Expression | float | int) -> Constraint:
        """Create a >= constraint: self >= other."""
        from optyx.constraints import _make_constraint

        return _make_constraint(self, ">=", other)

    def eq(self, other: Expression | float | int) -> Constraint:
        """Create an == constraint: self == other.

        Note: We use eq() instead of __eq__ because __eq__ is used
        for object identity comparison which is needed for sets/dicts.
        """
        from optyx.constraints import _make_constraint

        return _make_constraint(self, "==", other)

    def constraint_eq(self, other: Expression | float | int) -> Constraint:
        """Create an == constraint: self == other.

        Note: We use constraint_eq() instead of __eq__ because __eq__ is used
        for object identity comparison which is needed for sets/dicts.

        .. deprecated::
            Use :meth:`eq` instead. This method is kept for backwards compatibility.
        """
        return self.eq(other)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(... )"

    @property
    def degree(self) -> int | None:
        """Polynomial degree of the expression (cached).

        Returns:
            - integer degree >= 0 if the expression is a polynomial
            - None if the expression is non-polynomial (e.g., sin, exp,
              division by variable, non-integer powers)

        The result is computed once and cached for all subsequent calls.
        """
        # Check if cached (_degree attr exists and is not uninitialized)
        # We use a sentinel: not hasattr means uninitialized
        # -1 means cached as non-polynomial (return None)
        # >= 0 means cached polynomial degree
        if hasattr(self, "_degree") and self._degree is not None:
            return None if self._degree == -1 else self._degree

        # Compute and cache
        from optyx.analysis import compute_degree

        result = compute_degree(self)
        self._degree = result if result is not None else -1
        return result

    def is_linear(self) -> bool:
        """Check if this expression is linear (degree <= 1).

        Returns:
            True if the expression is constant or linear in variables.

        Uses cached degree computation for performance.
        """
        deg = self.degree
        return deg is not None and deg <= 1


class Constant(Expression):
    """A constant numerical value in an expression.

    Wraps scalars or numpy arrays as expression nodes.

    Example:
        >>> c = Constant(5.0)
        >>> c.evaluate({})  # No variables needed
        5.0
    """

    __slots__ = ("value",)

    def __init__(self, value: float | int | ArrayLike) -> None:
        self.value = np.asarray(value) if not isinstance(value, (int, float)) else value

    def evaluate(
        self, values: Mapping[str, ArrayLike | float]
    ) -> NDArray[np.floating] | float:
        return self.value

    def get_variables(self) -> set[Variable]:
        return set()

    def __repr__(self) -> str:
        return f"Constant({self.value})"


class Variable(Expression):
    """A decision variable in an optimization problem.

    Args:
        name: Unique identifier for this variable.
        lb: Lower bound (None for unbounded).
        ub: Upper bound (None for unbounded).
        domain: Variable type - 'continuous', 'integer', or 'binary'.

    Example:
        >>> x = Variable("x", lb=0, ub=10)
        >>> y = Variable("y", domain="binary")
        >>> x.evaluate({"x": 5.0})
        5.0
    """

    __slots__ = ("name", "lb", "ub", "domain")

    def __init__(
        self,
        name: str,
        lb: float | None = None,
        ub: float | None = None,
        domain: Literal["continuous", "integer", "binary"] = "continuous",
    ) -> None:
        self.name = name
        self.lb = lb
        self.ub = ub
        self.domain = domain

        # Binary variables have implicit bounds
        if domain == "binary":
            self.lb = 0.0
            self.ub = 1.0

    def evaluate(
        self, values: Mapping[str, ArrayLike | float]
    ) -> NDArray[np.floating] | float:
        if self.name not in values:
            raise KeyError(f"Variable '{self.name}' not found in values")
        value = values[self.name]
        # Return as-is, can be array or scalar
        return value  # type: ignore[return-value]

    def get_variables(self) -> set[Variable]:
        return {self}

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Variable):
            return self.name == other.name
        return False

    def __repr__(self) -> str:
        bounds = ""
        if self.lb is not None or self.ub is not None:
            bounds = f", lb={self.lb}, ub={self.ub}"
        domain_str = "" if self.domain == "continuous" else f", domain='{self.domain}'"
        return f"Variable('{self.name}'{bounds}{domain_str})"


class BinaryOp(Expression):
    """A binary operation between two expressions.

    Supported operators: +, -, *, /, **
    """

    __slots__ = ("left", "right", "op")

    # Operator dispatch table for evaluation
    _OPS = {
        "+": np.add,
        "-": np.subtract,
        "*": np.multiply,
        "/": np.divide,
        "**": np.power,
    }

    def __init__(
        self,
        left: Expression,
        right: Expression,
        op: Literal["+", "-", "*", "/", "**"],
    ) -> None:
        self.left = left
        self.right = right
        self.op = op

    def evaluate(
        self, values: Mapping[str, ArrayLike | float]
    ) -> NDArray[np.floating] | float:
        left_val = self.left.evaluate(values)
        right_val = self.right.evaluate(values)
        return self._OPS[self.op](left_val, right_val)

    def get_variables(self) -> set[Variable]:
        return self.left.get_variables() | self.right.get_variables()

    def __repr__(self) -> str:
        return f"({self.left!r} {self.op} {self.right!r})"


class UnaryOp(Expression):
    """A unary operation on an expression.

    Supported operators: neg, abs, and transcendental functions.
    """

    __slots__ = ("operand", "op", "_numpy_func")

    # Operator dispatch table
    _OPS: dict[str, np.ufunc] = {
        "neg": np.negative,
        "abs": np.abs,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "log2": np.log2,
        "log10": np.log10,
        "sqrt": np.sqrt,
        "tanh": np.tanh,
        "sinh": np.sinh,
        "cosh": np.cosh,
        "asin": np.arcsin,
        "acos": np.arccos,
        "atan": np.arctan,
        "asinh": np.arcsinh,
        "acosh": np.arccosh,
        "atanh": np.arctanh,
    }

    def __init__(self, operand: Expression, op: str) -> None:
        if op not in self._OPS:
            raise ValueError(f"Unknown unary operator: {op}")
        self.operand = operand
        self.op = op
        self._numpy_func = self._OPS[op]

    def evaluate(
        self, values: Mapping[str, ArrayLike | float]
    ) -> NDArray[np.floating] | float:
        operand_val = self.operand.evaluate(values)
        return self._numpy_func(operand_val)

    def get_variables(self) -> set[Variable]:
        return self.operand.get_variables()

    def __repr__(self) -> str:
        return f"{self.op}({self.operand!r})"


def _ensure_expr(value: Expression | float | int | ArrayLike) -> Expression:
    """Convert a value to an Expression if it isn't one already."""
    if isinstance(value, Expression):
        return value
    return Constant(value)


# =============================================================================
# Iterative Variable Extraction (for deep expression trees)
# =============================================================================

# Recursion threshold - use iterative for trees deeper than this
_RECURSION_THRESHOLD = 400


def _estimate_tree_depth(expr: Expression) -> int:
    """Estimate the depth of an expression tree.

    Uses a fast heuristic that follows the left spine of the tree,
    which catches the common case of left-associative chains.
    """
    depth = 0
    current = expr
    while True:
        if isinstance(current, (Constant, Variable)):
            break
        elif isinstance(current, BinaryOp):
            depth += 1
            current = current.left
        elif isinstance(current, UnaryOp):
            depth += 1
            current = current.operand
        else:
            # Vector/matrix expressions - check for deep nesting
            from optyx.core.vectors import LinearCombination, VectorSum, DotProduct

            if isinstance(current, (LinearCombination, VectorSum)):
                break  # These don't recurse deeply
            elif isinstance(current, DotProduct):
                depth += 1
                current = current.left
            else:
                break
    return depth


def get_all_variables(expr: Expression) -> set[Variable]:
    """Extract all variables from an expression, using iterative method for deep trees.

    This function handles arbitrarily deep expression trees without hitting
    Python's recursion limit. For shallow trees, it delegates to the expression's
    get_variables() method. For deep trees, it uses an explicit stack.

    Args:
        expr: The expression to extract variables from.

    Returns:
        Set of all Variable objects in the expression tree.
    """
    depth = _estimate_tree_depth(expr)
    if depth < _RECURSION_THRESHOLD:
        return expr.get_variables()
    return _get_variables_iterative(expr)


def _get_variables_iterative(expr: Expression) -> set[Variable]:
    """Extract variables from expression using explicit stack.

    Handles deep expression trees that would cause RecursionError.
    """
    from optyx.core.vectors import (
        LinearCombination,
        VectorSum,
        DotProduct,
        L2Norm,
        L1Norm,
    )

    variables: set[Variable] = set()
    stack: list[Expression] = [expr]
    seen: set[int] = set()

    while stack:
        node = stack.pop()
        node_id = id(node)

        # Avoid processing the same node twice
        if node_id in seen:
            continue
        seen.add(node_id)

        # Leaf: Variable
        if isinstance(node, Variable):
            variables.add(node)
            continue

        # Leaf: Constant
        if isinstance(node, Constant):
            continue

        # Vector expressions - use their O(1) get_variables methods
        if isinstance(node, LinearCombination):
            # LinearCombination stores variables directly
            variables.update(node.get_variables())
            continue
        if isinstance(node, VectorSum):
            variables.update(node.get_variables())
            continue
        if isinstance(node, DotProduct):
            # DotProduct has get_variables(), use it directly
            variables.update(node.get_variables())
            continue
        if isinstance(node, (L2Norm, L1Norm)):
            variables.update(node.get_variables())
            continue

        # Binary operation
        if isinstance(node, BinaryOp):
            stack.append(node.left)
            stack.append(node.right)
            continue

        # Unary operation
        if isinstance(node, UnaryOp):
            stack.append(node.operand)
            continue

        # Fallback: call get_variables (might recurse for custom expressions)
        try:
            variables.update(node.get_variables())
        except RecursionError:
            # If recursion fails, we can't process this node
            pass

    return variables
