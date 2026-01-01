"""Constraint classes for optimization problems.

Provides constraint representation with natural Python syntax:
    x + y <= 10
    x**2 + y**2 >= 1
    x - y == 0
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from optyx.core.expressions import Expression, Variable


@dataclass(frozen=True)
class Constraint:
    """Represents an optimization constraint.

    A constraint has the form: expr sense 0
    where sense is one of: <=, >=, ==

    The expression is normalized so RHS is always 0.
    For example, x + y <= 5 becomes (x + y - 5) <= 0.

    Attributes:
        expr: The constraint expression (LHS - RHS, so constraint is expr sense 0)
        sense: The constraint type: "<=", ">=", or "=="
        name: Optional name for the constraint

    Example:
        >>> x = Variable("x")
        >>> c = x + 2 <= 10  # Creates Constraint with expr = x + 2 - 10
        >>> c.is_satisfied({"x": 5.0})
        True
    """

    expr: Expression
    sense: str  # "<=", ">=", "=="
    name: str | None = None

    def __post_init__(self):
        if self.sense not in ("<=", ">=", "=="):
            raise ValueError(
                f"Invalid constraint sense: {self.sense}. Must be <=, >=, or =="
            )

    def evaluate(self, point: Mapping[str, float]) -> float:
        """Evaluate the constraint expression at a point.

        Returns the value of expr at the point.
        For a satisfied constraint:
            - <= constraint: value <= 0
            - >= constraint: value >= 0
            - == constraint: value == 0
        """
        result = self.expr.evaluate(point)
        # Ensure we return a scalar float
        if isinstance(result, np.ndarray):
            return float(result.item())
        return float(result)

    def violation(self, point: dict[str, float]) -> float:
        """Compute constraint violation (0 if satisfied).

        Args:
            point: Dictionary of variable values.

        Returns:
            Non-negative violation amount. Zero means satisfied.
        """
        value = self.evaluate(point)

        if self.sense == "<=":
            return max(0.0, value)
        elif self.sense == ">=":
            return max(0.0, -value)
        else:  # ==
            return abs(value)

    def is_satisfied(self, point: dict[str, float], tol: float = 1e-8) -> bool:
        """Check if constraint is satisfied at a point.

        Args:
            point: Dictionary of variable values.
            tol: Tolerance for satisfaction check.

        Returns:
            True if constraint is satisfied within tolerance.
        """
        return self.violation(point) <= tol

    def get_variables(self) -> set[Variable]:
        """Get all variables in this constraint."""
        return self.expr.get_variables()

    def __repr__(self) -> str:
        name_str = f"'{self.name}': " if self.name else ""
        return f"Constraint({name_str}{self.expr} {self.sense} 0)"


def _make_constraint(lhs: Expression, sense: str, rhs) -> Constraint:
    """Helper to create a constraint from lhs sense rhs.

    Normalizes to (lhs - rhs) sense 0 form.
    """
    from optyx.core.expressions import Constant, Expression

    # Convert rhs to expression if needed
    if isinstance(rhs, (int, float)):
        rhs = Constant(rhs)

    # Normalize: expr sense 0
    if isinstance(rhs, Expression):
        expr = lhs - rhs
    else:
        expr = lhs - Constant(float(rhs))

    return Constraint(expr=expr, sense=sense)
