"""Custom exception classes for Optyx.

This module provides clear, actionable error messages for common issues
in optimization problem formulation. All errors inherit from OptyxError
for easy catching of library-specific exceptions.

Example:
    >>> try:
    ...     x = VectorVariable("x", 3)
    ...     y = VectorVariable("y", 5)
    ...     z = x + y  # Size mismatch!
    ... except DimensionMismatchError as e:
    ...     print(e)  # Clear message with sizes
"""

from __future__ import annotations

from typing import Any


class OptyxError(Exception):
    """Base class for all Optyx exceptions.

    Catch this to handle any library-specific error:

        try:
            problem.solve()
        except OptyxError as e:
            print(f"Optimization error: {e}")
    """

    pass


class DimensionMismatchError(OptyxError, ValueError):
    """Raised when vector/matrix dimensions don't match for an operation.

    This error provides clear information about what dimensions were
    expected and what was received, making debugging easier.

    Attributes:
        operation: The operation that failed (e.g., "addition", "dot product").
        left_shape: Shape/size of the left operand.
        right_shape: Shape/size of the right operand.
        suggestion: Optional hint for fixing the issue.
    """

    def __init__(
        self,
        operation: str,
        left_shape: tuple[int, ...] | int,
        right_shape: tuple[int, ...] | int,
        suggestion: str | None = None,
    ) -> None:
        """Create a dimension mismatch error.

        Args:
            operation: Name of the operation (e.g., "vector addition").
            left_shape: Shape of left operand (tuple or int for vectors).
            right_shape: Shape of right operand.
            suggestion: Optional hint for fixing the error.
        """
        self.operation = operation
        self.left_shape = left_shape
        self.right_shape = right_shape
        self.suggestion = suggestion

        # Format shapes nicely
        left_str = self._format_shape(left_shape)
        right_str = self._format_shape(right_shape)

        msg = (
            f"Dimension mismatch in {operation}: "
            f"left operand has shape {left_str}, "
            f"right operand has shape {right_str}"
        )

        if suggestion:
            msg += f". {suggestion}"

        super().__init__(msg)

    @staticmethod
    def _format_shape(shape: tuple[int, ...] | int) -> str:
        """Format shape for display."""
        if isinstance(shape, int):
            return f"({shape},)"
        return str(shape)


class InvalidOperationError(OptyxError, TypeError):
    """Raised when an operation is invalid for the given types.

    This error explains what operation was attempted and why it's
    not supported, with suggestions for alternatives when possible.

    Attributes:
        operation: The operation that was attempted.
        operand_types: Types of the operands involved.
        reason: Why the operation is invalid.
        suggestion: Optional alternative approach.
    """

    def __init__(
        self,
        operation: str,
        operand_types: tuple[type, ...] | type,
        reason: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Create an invalid operation error.

        Args:
            operation: Name of the operation attempted.
            operand_types: Type(s) of operand(s) involved.
            reason: Why the operation is invalid.
            suggestion: Alternative approach to try.
        """
        self.operation = operation
        self.operand_types = operand_types
        self.reason = reason
        self.suggestion = suggestion

        # Format types nicely
        if isinstance(operand_types, tuple):
            type_str = ", ".join(t.__name__ for t in operand_types)
        else:
            type_str = operand_types.__name__

        msg = f"Invalid operation '{operation}' for type(s): {type_str}"

        if reason:
            msg += f". {reason}"
        if suggestion:
            msg += f" Try: {suggestion}"

        super().__init__(msg)


class BoundsError(OptyxError, ValueError):
    """Raised when variable bounds are invalid.

    Common issues include:
    - Lower bound greater than upper bound
    - Infinite bounds where finite required
    - Bounds incompatible with variable type
    """

    def __init__(
        self,
        variable_name: str,
        lower: float,
        upper: float,
        reason: str | None = None,
    ) -> None:
        """Create a bounds error.

        Args:
            variable_name: Name of the variable with invalid bounds.
            lower: The lower bound value.
            upper: The upper bound value.
            reason: Additional context about why bounds are invalid.
        """
        self.variable_name = variable_name
        self.lower = lower
        self.upper = upper
        self.reason = reason

        msg = f"Invalid bounds for variable '{variable_name}': lb={lower}, ub={upper}"

        if reason:
            msg += f". {reason}"
        elif lower > upper:
            msg += ". Lower bound cannot exceed upper bound"

        super().__init__(msg)


# Capture Python's builtin IndexError before we shadow it
_BuiltinIndexError = IndexError


class IndexError(OptyxError, _BuiltinIndexError):
    """Raised when indexing a VectorVariable or MatrixVariable is out of bounds.

    Provides clear information about the valid index range and what
    index was attempted.
    """

    def __init__(
        self,
        container_name: str,
        index: int | tuple[int, ...],
        valid_range: tuple[int, ...] | int,
        container_type: str = "vector",
    ) -> None:
        """Create an index error.

        Args:
            container_name: Name of the vector/matrix being indexed.
            index: The invalid index that was used.
            valid_range: Valid range (size for vector, shape for matrix).
            container_type: "vector" or "matrix".
        """
        self.container_name = container_name
        self.index = index
        self.valid_range = valid_range
        self.container_type = container_type

        if container_type == "vector":
            if isinstance(valid_range, int):
                range_str = f"0 to {valid_range - 1}"
            else:
                range_str = f"0 to {valid_range[0] - 1}"
            msg = (
                f"Index {index} out of bounds for {container_type} "
                f"'{container_name}' with size {valid_range}. "
                f"Valid indices: {range_str}"
            )
        else:  # matrix
            msg = (
                f"Index {index} out of bounds for {container_type} "
                f"'{container_name}' with shape {valid_range}"
            )

        super().__init__(msg)


class EmptyContainerError(OptyxError, ValueError):
    """Raised when a vector or matrix would be empty.

    Operations like slicing can result in empty containers, which
    are not supported in optimization problems.
    """

    def __init__(
        self,
        container_type: str,
        operation: str,
    ) -> None:
        """Create an empty container error.

        Args:
            container_type: "vector" or "matrix".
            operation: The operation that caused the empty result.
        """
        self.container_type = container_type
        self.operation = operation

        msg = (
            f"Operation '{operation}' would result in an empty {container_type}. "
            f"Vectors and matrices must have at least one element."
        )

        super().__init__(msg)


class SolverError(OptyxError):
    """Raised when the solver encounters an error.

    This wraps solver-specific errors with additional context about
    the optimization problem.
    """

    def __init__(
        self,
        message: str,
        solver_name: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        """Create a solver error.

        Args:
            message: Description of what went wrong.
            solver_name: Name of the solver (e.g., "scipy", "cvxpy").
            original_error: The underlying exception, if any.
        """
        self.solver_name = solver_name
        self.original_error = original_error

        if solver_name:
            msg = f"[{solver_name}] {message}"
        else:
            msg = message

        if original_error:
            msg += f" (Original error: {original_error})"

        super().__init__(msg)


class InfeasibleError(SolverError):
    """Raised when the optimization problem has no feasible solution.

    The constraints cannot all be satisfied simultaneously.
    """

    def __init__(
        self,
        message: str = "Problem is infeasible",
        solver_name: str | None = None,
        conflicting_constraints: list[str] | None = None,
    ) -> None:
        """Create an infeasibility error.

        Args:
            message: Description of the infeasibility.
            solver_name: Name of the solver.
            conflicting_constraints: Names of potentially conflicting constraints.
        """
        self.conflicting_constraints = conflicting_constraints

        if conflicting_constraints:
            message += f". Potentially conflicting: {conflicting_constraints}"

        super().__init__(message, solver_name)


class UnboundedError(SolverError):
    """Raised when the optimization problem is unbounded.

    The objective can be improved indefinitely.
    """

    def __init__(
        self,
        message: str = "Problem is unbounded",
        solver_name: str | None = None,
        unbounded_direction: str | None = None,
    ) -> None:
        """Create an unboundedness error.

        Args:
            message: Description of the unboundedness.
            solver_name: Name of the solver.
            unbounded_direction: Which variable(s) can grow without bound.
        """
        self.unbounded_direction = unbounded_direction

        if unbounded_direction:
            message += f". Unbounded direction: {unbounded_direction}"

        super().__init__(message, solver_name)


class NotSolvedError(OptyxError):
    """Raised when accessing solution before solving the problem."""

    def __init__(self, attribute: str = "solution") -> None:
        """Create a not-solved error.

        Args:
            attribute: What was accessed (e.g., "objective_value", "x").
        """
        self.attribute = attribute

        msg = (
            f"Cannot access '{attribute}' before solving the problem. "
            f"Call problem.solve() first."
        )

        super().__init__(msg)


class ParameterError(OptyxError, ValueError):
    """Raised when parameter values are invalid.

    This includes shape mismatches when updating parameters,
    or invalid initial values.
    """

    def __init__(
        self,
        parameter_name: str,
        message: str,
        expected: Any = None,
        got: Any = None,
    ) -> None:
        """Create a parameter error.

        Args:
            parameter_name: Name of the parameter.
            message: Description of what's wrong.
            expected: What was expected (optional).
            got: What was received (optional).
        """
        self.parameter_name = parameter_name
        self.expected = expected
        self.got = got

        msg = f"Parameter '{parameter_name}': {message}"

        if expected is not None and got is not None:
            msg += f" (expected {expected}, got {got})"

        super().__init__(msg)


# =============================================================================
# Problem Formulation Errors
# =============================================================================


class NoObjectiveError(OptyxError, ValueError):
    """Raised when trying to solve a problem with no objective function.

    Example:
        >>> problem = Problem()
        >>> problem.subject_to(x >= 0)
        >>> problem.solve()  # Raises NoObjectiveError
    """

    def __init__(
        self,
        message: str = "No objective set",
        suggestion: str = "Call minimize() or maximize() first",
    ) -> None:
        """Create a no-objective error.

        Args:
            message: Description of the error.
            suggestion: How to fix the problem.
        """
        self.suggestion = suggestion

        msg = f"{message}. {suggestion}"
        super().__init__(msg)


class ConstraintError(OptyxError, ValueError):
    """Raised when a constraint is invalid or malformed.

    Common issues:
    - Invalid constraint type (not ==, <=, >=)
    - Constraint on non-comparable expressions
    - Missing constraint bounds
    """

    def __init__(
        self,
        message: str,
        constraint_expr: str | None = None,
        constraint_type: str | None = None,
    ) -> None:
        """Create a constraint error.

        Args:
            message: Description of what's wrong.
            constraint_expr: String representation of the constraint.
            constraint_type: The constraint relation (==, <=, >=).
        """
        self.constraint_expr = constraint_expr
        self.constraint_type = constraint_type

        if constraint_expr:
            msg = f"Invalid constraint '{constraint_expr}': {message}"
        else:
            msg = f"Invalid constraint: {message}"

        super().__init__(msg)


class NonLinearError(OptyxError, ValueError):
    """Raised when an operation requires linear expressions but gets nonlinear.

    This commonly occurs when:
    - Using LP solver with quadratic objectives
    - Extracting coefficients from nonlinear expressions
    - Using linear-only constraint handlers
    """

    def __init__(
        self,
        context: str,
        expression: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Create a nonlinearity error.

        Args:
            context: Where linearity is required (e.g., "LP objective").
            expression: The nonlinear expression (optional).
            suggestion: How to work around this.
        """
        self.context = context
        self.expression = expression
        self.suggestion = suggestion

        msg = f"{context} requires a linear expression"

        if expression:
            msg += f", but got: {expression}"

        if suggestion:
            msg += f". {suggestion}"

        super().__init__(msg)


# =============================================================================
# Expression and Compilation Errors
# =============================================================================


class UnknownOperatorError(OptyxError, ValueError):
    """Raised when an unknown operator is encountered during compilation.

    This typically indicates a bug in the expression system or an
    unsupported operation.
    """

    def __init__(
        self,
        operator: str,
        context: str = "expression compilation",
    ) -> None:
        """Create an unknown operator error.

        Args:
            operator: The unknown operator symbol/name.
            context: Where the error occurred.
        """
        self.operator = operator
        self.context = context

        msg = f"Unknown operator '{operator}' in {context}"
        super().__init__(msg)


class InvalidExpressionError(OptyxError, TypeError):
    """Raised when an expression type is not recognized.

    This can occur when mixing optyx expressions with incompatible types.
    """

    def __init__(
        self,
        expr_type: type,
        context: str = "expression evaluation",
        suggestion: str | None = None,
    ) -> None:
        """Create an invalid expression error.

        Args:
            expr_type: The unrecognized type.
            context: Where the error occurred.
            suggestion: How to fix this.
        """
        self.expr_type = expr_type
        self.context = context
        self.suggestion = suggestion

        msg = f"Unknown expression type '{expr_type.__name__}' in {context}"

        if suggestion:
            msg += f". {suggestion}"

        super().__init__(msg)


# =============================================================================
# Solver Configuration Errors
# =============================================================================


class SolverConfigurationError(SolverError):
    """Raised when the solver is misconfigured for the problem type.

    Common issues:
    - Integer variables with continuous-only solver
    - Nonlinear constraints with LP solver
    - Missing required solver options
    """

    def __init__(
        self,
        message: str,
        solver_name: str,
        problem_feature: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Create a solver configuration error.

        Args:
            message: Description of the configuration issue.
            solver_name: Name of the solver.
            problem_feature: The unsupported feature (e.g., "integer variables").
            suggestion: How to fix this.
        """
        self.problem_feature = problem_feature
        self.suggestion = suggestion

        full_msg = message

        if problem_feature:
            full_msg += f". Unsupported feature: {problem_feature}"

        if suggestion:
            full_msg += f". {suggestion}"

        super().__init__(full_msg, solver_name)


class IntegerVariableError(SolverConfigurationError):
    """Raised when integer/binary variables are used with a continuous solver."""

    def __init__(
        self,
        solver_name: str,
        variable_names: list[str] | None = None,
    ) -> None:
        """Create an integer variable error.

        Args:
            solver_name: Name of the continuous-only solver.
            variable_names: Names of the integer/binary variables.
        """
        self.variable_names = variable_names

        message = f"Solver '{solver_name}' does not support integer/binary variables"

        if variable_names:
            message += f": {variable_names}"

        super().__init__(
            message,
            solver_name,
            problem_feature="integer/binary variables",
            suggestion="Use a MIP solver (e.g., CBC, GLPK, Gurobi)",
        )


# =============================================================================
# Matrix-Specific Errors
# =============================================================================


class SymmetryError(OptyxError, ValueError):
    """Raised when symmetry constraints are violated.

    This occurs when:
    - A non-symmetric matrix is used where symmetric is required
    - Updating a symmetric parameter with non-symmetric values
    """

    def __init__(
        self,
        context: str,
        matrix_name: str | None = None,
    ) -> None:
        """Create a symmetry error.

        Args:
            context: What required symmetry (e.g., "covariance matrix update").
            matrix_name: Name of the matrix, if known.
        """
        self.context = context
        self.matrix_name = matrix_name

        if matrix_name:
            msg = f"Matrix '{matrix_name}' must be symmetric for {context}"
        else:
            msg = f"Matrix must be symmetric for {context}"

        super().__init__(msg)


class SquareMatrixError(OptyxError, ValueError):
    """Raised when a square matrix is required but not provided.

    Operations like trace, determinant, and eigenvalue decomposition
    require square matrices.
    """

    def __init__(
        self,
        operation: str,
        shape: tuple[int, int],
    ) -> None:
        """Create a square matrix error.

        Args:
            operation: The operation that requires a square matrix.
            shape: The actual (non-square) shape.
        """
        self.operation = operation
        self.shape = shape

        msg = f"Operation '{operation}' requires a square matrix, got shape {shape}"
        super().__init__(msg)


# =============================================================================
# Size and Shape Errors
# =============================================================================


class InvalidSizeError(OptyxError, ValueError):
    """Raised when a size/dimension is invalid.

    Common issues:
    - Zero or negative size for vector/matrix
    - Size doesn't match existing data
    """

    def __init__(
        self,
        entity: str,
        size: int | tuple[int, ...],
        reason: str = "must be positive",
    ) -> None:
        """Create an invalid size error.

        Args:
            entity: What has the invalid size (e.g., "VectorVariable").
            size: The invalid size value.
            reason: Why it's invalid.
        """
        self.entity = entity
        self.size = size
        self.reason = reason

        msg = f"Invalid size for {entity}: {size} ({reason})"
        super().__init__(msg)


class ShapeMismatchError(OptyxError, ValueError):
    """Raised when array shapes don't match expectations.

    This is more specific than DimensionMismatchError, focused on
    parameter updates and array operations.
    """

    def __init__(
        self,
        context: str,
        expected: tuple[int, ...],
        got: tuple[int, ...],
    ) -> None:
        """Create a shape mismatch error.

        Args:
            context: What operation had the mismatch.
            expected: The expected shape.
            got: The actual shape.
        """
        self.context = context
        self.expected = expected
        self.got = got

        msg = f"Shape mismatch in {context}: expected {expected}, got {got}"
        super().__init__(msg)


class WrongDimensionalityError(OptyxError, ValueError):
    """Raised when array has wrong number of dimensions.

    Example: Passing a 3D array to MatrixParameter (expects 2D).
    """

    def __init__(
        self,
        context: str,
        expected_ndim: int,
        got_ndim: int,
    ) -> None:
        """Create a wrong dimensionality error.

        Args:
            context: What received the wrong array.
            expected_ndim: Expected number of dimensions.
            got_ndim: Actual number of dimensions.
        """
        self.context = context
        self.expected_ndim = expected_ndim
        self.got_ndim = got_ndim

        dim_names = {1: "1D (vector)", 2: "2D (matrix)", 3: "3D (tensor)"}
        expected_str = dim_names.get(expected_ndim, f"{expected_ndim}D")
        got_str = dim_names.get(got_ndim, f"{got_ndim}D")

        msg = f"{context} requires {expected_str} array, got {got_str}"
        super().__init__(msg)


# Convenience functions for common error patterns


def dimension_error(
    operation: str,
    left: Any,
    right: Any,
    suggestion: str | None = None,
) -> DimensionMismatchError:
    """Create a DimensionMismatchError from operands.

    Automatically extracts shapes from common types.

    Args:
        operation: Name of the operation.
        left: Left operand (VectorVariable, MatrixVariable, array, etc.).
        right: Right operand.
        suggestion: Optional fix suggestion.

    Returns:
        A DimensionMismatchError with appropriate shapes.
    """
    left_shape = _get_shape(left)
    right_shape = _get_shape(right)
    return DimensionMismatchError(operation, left_shape, right_shape, suggestion)


def _get_shape(obj: Any) -> tuple[int, ...] | int:
    """Extract shape from various types."""
    if hasattr(obj, "shape"):
        return obj.shape
    if hasattr(obj, "size"):
        return obj.size
    if hasattr(obj, "__len__"):
        return len(obj)
    return (1,)


__all__ = [
    # Base
    "OptyxError",
    # Dimension/Shape errors
    "DimensionMismatchError",
    "ShapeMismatchError",
    "WrongDimensionalityError",
    "InvalidSizeError",
    # Type/Operation errors
    "InvalidOperationError",
    "UnknownOperatorError",
    "InvalidExpressionError",
    # Variable errors
    "BoundsError",
    "IndexError",
    "EmptyContainerError",
    # Parameter errors
    "ParameterError",
    # Problem formulation errors
    "NoObjectiveError",
    "ConstraintError",
    "NonLinearError",
    # Matrix-specific errors
    "SymmetryError",
    "SquareMatrixError",
    # Solver errors
    "SolverError",
    "SolverConfigurationError",
    "IntegerVariableError",
    "InfeasibleError",
    "UnboundedError",
    "NotSolvedError",
    # Convenience functions
    "dimension_error",
]
