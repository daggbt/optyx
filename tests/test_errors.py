"""Tests for the errors module."""

from __future__ import annotations

import pytest

from optyx.core.errors import (
    OptyxError,
    DimensionMismatchError,
    InvalidOperationError,
    BoundsError,
    IndexError,
    EmptyContainerError,
    SolverError,
    InfeasibleError,
    UnboundedError,
    NotSolvedError,
    ParameterError,
    dimension_error,
    # New comprehensive errors
    NoObjectiveError,
    ConstraintError,
    NonLinearError,
    UnknownOperatorError,
    InvalidExpressionError,
    SolverConfigurationError,
    IntegerVariableError,
    SymmetryError,
    SquareMatrixError,
    InvalidSizeError,
    ShapeMismatchError,
    WrongDimensionalityError,
)


class TestOptyxError:
    """Test base exception class."""

    def test_is_exception(self):
        """OptyxError is an Exception."""
        assert issubclass(OptyxError, Exception)

    def test_can_catch(self):
        """Can catch OptyxError."""
        with pytest.raises(OptyxError):
            raise OptyxError("test error")

    def test_message(self):
        """Error message is preserved."""
        err = OptyxError("test message")
        assert str(err) == "test message"


class TestDimensionMismatchError:
    """Test dimension mismatch errors."""

    def test_basic_message(self):
        """Basic error message format."""
        err = DimensionMismatchError("addition", (3,), (5,))

        assert "Dimension mismatch" in str(err)
        assert "addition" in str(err)
        assert "(3,)" in str(err)
        assert "(5,)" in str(err)

    def test_with_int_shapes(self):
        """Handles integer sizes (for vectors)."""
        err = DimensionMismatchError("dot product", 3, 5)

        assert "(3,)" in str(err)
        assert "(5,)" in str(err)

    def test_with_suggestion(self):
        """Includes suggestion when provided."""
        err = DimensionMismatchError(
            "matrix multiply",
            (3, 4),
            (5, 6),
            suggestion="Transpose the right matrix",
        )

        assert "Transpose the right matrix" in str(err)

    def test_attributes(self):
        """Stores attributes correctly."""
        err = DimensionMismatchError("op", (2, 3), (4, 5), "hint")

        assert err.operation == "op"
        assert err.left_shape == (2, 3)
        assert err.right_shape == (4, 5)
        assert err.suggestion == "hint"

    def test_is_value_error(self):
        """Also inherits from ValueError for compatibility."""
        assert issubclass(DimensionMismatchError, ValueError)
        assert issubclass(DimensionMismatchError, OptyxError)


class TestInvalidOperationError:
    """Test invalid operation errors."""

    def test_basic_message(self):
        """Basic error message format."""
        err = InvalidOperationError("matmul", (int, str))

        assert "Invalid operation" in str(err)
        assert "matmul" in str(err)
        assert "int" in str(err)
        assert "str" in str(err)

    def test_single_type(self):
        """Handles single type."""
        err = InvalidOperationError("negate", float)

        assert "float" in str(err)

    def test_with_reason(self):
        """Includes reason when provided."""
        err = InvalidOperationError(
            "dot", (list, list), reason="Lists are not supported"
        )

        assert "Lists are not supported" in str(err)

    def test_with_suggestion(self):
        """Includes suggestion when provided."""
        err = InvalidOperationError(
            "multiply",
            (str, int),
            suggestion="Convert string to number first",
        )

        assert "Try: Convert string to number first" in str(err)

    def test_is_type_error(self):
        """Also inherits from TypeError for compatibility."""
        assert issubclass(InvalidOperationError, TypeError)


class TestBoundsError:
    """Test bounds errors."""

    def test_basic_message(self):
        """Basic error message format."""
        err = BoundsError("x", 10, 5)

        assert "Invalid bounds" in str(err)
        assert "x" in str(err)
        assert "lb=10" in str(err)
        assert "ub=5" in str(err)
        assert "cannot exceed" in str(err)

    def test_with_reason(self):
        """Custom reason overrides default."""
        err = BoundsError("y", 0, 100, reason="Binary variable requires 0/1 bounds")

        assert "Binary variable" in str(err)

    def test_attributes(self):
        """Stores attributes correctly."""
        err = BoundsError("z", -1, 1)

        assert err.variable_name == "z"
        assert err.lower == -1
        assert err.upper == 1


class TestIndexError:
    """Test index errors."""

    def test_vector_message(self):
        """Vector index error message."""
        err = IndexError("x", 10, 5, "vector")

        assert "10" in str(err)
        assert "x" in str(err)
        assert "size 5" in str(err)
        assert "0 to 4" in str(err)

    def test_matrix_message(self):
        """Matrix index error message."""
        err = IndexError("A", (5, 3), (3, 3), "matrix")

        assert "(5, 3)" in str(err)
        assert "A" in str(err)
        assert "shape (3, 3)" in str(err)

    def test_is_builtin_index_error(self):
        """Also inherits from built-in IndexError."""
        # Our IndexError should be catchable with except IndexError
        with pytest.raises(IndexError):
            raise IndexError("x", 5, 3, "vector")


class TestEmptyContainerError:
    """Test empty container errors."""

    def test_vector_message(self):
        """Empty vector error message."""
        err = EmptyContainerError("vector", "slice [5:5]")

        assert "empty vector" in str(err)
        assert "slice [5:5]" in str(err)

    def test_matrix_message(self):
        """Empty matrix error message."""
        err = EmptyContainerError("matrix", "row slice [0:0]")

        assert "empty matrix" in str(err)


class TestSolverError:
    """Test solver errors."""

    def test_basic_message(self):
        """Basic error message."""
        err = SolverError("Solver failed to converge")

        assert "Solver failed to converge" in str(err)

    def test_with_solver_name(self):
        """Includes solver name."""
        err = SolverError("Maximum iterations reached", solver_name="scipy")

        assert "[scipy]" in str(err)
        assert "Maximum iterations" in str(err)

    def test_with_original_error(self):
        """Includes original error."""
        original = ValueError("bad value")
        err = SolverError("Optimization failed", original_error=original)

        assert "Original error" in str(err)
        assert "bad value" in str(err)

    def test_attributes(self):
        """Stores attributes correctly."""
        original = RuntimeError("boom")
        err = SolverError("msg", "cvxpy", original)

        assert err.solver_name == "cvxpy"
        assert err.original_error is original


class TestInfeasibleError:
    """Test infeasibility errors."""

    def test_default_message(self):
        """Default message."""
        err = InfeasibleError()

        assert "infeasible" in str(err).lower()

    def test_with_conflicting_constraints(self):
        """Shows conflicting constraints."""
        err = InfeasibleError(
            "No solution exists",
            conflicting_constraints=["x >= 10", "x <= 5"],
        )

        assert "x >= 10" in str(err)
        assert "x <= 5" in str(err)

    def test_is_solver_error(self):
        """Inherits from SolverError."""
        assert issubclass(InfeasibleError, SolverError)


class TestUnboundedError:
    """Test unboundedness errors."""

    def test_default_message(self):
        """Default message."""
        err = UnboundedError()

        assert "unbounded" in str(err).lower()

    def test_with_direction(self):
        """Shows unbounded direction."""
        err = UnboundedError(
            "Objective can decrease forever",
            unbounded_direction="x → -∞",
        )

        assert "x → -∞" in str(err)


class TestNotSolvedError:
    """Test not-solved errors."""

    def test_default_message(self):
        """Default message."""
        err = NotSolvedError()

        assert "solution" in str(err)
        assert "solve()" in str(err)

    def test_with_attribute(self):
        """Custom attribute name."""
        err = NotSolvedError("objective_value")

        assert "objective_value" in str(err)


class TestParameterError:
    """Test parameter errors."""

    def test_basic_message(self):
        """Basic error message."""
        err = ParameterError("price", "must be positive")

        assert "price" in str(err)
        assert "must be positive" in str(err)

    def test_with_expected_got(self):
        """Shows expected vs got."""
        err = ParameterError(
            "Sigma",
            "shape mismatch",
            expected=(3, 3),
            got=(5, 5),
        )

        assert "(3, 3)" in str(err)
        assert "(5, 5)" in str(err)


class TestDimensionErrorHelper:
    """Test dimension_error convenience function."""

    def test_with_tuple_shapes(self):
        """Works with objects that have .shape."""

        class FakeMatrix:
            shape = (3, 4)

        class FakeVector:
            shape = (5,)

        err = dimension_error("multiply", FakeMatrix(), FakeVector())

        assert "(3, 4)" in str(err)
        assert "(5,)" in str(err)

    def test_with_size_attribute(self):
        """Works with objects that have .size."""

        class FakeVector:
            size = 10

        err = dimension_error("add", FakeVector(), FakeVector())
        assert "10" in str(err)

    def test_with_len(self):
        """Works with objects that have __len__."""
        err = dimension_error("concat", [1, 2, 3], [1, 2, 3, 4, 5])

        assert "3" in str(err)
        assert "5" in str(err)


class TestExceptionHierarchy:
    """Test that exception hierarchy allows proper catching."""

    def test_catch_all_optyx(self):
        """Can catch all Optyx errors with OptyxError."""
        errors = [
            DimensionMismatchError("op", (1,), (2,)),
            InvalidOperationError("op", int),
            BoundsError("x", 1, 0),
            SolverError("failed"),
            InfeasibleError(),
            UnboundedError(),
            NotSolvedError(),
            ParameterError("p", "bad"),
        ]

        for err in errors:
            with pytest.raises(OptyxError):
                raise err

    def test_dimension_as_value_error(self):
        """DimensionMismatchError catchable as ValueError."""
        with pytest.raises(ValueError):
            raise DimensionMismatchError("op", (1,), (2,))

    def test_invalid_op_as_type_error(self):
        """InvalidOperationError catchable as TypeError."""
        with pytest.raises(TypeError):
            raise InvalidOperationError("op", int)


# =============================================================================
# Tests for Problem Formulation Errors
# =============================================================================


class TestNoObjectiveError:
    """Test NoObjectiveError for missing objectives."""

    def test_default_message(self):
        """Default error message."""
        err = NoObjectiveError()
        assert "No objective set" in str(err)
        assert "minimize()" in str(err) or "maximize()" in str(err)

    def test_custom_message(self):
        """Custom message and suggestion."""
        err = NoObjectiveError(
            message="Objective required",
            suggestion="Define an objective function",
        )
        assert "Objective required" in str(err)
        assert "Define an objective function" in str(err)

    def test_is_value_error(self):
        """NoObjectiveError is a ValueError."""
        with pytest.raises(ValueError):
            raise NoObjectiveError()


class TestConstraintError:
    """Test ConstraintError for invalid constraints."""

    def test_basic_message(self):
        """Basic error message."""
        err = ConstraintError("Constraint must be linear")
        assert "Invalid constraint" in str(err)
        assert "linear" in str(err)

    def test_with_expression(self):
        """Error with constraint expression."""
        err = ConstraintError("Non-linear term detected", constraint_expr="x**2 <= 1")
        assert "x**2 <= 1" in str(err)
        assert "Non-linear" in str(err)

    def test_with_type(self):
        """Error with constraint type."""
        err = ConstraintError(
            "Unknown constraint type",
            constraint_type="!=",
        )
        assert "Unknown constraint type" in str(err)

    def test_is_value_error(self):
        """ConstraintError is a ValueError."""
        with pytest.raises(ValueError):
            raise ConstraintError("bad constraint")


class TestNonLinearError:
    """Test NonLinearError for linearity violations."""

    def test_basic_message(self):
        """Basic error message."""
        err = NonLinearError("LP solver objective")
        assert "requires a linear expression" in str(err)
        assert "LP solver" in str(err)

    def test_with_expression(self):
        """Error with expression string."""
        err = NonLinearError("LP objective", expression="x**2 + y")
        assert "x**2 + y" in str(err)

    def test_with_suggestion(self):
        """Error with suggestion."""
        err = NonLinearError(
            "LP solver",
            suggestion="Use a nonlinear solver like SLSQP",
        )
        assert "SLSQP" in str(err)

    def test_is_value_error(self):
        """NonLinearError is a ValueError."""
        with pytest.raises(ValueError):
            raise NonLinearError("context")


# =============================================================================
# Tests for Expression Errors
# =============================================================================


class TestUnknownOperatorError:
    """Test UnknownOperatorError for invalid operators."""

    def test_basic_message(self):
        """Basic error message."""
        err = UnknownOperatorError("%%%")
        assert "Unknown operator" in str(err)
        assert "%%%" in str(err)

    def test_with_context(self):
        """Error with context."""
        err = UnknownOperatorError("@@@", context="gradient computation")
        assert "@@@" in str(err)
        assert "gradient computation" in str(err)

    def test_is_value_error(self):
        """UnknownOperatorError is a ValueError."""
        with pytest.raises(ValueError):
            raise UnknownOperatorError("op")


class TestInvalidExpressionError:
    """Test InvalidExpressionError for unknown types."""

    def test_basic_message(self):
        """Basic error message."""
        err = InvalidExpressionError(dict)
        assert "Unknown expression type" in str(err)
        assert "dict" in str(err)

    def test_with_context(self):
        """Error with context."""
        err = InvalidExpressionError(list, context="autodiff")
        assert "autodiff" in str(err)

    def test_with_suggestion(self):
        """Error with suggestion."""
        err = InvalidExpressionError(
            str,
            suggestion="Wrap constants with Constant()",
        )
        assert "Wrap constants" in str(err)

    def test_is_type_error(self):
        """InvalidExpressionError is a TypeError."""
        with pytest.raises(TypeError):
            raise InvalidExpressionError(int)


# =============================================================================
# Tests for Solver Configuration Errors
# =============================================================================


class TestSolverConfigurationError:
    """Test SolverConfigurationError for bad solver settings."""

    def test_basic_message(self):
        """Basic error message."""
        err = SolverConfigurationError(
            "Cannot handle quadratic constraints",
            solver_name="linprog",
        )
        assert "linprog" in str(err)
        assert "quadratic" in str(err)

    def test_with_feature(self):
        """Error with problem feature."""
        err = SolverConfigurationError(
            "Unsupported feature",
            solver_name="scipy",
            problem_feature="integer variables",
        )
        assert "integer variables" in str(err)

    def test_with_suggestion(self):
        """Error with suggestion."""
        err = SolverConfigurationError(
            "Feature not supported",
            solver_name="linprog",
            suggestion="Try CVXPY instead",
        )
        assert "CVXPY" in str(err)

    def test_inherits_solver_error(self):
        """SolverConfigurationError is a SolverError."""
        err = SolverConfigurationError("msg", "solver")
        assert isinstance(err, SolverError)


class TestIntegerVariableError:
    """Test IntegerVariableError for integer/binary var issues."""

    def test_basic_message(self):
        """Basic error message."""
        err = IntegerVariableError("scipy")
        assert "scipy" in str(err)
        assert "integer" in str(err) or "binary" in str(err)

    def test_with_variable_names(self):
        """Error with variable names."""
        err = IntegerVariableError("linprog", variable_names=["x", "y"])
        assert "x" in str(err)
        assert "y" in str(err)

    def test_has_suggestion(self):
        """Error includes suggestion for MIP solver."""
        err = IntegerVariableError("scipy")
        assert "MIP" in str(err) or "CBC" in str(err) or "Gurobi" in str(err)

    def test_inherits_solver_config(self):
        """IntegerVariableError is a SolverConfigurationError."""
        err = IntegerVariableError("solver")
        assert isinstance(err, SolverConfigurationError)


# =============================================================================
# Tests for Matrix-Specific Errors
# =============================================================================


class TestSymmetryError:
    """Test SymmetryError for symmetry violations."""

    def test_basic_message(self):
        """Basic error message."""
        err = SymmetryError("PSD constraint")
        assert "symmetric" in str(err)
        assert "PSD constraint" in str(err)

    def test_with_matrix_name(self):
        """Error with matrix name."""
        err = SymmetryError("covariance update", matrix_name="Sigma")
        assert "Sigma" in str(err)
        assert "symmetric" in str(err)

    def test_is_value_error(self):
        """SymmetryError is a ValueError."""
        with pytest.raises(ValueError):
            raise SymmetryError("operation")


class TestSquareMatrixError:
    """Test SquareMatrixError for non-square matrices."""

    def test_basic_message(self):
        """Basic error message."""
        err = SquareMatrixError("trace", (3, 5))
        assert "trace" in str(err)
        assert "square" in str(err)
        assert "(3, 5)" in str(err)

    def test_is_value_error(self):
        """SquareMatrixError is a ValueError."""
        with pytest.raises(ValueError):
            raise SquareMatrixError("det", (2, 3))


# =============================================================================
# Tests for Size and Shape Errors
# =============================================================================


class TestInvalidSizeError:
    """Test InvalidSizeError for bad dimensions."""

    def test_basic_message(self):
        """Basic error message."""
        err = InvalidSizeError("VectorVariable", -5)
        assert "VectorVariable" in str(err)
        assert "-5" in str(err)
        assert "positive" in str(err)

    def test_with_reason(self):
        """Error with custom reason."""
        err = InvalidSizeError("matrix", 0, reason="cannot be zero")
        assert "cannot be zero" in str(err)

    def test_tuple_size(self):
        """Error with tuple size."""
        err = InvalidSizeError("MatrixVariable", (0, 5))
        assert "(0, 5)" in str(err)

    def test_is_value_error(self):
        """InvalidSizeError is a ValueError."""
        with pytest.raises(ValueError):
            raise InvalidSizeError("vec", -1)


class TestShapeMismatchError:
    """Test ShapeMismatchError for array shape issues."""

    def test_basic_message(self):
        """Basic error message."""
        err = ShapeMismatchError("parameter update", (3, 3), (4, 4))
        assert "parameter update" in str(err)
        assert "(3, 3)" in str(err)
        assert "(4, 4)" in str(err)

    def test_is_value_error(self):
        """ShapeMismatchError is a ValueError."""
        with pytest.raises(ValueError):
            raise ShapeMismatchError("op", (1,), (2,))


class TestWrongDimensionalityError:
    """Test WrongDimensionalityError for ndim issues."""

    def test_basic_message(self):
        """Basic error message."""
        err = WrongDimensionalityError("MatrixParameter", 2, 3)
        assert "MatrixParameter" in str(err)
        assert "2D" in str(err)
        assert "3D" in str(err)

    def test_friendly_names(self):
        """Uses friendly dimension names."""
        err = WrongDimensionalityError("VectorParameter", 1, 2)
        assert "vector" in str(err).lower() or "1D" in str(err)

    def test_is_value_error(self):
        """WrongDimensionalityError is a ValueError."""
        with pytest.raises(ValueError):
            raise WrongDimensionalityError("context", 1, 2)


# =============================================================================
# Comprehensive Hierarchy Tests
# =============================================================================


class TestComprehensiveHierarchy:
    """Test the full exception hierarchy."""

    def test_all_inherit_from_optyx_error(self):
        """All custom errors inherit from OptyxError."""
        errors = [
            DimensionMismatchError("op", (1,), (2,)),
            InvalidOperationError("op", int),
            BoundsError("x", 1, 0),
            SolverError("failed"),
            InfeasibleError(),
            UnboundedError(),
            NotSolvedError(),
            ParameterError("p", "bad"),
            NoObjectiveError(),
            ConstraintError("bad"),
            NonLinearError("context"),
            UnknownOperatorError("op"),
            InvalidExpressionError(int),
            SolverConfigurationError("msg", "solver"),
            IntegerVariableError("solver"),
            SymmetryError("context"),
            SquareMatrixError("op", (2, 3)),
            InvalidSizeError("entity", -1),
            ShapeMismatchError("ctx", (1,), (2,)),
            WrongDimensionalityError("ctx", 1, 2),
        ]

        for err in errors:
            assert isinstance(
                err, OptyxError
            ), f"{type(err).__name__} should inherit from OptyxError"

    def test_solver_errors_hierarchy(self):
        """Solver errors have correct hierarchy."""
        assert issubclass(InfeasibleError, SolverError)
        assert issubclass(UnboundedError, SolverError)
        assert issubclass(SolverConfigurationError, SolverError)
        assert issubclass(IntegerVariableError, SolverConfigurationError)

    def test_value_errors(self):
        """Value-like errors are catchable as ValueError."""
        value_errors = [
            DimensionMismatchError("op", (1,), (2,)),
            BoundsError("x", 1, 0),
            ParameterError("p", "bad"),
            NoObjectiveError(),
            ConstraintError("bad"),
            NonLinearError("context"),
            UnknownOperatorError("op"),
            SymmetryError("context"),
            SquareMatrixError("op", (2, 3)),
            InvalidSizeError("entity", -1),
            ShapeMismatchError("ctx", (1,), (2,)),
            WrongDimensionalityError("ctx", 1, 2),
        ]

        for err in value_errors:
            assert isinstance(
                err, ValueError
            ), f"{type(err).__name__} should be a ValueError"

    def test_type_errors(self):
        """Type-like errors are catchable as TypeError."""
        type_errors = [
            InvalidOperationError("op", int),
            InvalidExpressionError(int),
        ]

        for err in type_errors:
            assert isinstance(
                err, TypeError
            ), f"{type(err).__name__} should be a TypeError"
