"""Tests for MatrixVariable."""

import numpy as np
import pytest

from optyx.core.errors import (
    DimensionMismatchError,
    InvalidOperationError,
    InvalidSizeError,
    SquareMatrixError,
    WrongDimensionalityError,
)
from optyx.core.matrices import (
    MatrixVariable,
    MatrixVectorProduct,
    QuadraticForm,
    FrobeniusNorm,
    matmul,
    quadratic_form,
    trace,
    diag,
    diag_matrix,
    frobenius_norm,
)
from optyx.core.vectors import VectorVariable
from optyx.core.expressions import Variable, Expression


class TestMatrixVariableCreation:
    """Tests for MatrixVariable creation."""

    def test_basic_creation(self):
        """MatrixVariable creates the specified number of variables."""
        A = MatrixVariable("A", 3, 4)
        assert A.rows == 3
        assert A.cols == 4
        assert A.shape == (3, 4)
        assert A.name == "A"

    def test_square_matrix(self):
        """Square matrix works."""
        A = MatrixVariable("A", 5, 5)
        assert A.shape == (5, 5)

    def test_large_matrix(self):
        """MatrixVariable can handle larger sizes."""
        A = MatrixVariable("A", 10, 10)
        assert len(A.get_variables()) == 100

    def test_element_naming(self):
        """Elements are named with bracket notation."""
        A = MatrixVariable("A", 3, 4)
        assert A[0, 0].name == "A[0,0]"
        assert A[1, 2].name == "A[1,2]"
        assert A[2, 3].name == "A[2,3]"

    def test_zero_rows_raises(self):
        """Rows must be positive."""
        with pytest.raises(InvalidSizeError, match="positive"):
            MatrixVariable("A", 0, 5)

    def test_zero_cols_raises(self):
        """Cols must be positive."""
        with pytest.raises(InvalidSizeError, match="positive"):
            MatrixVariable("A", 5, 0)

    def test_negative_size_raises(self):
        """Negative size raises InvalidSizeError."""
        with pytest.raises(InvalidSizeError, match="positive"):
            MatrixVariable("A", -3, 4)


class TestMatrixVariableBounds:
    """Tests for bounds propagation."""

    def test_lower_bound_propagates(self):
        """Lower bound applies to all elements."""
        A = MatrixVariable("A", 2, 2, lb=0)
        assert A[0, 0].lb == 0
        assert A[1, 1].lb == 0

    def test_upper_bound_propagates(self):
        """Upper bound applies to all elements."""
        A = MatrixVariable("A", 2, 2, ub=100)
        assert A[0, 0].ub == 100
        assert A[1, 1].ub == 100

    def test_both_bounds(self):
        """Both bounds work together."""
        A = MatrixVariable("A", 2, 2, lb=0, ub=1)
        assert A[0, 0].lb == 0
        assert A[0, 0].ub == 1


class TestMatrixVariableDomain:
    """Tests for domain types."""

    def test_integer_domain(self):
        """Integer domain propagates to elements."""
        A = MatrixVariable("A", 2, 2, domain="integer")
        assert A[0, 0].domain == "integer"
        assert A.domain == "integer"

    def test_binary_domain(self):
        """Binary domain propagates to elements."""
        A = MatrixVariable("A", 2, 2, domain="binary")
        assert A[0, 0].domain == "binary"

    def test_continuous_domain_default(self):
        """Default domain is continuous."""
        A = MatrixVariable("A", 2, 2)
        assert A.domain == "continuous"


class TestMatrixVariableIndexing:
    """Tests for 2D indexing."""

    def test_single_element(self):
        """A[i, j] returns a Variable."""
        A = MatrixVariable("A", 3, 4)
        elem = A[1, 2]
        assert isinstance(elem, Variable)
        assert elem.name == "A[1,2]"

    def test_negative_row_index(self):
        """Negative row index works."""
        A = MatrixVariable("A", 3, 4)
        assert A[-1, 0].name == "A[2,0]"

    def test_negative_col_index(self):
        """Negative column index works."""
        A = MatrixVariable("A", 3, 4)
        assert A[0, -1].name == "A[0,3]"

    def test_row_out_of_range(self):
        """Out of range row raises IndexError."""
        A = MatrixVariable("A", 3, 4)
        with pytest.raises(IndexError, match="Row index"):
            _ = A[5, 0]

    def test_col_out_of_range(self):
        """Out of range column raises IndexError."""
        A = MatrixVariable("A", 3, 4)
        with pytest.raises(IndexError, match="Column index"):
            _ = A[0, 10]


class TestMatrixVariableRowSlicing:
    """Tests for row slicing A[i, :]."""

    def test_row_slice_returns_vector(self):
        """A[i, :] returns VectorVariable."""
        A = MatrixVariable("A", 3, 4)
        row = A[1, :]
        assert isinstance(row, VectorVariable)
        assert len(row) == 4

    def test_row_slice_correct_variables(self):
        """Row slice contains correct variables."""
        A = MatrixVariable("A", 3, 4)
        row = A[1, :]
        assert row[0].name == "A[1,0]"
        assert row[3].name == "A[1,3]"

    def test_partial_row_slice(self):
        """A[i, j1:j2] returns partial row."""
        A = MatrixVariable("A", 3, 4)
        partial = A[1, 1:3]
        assert isinstance(partial, VectorVariable)
        assert len(partial) == 2
        assert partial[0].name == "A[1,1]"
        assert partial[1].name == "A[1,2]"


class TestMatrixVariableColSlicing:
    """Tests for column slicing A[:, j]."""

    def test_col_slice_returns_vector(self):
        """A[:, j] returns VectorVariable."""
        A = MatrixVariable("A", 3, 4)
        col = A[:, 2]
        assert isinstance(col, VectorVariable)
        assert len(col) == 3

    def test_col_slice_correct_variables(self):
        """Column slice contains correct variables."""
        A = MatrixVariable("A", 3, 4)
        col = A[:, 2]
        assert col[0].name == "A[0,2]"
        assert col[2].name == "A[2,2]"

    def test_partial_col_slice(self):
        """A[i1:i2, j] returns partial column."""
        A = MatrixVariable("A", 3, 4)
        partial = A[0:2, 1]
        assert isinstance(partial, VectorVariable)
        assert len(partial) == 2
        assert partial[0].name == "A[0,1]"
        assert partial[1].name == "A[1,1]"


class TestMatrixVariableSubmatrix:
    """Tests for submatrix slicing A[i1:i2, j1:j2]."""

    def test_submatrix_returns_matrix(self):
        """A[i1:i2, j1:j2] returns MatrixVariable."""
        A = MatrixVariable("A", 4, 5)
        sub = A[1:3, 2:4]
        assert isinstance(sub, MatrixVariable)
        assert sub.shape == (2, 2)

    def test_submatrix_correct_variables(self):
        """Submatrix contains correct variables."""
        A = MatrixVariable("A", 4, 5)
        sub = A[1:3, 2:4]
        assert sub[0, 0].name == "A[1,2]"
        assert sub[1, 1].name == "A[2,3]"


class TestMatrixVariableTranspose:
    """Tests for transpose."""

    def test_transpose_shape(self):
        """A.T swaps rows and cols."""
        A = MatrixVariable("A", 3, 4)
        assert A.T.shape == (4, 3)

    def test_transpose_indexing(self):
        """A.T[i, j] == A[j, i]."""
        A = MatrixVariable("A", 3, 4)
        # A[1, 2] should be same variable as A.T[2, 1]
        assert A[1, 2] is A.T[2, 1]

    def test_double_transpose(self):
        """A.T.T has same shape as A."""
        A = MatrixVariable("A", 3, 4)
        assert A.T.T.shape == (3, 4)

    def test_transpose_row_is_col(self):
        """Row of A.T is column of A."""
        A = MatrixVariable("A", 3, 4)
        # A.T[0, :] should be same as A[:, 0]
        at_row = A.T[0, :]
        a_col = A[:, 0]
        assert len(at_row) == len(a_col)
        for i in range(len(at_row)):
            assert at_row[i] is a_col[i]


class TestSymmetricMatrix:
    """Tests for symmetric matrices."""

    def test_symmetric_must_be_square(self):
        """Symmetric matrix must be square."""
        with pytest.raises(SquareMatrixError, match="square"):
            MatrixVariable("S", 3, 4, symmetric=True)

    def test_symmetric_shares_variables(self):
        """S[i, j] and S[j, i] are the same variable."""
        S = MatrixVariable("S", 3, 3, symmetric=True)
        assert S[0, 1] is S[1, 0]
        assert S[0, 2] is S[2, 0]
        assert S[1, 2] is S[2, 1]

    def test_symmetric_diagonal_unique(self):
        """Diagonal elements are unique."""
        S = MatrixVariable("S", 3, 3, symmetric=True)
        assert S[0, 0] is not S[1, 1]
        assert S[1, 1] is not S[2, 2]

    def test_symmetric_fewer_variables(self):
        """Symmetric 3x3 has 6 unique variables, not 9."""
        S = MatrixVariable("S", 3, 3, symmetric=True)
        # Upper triangle + diagonal = 3 + 2 + 1 = 6
        assert len(S.get_variables()) == 6

    def test_symmetric_5x5_variable_count(self):
        """Symmetric 5x5 has 15 unique variables."""
        S = MatrixVariable("S", 5, 5, symmetric=True)
        # 5 + 4 + 3 + 2 + 1 = 15
        assert len(S.get_variables()) == 15


class TestMatrixVariableIteration:
    """Tests for iteration."""

    def test_iter_yields_rows(self):
        """Iterating yields rows as VectorVariables."""
        A = MatrixVariable("A", 3, 4)
        rows = list(A)
        assert len(rows) == 3
        for row in rows:
            assert isinstance(row, VectorVariable)
            assert len(row) == 4

    def test_len_returns_rows(self):
        """len(A) returns number of rows."""
        A = MatrixVariable("A", 3, 4)
        assert len(A) == 3


class TestMatrixVariableGetVariables:
    """Tests for get_variables."""

    def test_get_variables_count(self):
        """get_variables returns all variables."""
        A = MatrixVariable("A", 3, 4)
        vars_list = A.get_variables()
        assert len(vars_list) == 12

    def test_get_variables_order(self):
        """Variables are in row-major order."""
        A = MatrixVariable("A", 2, 3)
        vars_list = A.get_variables()
        expected_names = [
            "A[0,0]",
            "A[0,1]",
            "A[0,2]",
            "A[1,0]",
            "A[1,1]",
            "A[1,2]",
        ]
        assert [v.name for v in vars_list] == expected_names


class TestMatrixVariableToNumpy:
    """Tests for to_numpy method."""

    def test_to_numpy_basic(self):
        """to_numpy extracts solution as array."""
        A = MatrixVariable("A", 2, 2)
        solution = {
            "A[0,0]": 1.0,
            "A[0,1]": 2.0,
            "A[1,0]": 3.0,
            "A[1,1]": 4.0,
        }
        result = A.to_numpy(solution)
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(result, expected)

    def test_to_numpy_rectangular(self):
        """to_numpy works for non-square matrices."""
        A = MatrixVariable("A", 2, 3)
        solution = {
            "A[0,0]": 1.0,
            "A[0,1]": 2.0,
            "A[0,2]": 3.0,
            "A[1,0]": 4.0,
            "A[1,1]": 5.0,
            "A[1,2]": 6.0,
        }
        result = A.to_numpy(solution)
        assert result.shape == (2, 3)
        np.testing.assert_array_equal(result, [[1, 2, 3], [4, 5, 6]])


class TestMatrixVariableRepr:
    """Tests for __repr__."""

    def test_basic_repr(self):
        """Basic matrix has readable repr."""
        A = MatrixVariable("A", 3, 4)
        r = repr(A)
        assert "MatrixVariable" in r
        assert "'A'" in r
        assert "3" in r
        assert "4" in r

    def test_symmetric_repr(self):
        """Symmetric matrix shows in repr."""
        S = MatrixVariable("S", 3, 3, symmetric=True)
        assert "symmetric=True" in repr(S)

    def test_bounds_repr(self):
        """Bounds show in repr."""
        A = MatrixVariable("A", 2, 2, lb=0, ub=1)
        r = repr(A)
        assert "lb=0" in r
        assert "ub=1" in r


# =============================================================================
# Matrix-Vector Multiplication Tests
# =============================================================================


class TestMatrixVectorProduct:
    """Tests for matrix-vector multiplication."""

    def test_basic_matmul(self):
        """A @ x produces MatrixVectorProduct."""
        A = np.array([[1, 2], [3, 4]])
        x = VectorVariable("x", 2)
        result = matmul(A, x)
        assert isinstance(result, MatrixVectorProduct)
        assert result.size == 2

    def test_matmul_evaluate(self):
        """Matrix-vector product evaluates correctly."""
        A = np.array([[1, 2], [3, 4]])
        x = VectorVariable("x", 2)
        result = matmul(A, x)
        values = {"x[0]": 1, "x[1]": 2}
        output = result.evaluate(values)
        # [1*1+2*2, 3*1+4*2] = [5, 11]
        assert output == [5.0, 11.0]

    def test_matmul_non_square(self):
        """Matrix-vector product works with non-square matrix."""
        A = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3
        x = VectorVariable("x", 3)
        result = matmul(A, x)
        assert result.size == 2
        values = {"x[0]": 1, "x[1]": 1, "x[2]": 1}
        output = result.evaluate(values)
        # [1+2+3, 4+5+6] = [6, 15]
        assert output == [6.0, 15.0]

    def test_matmul_size_mismatch(self):
        """Size mismatch raises DimensionMismatchError."""
        A = np.array([[1, 2], [3, 4]])  # 2x2
        x = VectorVariable("x", 3)  # 3 elements
        with pytest.raises(DimensionMismatchError, match="mismatch"):
            matmul(A, x)

    def test_matmul_1d_matrix_raises(self):
        """1D array raises WrongDimensionalityError."""
        A = np.array([1, 2, 3])
        x = VectorVariable("x", 3)
        with pytest.raises(WrongDimensionalityError, match="2"):
            matmul(A, x)

    def test_matmul_get_variables(self):
        """MatrixVectorProduct tracks correct variables."""
        A = np.array([[1, 2], [3, 4]])
        x = VectorVariable("x", 2)
        result = matmul(A, x)
        vars_set = result.get_variables()
        assert len(vars_set) == 2
        names = {v.name for v in vars_set}
        assert names == {"x[0]", "x[1]"}

    def test_matmul_indexing(self):
        """Can index into MatrixVectorProduct result."""
        A = np.array([[1, 2], [3, 4]])
        x = VectorVariable("x", 2)
        result = matmul(A, x)
        assert len(result) == 2
        # Each element is a LinearCombination
        values = {"x[0]": 1, "x[1]": 2}
        assert result[0].evaluate(values) == 5.0
        assert result[1].evaluate(values) == 11.0

    def test_matmul_repr(self):
        """MatrixVectorProduct has readable repr."""
        A = np.array([[1, 2], [3, 4]])
        x = VectorVariable("x", 2)
        result = matmul(A, x)
        r = repr(result)
        assert "MatrixVectorProduct" in r
        assert "(2, 2)" in r
        assert "x" in r

    def test_matmul_with_identity(self):
        """Identity matrix returns original vector values."""
        eye = np.eye(3)
        x = VectorVariable("x", 3)
        result = matmul(eye, x)
        values = {"x[0]": 7, "x[1]": 8, "x[2]": 9}
        output = result.evaluate(values)
        assert output == [7.0, 8.0, 9.0]


# =============================================================================
# Quadratic Form Tests
# =============================================================================


class TestQuadraticForm:
    """Tests for quadratic form x' @ Q @ x."""

    def test_basic_quadratic_form(self):
        """quadratic_form creates QuadraticForm expression."""
        Q = np.array([[1, 0], [0, 1]])  # Identity
        x = VectorVariable("x", 2)
        qf = quadratic_form(x, Q)
        assert isinstance(qf, QuadraticForm)

    def test_quadratic_form_identity(self):
        """x' @ I @ x = sum of squares."""
        Q = np.eye(3)
        x = VectorVariable("x", 3)
        qf = quadratic_form(x, Q)
        values = {"x[0]": 1, "x[1]": 2, "x[2]": 3}
        result = qf.evaluate(values)
        # 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14
        assert result == 14.0

    def test_quadratic_form_symmetric(self):
        """Quadratic form with symmetric matrix."""
        Q = np.array([[1, 0.5], [0.5, 2]])
        x = VectorVariable("x", 2)
        qf = quadratic_form(x, Q)
        values = {"x[0]": 1, "x[1]": 1}
        result = qf.evaluate(values)
        # x'Qx = 1*1*1 + 1*0.5*1 + 1*0.5*1 + 1*2*1 = 1 + 0.5 + 0.5 + 2 = 4
        assert result == 4.0

    def test_quadratic_form_portfolio_variance(self):
        """Portfolio variance: w' @ Σ @ w."""
        # Covariance matrix
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        w = VectorVariable("w", 2, lb=0, ub=1)
        variance = quadratic_form(w, cov)
        # Equal weights
        values = {"w[0]": 0.5, "w[1]": 0.5}
        result = variance.evaluate(values)
        # 0.5*0.5*0.04 + 0.5*0.5*0.01 + 0.5*0.5*0.01 + 0.5*0.5*0.09
        # = 0.01 + 0.0025 + 0.0025 + 0.0225 = 0.0375
        assert abs(result - 0.0375) < 1e-10

    def test_quadratic_form_non_square_raises(self):
        """Non-square matrix raises SquareMatrixError."""
        Q = np.array([[1, 2, 3], [4, 5, 6]])  # 2x3
        x = VectorVariable("x", 2)
        with pytest.raises(SquareMatrixError, match="square"):
            quadratic_form(x, Q)

    def test_quadratic_form_size_mismatch(self):
        """Size mismatch raises DimensionMismatchError."""
        Q = np.array([[1, 0], [0, 1]])  # 2x2
        x = VectorVariable("x", 3)  # 3 elements
        with pytest.raises(DimensionMismatchError, match="mismatch"):
            quadratic_form(x, Q)

    def test_quadratic_form_get_variables(self):
        """QuadraticForm tracks correct variables."""
        Q = np.eye(2)
        x = VectorVariable("x", 2)
        qf = quadratic_form(x, Q)
        vars_set = qf.get_variables()
        assert len(vars_set) == 2
        names = {v.name for v in vars_set}
        assert names == {"x[0]", "x[1]"}

    def test_quadratic_form_repr(self):
        """QuadraticForm has readable repr."""
        Q = np.eye(3)
        x = VectorVariable("x", 3)
        qf = quadratic_form(x, Q)
        r = repr(qf)
        assert "QuadraticForm" in r
        assert "x" in r
        assert "(3, 3)" in r

    def test_quadratic_form_zero_matrix(self):
        """Zero matrix gives zero result."""
        Q = np.zeros((2, 2))
        x = VectorVariable("x", 2)
        qf = quadratic_form(x, Q)
        values = {"x[0]": 100, "x[1]": 200}
        assert qf.evaluate(values) == 0.0


# =============================================================================
# Trace Tests
# =============================================================================


class TestTrace:
    """Tests for trace function."""

    def test_trace_matrix_variable(self):
        """trace of MatrixVariable returns Expression."""
        A = MatrixVariable("A", 3, 3)
        tr = trace(A)
        assert isinstance(tr, Expression)

    def test_trace_evaluates_correctly(self):
        """trace sums diagonal elements."""
        A = MatrixVariable("A", 3, 3)
        tr = trace(A)
        values = {
            "A[0,0]": 1,
            "A[0,1]": 2,
            "A[0,2]": 3,
            "A[1,0]": 4,
            "A[1,1]": 5,
            "A[1,2]": 6,
            "A[2,0]": 7,
            "A[2,1]": 8,
            "A[2,2]": 9,
        }
        result = tr.evaluate(values)
        # 1 + 5 + 9 = 15
        assert result == 15.0

    def test_trace_2x2(self):
        """trace of 2x2 matrix."""
        A = MatrixVariable("A", 2, 2)
        tr = trace(A)
        values = {"A[0,0]": 10, "A[0,1]": 20, "A[1,0]": 30, "A[1,1]": 40}
        assert tr.evaluate(values) == 50.0  # 10 + 40

    def test_trace_non_square_raises(self):
        """trace of non-square matrix raises SquareMatrixError."""
        A = MatrixVariable("A", 2, 3)
        with pytest.raises(SquareMatrixError, match="square"):
            trace(A)

    def test_trace_numpy_array(self):
        """trace of numpy array returns Constant."""
        A = np.array([[1, 2], [3, 4]])
        tr = trace(A)
        # Should return Constant(5)
        assert tr.evaluate({}) == 5.0

    def test_trace_symmetric(self):
        """trace of symmetric matrix works."""
        S = MatrixVariable("S", 3, 3, symmetric=True)
        tr = trace(S)
        values = {
            "S[0,0]": 1,
            "S[0,1]": 2,
            "S[0,2]": 3,
            "S[1,1]": 4,
            "S[1,2]": 5,
            "S[2,2]": 6,
        }
        result = tr.evaluate(values)
        # 1 + 4 + 6 = 11
        assert result == 11.0

    def test_trace_get_variables(self):
        """trace tracks only diagonal variables."""
        A = MatrixVariable("A", 2, 2)
        tr = trace(A)
        vars_set = tr.get_variables()
        names = {v.name for v in vars_set}
        # trace only involves diagonal elements
        assert names == {"A[0,0]", "A[1,1]"}


# =============================================================================
# Diag Tests
# =============================================================================


class TestDiag:
    """Tests for diag function (extract diagonal)."""

    def test_diag_extracts_diagonal(self):
        """diag extracts diagonal from MatrixVariable."""
        A = MatrixVariable("A", 3, 3)
        d = diag(A)
        assert isinstance(d, VectorVariable)
        assert len(d) == 3

    def test_diag_correct_variables(self):
        """diag contains correct diagonal variables."""
        A = MatrixVariable("A", 3, 3)
        d = diag(A)
        # Should be same Variable objects
        assert d[0] is A[0, 0]
        assert d[1] is A[1, 1]
        assert d[2] is A[2, 2]

    def test_diag_non_square_raises(self):
        """diag of non-square matrix raises SquareMatrixError."""
        A = MatrixVariable("A", 2, 3)
        with pytest.raises(SquareMatrixError, match="square"):
            diag(A)

    def test_diag_numpy_array(self):
        """diag of numpy array uses numpy.diag."""
        A = np.array([[1, 2], [3, 4]])
        d = diag(A)
        np.testing.assert_array_equal(d, [1, 4])

    def test_diag_vector_raises(self):
        """diag on VectorVariable raises InvalidOperationError."""
        x = VectorVariable("x", 3)
        with pytest.raises(InvalidOperationError, match="diag_matrix"):
            diag(x)

    def test_diag_symmetric(self):
        """diag works on symmetric matrix."""
        S = MatrixVariable("S", 3, 3, symmetric=True)
        d = diag(S)
        assert len(d) == 3
        assert d[0] is S[0, 0]
        assert d[1] is S[1, 1]
        assert d[2] is S[2, 2]


class TestDiagMatrix:
    """Tests for diag_matrix function (create diagonal matrix)."""

    def test_diag_matrix_basic(self):
        """diag_matrix creates MatrixVariable from vector."""
        x = VectorVariable("x", 3)
        D = diag_matrix(x)
        assert isinstance(D, MatrixVariable)
        assert D.shape == (3, 3)

    def test_diag_matrix_diagonal_is_vector(self):
        """Diagonal elements are the vector's variables."""
        x = VectorVariable("x", 3)
        D = diag_matrix(x)
        # Diagonal should be same Variable objects
        assert D[0, 0] is x[0]
        assert D[1, 1] is x[1]
        assert D[2, 2] is x[2]

    def test_diag_matrix_off_diagonal_zero(self):
        """Off-diagonal elements are fixed at zero."""
        x = VectorVariable("x", 2)
        D = diag_matrix(x)
        # Off-diagonal elements should have lb=0, ub=0
        assert D[0, 1].lb == 0.0
        assert D[0, 1].ub == 0.0
        assert D[1, 0].lb == 0.0
        assert D[1, 0].ub == 0.0

    def test_diag_matrix_numpy_array(self):
        """diag_matrix of numpy array uses numpy.diag."""
        x = np.array([1, 2, 3])
        D = diag_matrix(x)
        expected = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        np.testing.assert_array_equal(D, expected)

    def test_diag_matrix_evaluate(self):
        """diag_matrix result evaluates correctly."""
        x = VectorVariable("x", 2)
        D = diag_matrix(x)
        values = {"x[0]": 5, "x[1]": 10, "_diag_x[0,1]": 0, "_diag_x[1,0]": 0}
        # Diagonal
        assert D[0, 0].evaluate(values) == 5
        assert D[1, 1].evaluate(values) == 10
        # Off-diagonal (fixed at 0)
        assert D[0, 1].evaluate(values) == 0
        assert D[1, 0].evaluate(values) == 0


# =============================================================================
# Frobenius Norm Tests
# =============================================================================


class TestFrobeniusNorm:
    """Tests for Frobenius norm."""

    def test_frobenius_norm_basic(self):
        """frobenius_norm creates FrobeniusNorm expression."""
        A = MatrixVariable("A", 2, 2)
        fn = frobenius_norm(A)
        assert isinstance(fn, FrobeniusNorm)

    def test_frobenius_norm_evaluates_correctly(self):
        """Frobenius norm is sqrt of sum of squares."""
        A = MatrixVariable("A", 2, 2)
        fn = frobenius_norm(A)
        values = {"A[0,0]": 1, "A[0,1]": 2, "A[1,0]": 3, "A[1,1]": 4}
        result = fn.evaluate(values)
        # sqrt(1 + 4 + 9 + 16) = sqrt(30) ≈ 5.477
        expected = np.sqrt(30)
        assert abs(result - expected) < 1e-10

    def test_frobenius_norm_identity(self):
        """Frobenius norm of identity matrix."""
        A = MatrixVariable("A", 3, 3)
        fn = frobenius_norm(A)
        values = {
            "A[0,0]": 1,
            "A[0,1]": 0,
            "A[0,2]": 0,
            "A[1,0]": 0,
            "A[1,1]": 1,
            "A[1,2]": 0,
            "A[2,0]": 0,
            "A[2,1]": 0,
            "A[2,2]": 1,
        }
        result = fn.evaluate(values)
        # sqrt(1 + 1 + 1) = sqrt(3)
        assert abs(result - np.sqrt(3)) < 1e-10

    def test_frobenius_norm_get_variables(self):
        """FrobeniusNorm tracks all matrix variables."""
        A = MatrixVariable("A", 2, 2)
        fn = frobenius_norm(A)
        vars_set = fn.get_variables()
        assert len(vars_set) == 4

    def test_frobenius_norm_repr(self):
        """FrobeniusNorm has readable repr."""
        A = MatrixVariable("A", 2, 2)
        fn = frobenius_norm(A)
        r = repr(fn)
        assert "FrobeniusNorm" in r
        assert "A" in r


# =============================================================================
# Integration Tests
# =============================================================================


class TestMatrixOperationsIntegration:
    """Integration tests for matrix operations."""

    def test_portfolio_optimization_expressions(self):
        """Build portfolio optimization expressions."""
        # Returns and covariance
        returns = np.array([0.12, 0.08, 0.10])
        cov = np.array(
            [
                [0.04, 0.01, 0.02],
                [0.01, 0.09, 0.01],
                [0.02, 0.01, 0.16],
            ]
        )

        # Weights
        w = VectorVariable("w", 3, lb=0, ub=1)

        # Expected return: r' @ w (LinearCombination)
        from optyx.core.vectors import LinearCombination

        expected_return = LinearCombination(returns, w)

        # Variance: w' @ Σ @ w
        variance = quadratic_form(w, cov)

        # Test with equal weights
        equal_weights = {"w[0]": 1 / 3, "w[1]": 1 / 3, "w[2]": 1 / 3}

        expected_ret = expected_return.evaluate(equal_weights)
        var = variance.evaluate(equal_weights)

        # Verify reasonable values
        assert 0 < expected_ret < 0.15
        assert 0 < var < 0.1

    def test_trace_of_outer_product_concept(self):
        """Trace of matrix equals sum of diagonal."""
        A = MatrixVariable("A", 3, 3)
        tr = trace(A)
        d = diag(A)

        # Create sum of diagonal
        from optyx.core.vectors import vector_sum

        diag_sum = vector_sum(d)

        values = {
            "A[0,0]": 1,
            "A[0,1]": 2,
            "A[0,2]": 3,
            "A[1,0]": 4,
            "A[1,1]": 5,
            "A[1,2]": 6,
            "A[2,0]": 7,
            "A[2,1]": 8,
            "A[2,2]": 9,
        }

        # Both should give same result
        assert tr.evaluate(values) == diag_sum.evaluate(values)

    def test_matrix_chain(self):
        """Can chain matrix operations."""
        A = np.array([[1, 2], [3, 4]])
        x = VectorVariable("x", 2)

        # A @ x gives VectorExpression
        y = matmul(A, x)
        assert len(y) == 2

        # Can do arithmetic on result
        z = y[0] + y[1]
        values = {"x[0]": 1, "x[1]": 1}
        # y = [3, 7], z = 10
        assert z.evaluate(values) == 10.0
