"""Tests for MatrixVariable."""

import numpy as np
import pytest

from optyx.core.matrices import MatrixVariable
from optyx.core.vectors import VectorVariable
from optyx.core.expressions import Variable


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
        with pytest.raises(ValueError, match="positive"):
            MatrixVariable("A", 0, 5)

    def test_zero_cols_raises(self):
        """Cols must be positive."""
        with pytest.raises(ValueError, match="positive"):
            MatrixVariable("A", 5, 0)

    def test_negative_size_raises(self):
        """Negative size raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
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
        with pytest.raises(ValueError, match="square"):
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
