"""Tests for MatrixVariable arithmetic operations."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from optyx.core.matrices import MatrixVariable, MatrixExpression
from optyx.core.expressions import BinaryOp
from optyx.core.errors import DimensionMismatchError


class TestMatrixExpressionBasics:
    """Test MatrixExpression class fundamentals."""

    def test_matrix_expression_shape(self):
        X = MatrixVariable("X", 2, 3)
        Y = X + 1
        assert Y.shape == (2, 3)
        assert Y.rows == 2
        assert Y.cols == 3

    def test_matrix_expression_indexing(self):
        X = MatrixVariable("X", 2, 2)
        Y = X + 1
        elem = Y[0, 1]
        assert isinstance(elem, BinaryOp)

    def test_matrix_expression_flatten(self):
        X = MatrixVariable("X", 2, 3)
        Y = X * 2
        flat = Y.flatten()
        assert len(flat) == 6  # 2 * 3


class TestMatrixMinusScalar:
    """Test matrix - scalar operations."""

    def test_matrix_minus_scalar(self):
        X = MatrixVariable("X", 2, 2)
        Y = X - 5
        assert isinstance(Y, MatrixExpression)
        assert Y.shape == (2, 2)

    def test_scalar_minus_matrix(self):
        X = MatrixVariable("X", 2, 2)
        Y = 10 - X
        assert isinstance(Y, MatrixExpression)
        vals = {"X[0,0]": 1, "X[0,1]": 2, "X[1,0]": 3, "X[1,1]": 4}
        result = Y.evaluate(vals)
        expected = np.array([[9, 8], [7, 6]])
        assert_array_equal(result, expected)


class TestMatrixMinusArray:
    """Test matrix - array operations."""

    def test_matrix_minus_array(self):
        X = MatrixVariable("X", 2, 2)
        C = np.ones((2, 2)) * 5.0
        Y = X - C
        assert isinstance(Y, MatrixExpression)
        vals = {"X[0,0]": 1, "X[0,1]": 2, "X[1,0]": 3, "X[1,1]": 4}
        result = Y.evaluate(vals)
        expected = np.array([[1 - 5, 2 - 5], [3 - 5, 4 - 5]])
        assert_array_equal(result, expected)

    def test_array_minus_matrix(self):
        X = MatrixVariable("X", 2, 2)
        C = np.ones((2, 2)) * 10.0
        Y = C - X
        vals = {"X[0,0]": 1, "X[0,1]": 2, "X[1,0]": 3, "X[1,1]": 4}
        result = Y.evaluate(vals)
        expected = np.array([[10 - 1, 10 - 2], [10 - 3, 10 - 4]])
        assert_array_equal(result, expected)


class TestMatrixPlusMatrix:
    """Test matrix + matrix operations."""

    def test_matrix_plus_matrix(self):
        X = MatrixVariable("X", 2, 2)
        Y = MatrixVariable("Y", 2, 2)
        Z = X + Y
        assert isinstance(Z, MatrixExpression)

    def test_matrix_plus_matrix_evaluate(self):
        X = MatrixVariable("X", 2, 2)
        Y = MatrixVariable("Y", 2, 2)
        Z = X + Y
        vals = {
            "X[0,0]": 1,
            "X[0,1]": 2,
            "X[1,0]": 3,
            "X[1,1]": 4,
            "Y[0,0]": 10,
            "Y[0,1]": 20,
            "Y[1,0]": 30,
            "Y[1,1]": 40,
        }
        result = Z.evaluate(vals)
        expected = np.array([[11, 22], [33, 44]])
        assert_array_equal(result, expected)


class TestScalarMultiplication:
    """Test scalar multiplication operations."""

    def test_scalar_times_matrix(self):
        X = MatrixVariable("X", 2, 2)
        Y = 2 * X
        assert Y.shape == (2, 2)
        vals = {"X[0,0]": 1, "X[0,1]": 2, "X[1,0]": 3, "X[1,1]": 4}
        result = Y.evaluate(vals)
        expected = np.array([[2, 4], [6, 8]])
        assert_array_equal(result, expected)

    def test_matrix_times_scalar(self):
        X = MatrixVariable("X", 2, 2)
        Y = X * 3
        assert Y.shape == (2, 2)
        vals = {"X[0,0]": 1, "X[0,1]": 2, "X[1,0]": 3, "X[1,1]": 4}
        result = Y.evaluate(vals)
        expected = np.array([[3, 6], [9, 12]])
        assert_array_equal(result, expected)


class TestNegation:
    """Test unary negation."""

    def test_negation(self):
        X = MatrixVariable("X", 2, 2)
        Y = -X
        vals = {"X[0,0]": 1, "X[0,1]": 2, "X[1,0]": 3, "X[1,1]": 4}
        result = Y.evaluate(vals)
        expected = np.array([[-1, -2], [-3, -4]])
        assert_array_equal(result, expected)


class TestDivision:
    """Test division operations."""

    def test_matrix_divided_by_scalar(self):
        X = MatrixVariable("X", 2, 2)
        Y = X / 2
        vals = {"X[0,0]": 4, "X[0,1]": 6, "X[1,0]": 8, "X[1,1]": 10}
        result = Y.evaluate(vals)
        expected = np.array([[2, 3], [4, 5]])
        assert_array_equal(result, expected)

    def test_scalar_divided_by_matrix(self):
        X = MatrixVariable("X", 2, 2)
        Y = 12 / X
        vals = {"X[0,0]": 1, "X[0,1]": 2, "X[1,0]": 3, "X[1,1]": 4}
        result = Y.evaluate(vals)
        expected = np.array([[12, 6], [4, 3]])
        assert_array_equal(result, expected)


class TestPower:
    """Test power operations."""

    def test_matrix_power(self):
        X = MatrixVariable("X", 2, 2)
        Y = X**2
        vals = {"X[0,0]": 2, "X[0,1]": 3, "X[1,0]": 4, "X[1,1]": 5}
        result = Y.evaluate(vals)
        expected = np.array([[4, 9], [16, 25]])
        assert_array_equal(result, expected)


class TestChainedOperations:
    """Test chained arithmetic operations."""

    def test_chained_operations(self):
        X = MatrixVariable("X", 2, 2)
        C = np.ones((2, 2))
        Y = (X - C) * 2 + 1
        assert isinstance(Y, MatrixExpression)

    def test_chained_operations_evaluate(self):
        X = MatrixVariable("X", 2, 2)
        C = np.array([[1, 2], [3, 4]])
        Y = (X - C) * 2 + 10
        vals = {"X[0,0]": 5, "X[0,1]": 5, "X[1,0]": 5, "X[1,1]": 5}
        result = Y.evaluate(vals)
        # (5-1)*2+10=18, (5-2)*2+10=16, (5-3)*2+10=14, (5-4)*2+10=12
        expected = np.array([[18, 16], [14, 12]])
        assert_array_equal(result, expected)


class TestListInput:
    """Test list input conversion."""

    def test_list_input(self):
        X = MatrixVariable("X", 2, 2)
        C = [[1, 2], [3, 4]]
        Y = X + C
        assert isinstance(Y, MatrixExpression)
        vals = {"X[0,0]": 10, "X[0,1]": 20, "X[1,0]": 30, "X[1,1]": 40}
        result = Y.evaluate(vals)
        expected = np.array([[11, 22], [33, 44]])
        assert_array_equal(result, expected)


class TestShapeMismatch:
    """Test shape mismatch error handling."""

    def test_shape_mismatch_raises(self):
        X = MatrixVariable("X", 2, 3)
        C = np.ones((3, 2))  # Wrong shape
        with pytest.raises(DimensionMismatchError):
            _ = X - C

    def test_matrix_matrix_shape_mismatch(self):
        X = MatrixVariable("X", 2, 2)
        Y = MatrixVariable("Y", 3, 3)
        with pytest.raises(DimensionMismatchError):
            _ = X + Y


class TestMatrixExpressionOperators:
    """Test operators on MatrixExpression (chaining)."""

    def test_expression_plus_scalar(self):
        X = MatrixVariable("X", 2, 2)
        Y = X + 1  # MatrixExpression
        Z = Y + 2  # Chained
        vals = {"X[0,0]": 10, "X[0,1]": 20, "X[1,0]": 30, "X[1,1]": 40}
        result = Z.evaluate(vals)
        expected = np.array([[13, 23], [33, 43]])
        assert_array_equal(result, expected)

    def test_expression_minus_expression(self):
        X = MatrixVariable("X", 2, 2)
        A = X + 1
        B = X * 2
        C = A - B  # (X+1) - (X*2) = 1 - X
        vals = {"X[0,0]": 1, "X[0,1]": 2, "X[1,0]": 3, "X[1,1]": 4}
        result = C.evaluate(vals)
        # (1+1)-(1*2)=0, (2+1)-(2*2)=-1, (3+1)-(3*2)=-2, (4+1)-(4*2)=-3
        expected = np.array([[0, -1], [-2, -3]])
        assert_array_equal(result, expected)

    def test_expression_negation(self):
        X = MatrixVariable("X", 2, 2)
        Y = X + 1
        Z = -Y
        vals = {"X[0,0]": 1, "X[0,1]": 2, "X[1,0]": 3, "X[1,1]": 4}
        result = Z.evaluate(vals)
        expected = np.array([[-2, -3], [-4, -5]])
        assert_array_equal(result, expected)


class TestHadamardProduct:
    """Test element-wise matrix multiplication (Hadamard product)."""

    def test_matrix_hadamard_matrix(self):
        X = MatrixVariable("X", 2, 2)
        Y = MatrixVariable("Y", 2, 2)
        Z = X * Y  # Element-wise, not matrix mult
        vals = {
            "X[0,0]": 1,
            "X[0,1]": 2,
            "X[1,0]": 3,
            "X[1,1]": 4,
            "Y[0,0]": 2,
            "Y[0,1]": 3,
            "Y[1,0]": 4,
            "Y[1,1]": 5,
        }
        result = Z.evaluate(vals)
        expected = np.array([[2, 6], [12, 20]])
        assert_array_equal(result, expected)

    def test_matrix_hadamard_array(self):
        X = MatrixVariable("X", 2, 2)
        C = np.array([[2, 3], [4, 5]])
        Z = X * C
        vals = {"X[0,0]": 1, "X[0,1]": 2, "X[1,0]": 3, "X[1,1]": 4}
        result = Z.evaluate(vals)
        expected = np.array([[2, 6], [12, 20]])
        assert_array_equal(result, expected)
