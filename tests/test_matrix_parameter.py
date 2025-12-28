"""Tests for MatrixParameter class."""

from __future__ import annotations

import numpy as np
import pytest

from optyx import MatrixParameter


class TestMatrixParameterCreation:
    """Test MatrixParameter construction."""

    def test_create_basic(self):
        """Create a basic 2x3 matrix parameter."""
        values = np.array([[1, 2, 3], [4, 5, 6]])
        M = MatrixParameter("M", values)

        assert M.name == "M"
        assert M.shape == (2, 3)
        assert M.rows == 2
        assert M.cols == 3

    def test_create_from_list(self):
        """Create from nested list."""
        M = MatrixParameter("M", [[1, 2], [3, 4]])

        assert M.shape == (2, 2)
        np.testing.assert_array_equal(M.values, [[1, 2], [3, 4]])

    def test_create_square(self):
        """Create a square matrix."""
        M = MatrixParameter("M", np.eye(5))

        assert M.shape == (5, 5)
        np.testing.assert_array_equal(M.values, np.eye(5))

    def test_create_symmetric(self):
        """Create a symmetric matrix parameter."""
        cov = np.array([[0.04, 0.01], [0.01, 0.09]])
        Sigma = MatrixParameter("Sigma", cov, symmetric=True)

        assert Sigma.symmetric is True
        assert Sigma.shape == (2, 2)

    def test_reject_1d_array(self):
        """Reject 1D arrays (use VectorParameter instead)."""
        with pytest.raises(ValueError, match="2D array"):
            MatrixParameter("v", np.array([1, 2, 3]))

    def test_reject_3d_array(self):
        """Reject 3D arrays."""
        with pytest.raises(ValueError, match="2D array"):
            MatrixParameter("T", np.ones((2, 3, 4)))

    def test_reject_non_symmetric(self):
        """Reject non-symmetric matrix when symmetric=True."""
        non_sym = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="not symmetric"):
            MatrixParameter("M", non_sym, symmetric=True)

    def test_reject_non_square_symmetric(self):
        """Reject non-square matrix when symmetric=True."""
        rect = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError, match="must be square"):
            MatrixParameter("M", rect, symmetric=True)

    def test_values_are_copied(self):
        """Ensure values are copied, not referenced."""
        original = np.array([[1, 2], [3, 4]], dtype=np.float64)
        M = MatrixParameter("M", original)

        original[0, 0] = 999
        assert M[0, 0] == 1  # Not affected


class TestMatrixParameterIndexing:
    """Test indexing operations."""

    def test_getitem_single(self):
        """Get single element with 2D indexing."""
        M = MatrixParameter("M", [[1, 2, 3], [4, 5, 6]])

        assert M[0, 0] == 1
        assert M[0, 2] == 3
        assert M[1, 1] == 5

    def test_getitem_negative_index(self):
        """Negative indices work as expected."""
        M = MatrixParameter("M", [[1, 2], [3, 4]])

        assert M[-1, -1] == 4
        assert M[-1, 0] == 3
        assert M[0, -1] == 2

    def test_getitem_returns_float(self):
        """Indexing returns Python float, not numpy scalar."""
        M = MatrixParameter("M", [[1, 2], [3, 4]])

        result = M[0, 0]
        assert isinstance(result, float)

    def test_getitem_requires_tuple(self):
        """Reject single-index access."""
        M = MatrixParameter("M", [[1, 2], [3, 4]])

        with pytest.raises(TypeError, match="2D indexing"):
            M[0]  # type: ignore

    def test_getitem_out_of_bounds(self):
        """Out-of-bounds raises IndexError."""
        M = MatrixParameter("M", [[1, 2], [3, 4]])

        with pytest.raises(IndexError):
            M[5, 0]


class TestMatrixParameterUpdate:
    """Test value updates."""

    def test_set_values(self):
        """Update all values."""
        M = MatrixParameter("M", [[1, 2], [3, 4]])
        M.set([[10, 20], [30, 40]])

        np.testing.assert_array_equal(M.values, [[10, 20], [30, 40]])

    def test_set_from_list(self):
        """Update from nested list."""
        M = MatrixParameter("M", np.zeros((2, 2)))
        M.set([[1, 2], [3, 4]])

        assert M[1, 1] == 4

    def test_set_preserves_shape(self):
        """Cannot change shape on update."""
        M = MatrixParameter("M", [[1, 2], [3, 4]])

        with pytest.raises(ValueError, match="Shape mismatch"):
            M.set([[1, 2, 3], [4, 5, 6]])

    def test_set_symmetric_valid(self):
        """Update symmetric matrix with symmetric values."""
        Sigma = MatrixParameter("Sigma", [[1, 0.5], [0.5, 2]], symmetric=True)
        Sigma.set([[2, 0.3], [0.3, 3]])

        assert Sigma[0, 1] == pytest.approx(0.3)
        assert Sigma[1, 0] == pytest.approx(0.3)

    def test_set_symmetric_invalid(self):
        """Reject non-symmetric update for symmetric matrix."""
        Sigma = MatrixParameter("Sigma", [[1, 0.5], [0.5, 2]], symmetric=True)

        with pytest.raises(ValueError, match="not symmetric"):
            Sigma.set([[1, 0.5], [0.9, 2]])  # Not symmetric

    def test_set_copies_values(self):
        """Set copies values, doesn't reference."""
        M = MatrixParameter("M", [[1, 2], [3, 4]])
        new_vals = np.array([[10, 20], [30, 40]], dtype=np.float64)
        M.set(new_vals)

        new_vals[0, 0] = 999
        assert M[0, 0] == 10  # Not affected


class TestMatrixParameterRowCol:
    """Test row/column access."""

    def test_row(self):
        """Get a row as 1D array."""
        M = MatrixParameter("M", [[1, 2, 3], [4, 5, 6]])

        row0 = M.row(0)
        np.testing.assert_array_equal(row0, [1, 2, 3])

        row1 = M.row(1)
        np.testing.assert_array_equal(row1, [4, 5, 6])

    def test_col(self):
        """Get a column as 1D array."""
        M = MatrixParameter("M", [[1, 2], [3, 4], [5, 6]])

        col0 = M.col(0)
        np.testing.assert_array_equal(col0, [1, 3, 5])

        col1 = M.col(1)
        np.testing.assert_array_equal(col1, [2, 4, 6])

    def test_row_returns_copy(self):
        """Row returns a copy, not a view."""
        M = MatrixParameter("M", [[1, 2], [3, 4]])
        row = M.row(0)
        row[0] = 999

        assert M[0, 0] == 1  # Original unchanged

    def test_col_returns_copy(self):
        """Col returns a copy, not a view."""
        M = MatrixParameter("M", [[1, 2], [3, 4]])
        col = M.col(0)
        col[0] = 999

        assert M[0, 0] == 1  # Original unchanged


class TestMatrixParameterOperations:
    """Test matrix operations."""

    def test_matmul_vector(self):
        """Matrix @ vector multiplication."""
        A = MatrixParameter("A", [[1, 2], [3, 4]])
        x = np.array([1, 1])

        result = A @ x
        np.testing.assert_array_equal(result, [3, 7])

    def test_matmul_matrix(self):
        """Matrix @ matrix multiplication."""
        A = MatrixParameter("A", [[1, 2], [3, 4]])
        B = np.array([[1, 0], [0, 1]])

        result = A @ B
        np.testing.assert_array_equal(result, [[1, 2], [3, 4]])

    def test_rmatmul_vector(self):
        """Vector @ matrix (left multiplication) via to_numpy."""
        A = MatrixParameter("A", [[1, 2], [3, 4]])
        x = np.array([1, 1])

        # Note: numpy array @ MatrixParameter doesn't call __rmatmul__
        # Use explicit conversion
        result = x @ A.to_numpy()
        np.testing.assert_array_equal(result, [4, 6])

    def test_to_numpy(self):
        """Export to numpy array."""
        M = MatrixParameter("M", [[1, 2], [3, 4]])
        arr = M.to_numpy()

        np.testing.assert_array_equal(arr, [[1, 2], [3, 4]])
        assert isinstance(arr, np.ndarray)

    def test_to_numpy_returns_copy(self):
        """to_numpy returns a copy."""
        M = MatrixParameter("M", [[1, 2], [3, 4]])
        arr = M.to_numpy()
        arr[0, 0] = 999

        assert M[0, 0] == 1  # Original unchanged


class TestMatrixParameterRepr:
    """Test string representation."""

    def test_repr_basic(self):
        """Basic repr format."""
        M = MatrixParameter("M", [[1, 2], [3, 4]])

        assert repr(M) == "MatrixParameter('M', shape=(2, 2))"

    def test_repr_rectangular(self):
        """Repr for non-square matrix."""
        M = MatrixParameter("M", np.zeros((3, 5)))

        assert repr(M) == "MatrixParameter('M', shape=(3, 5))"

    def test_repr_symmetric(self):
        """Repr shows symmetric flag."""
        Sigma = MatrixParameter("Sigma", np.eye(3), symmetric=True)

        assert repr(Sigma) == "MatrixParameter('Sigma', shape=(3, 3), symmetric=True)"


class TestMatrixParameterUseCases:
    """Test realistic use cases."""

    def test_covariance_matrix_portfolio(self):
        """Portfolio covariance matrix use case."""
        # 3 assets with known covariances
        cov = np.array(
            [
                [0.04, 0.01, 0.005],
                [0.01, 0.09, 0.02],
                [0.005, 0.02, 0.16],
            ]
        )
        Sigma = MatrixParameter("Sigma", cov, symmetric=True)

        # Compute portfolio variance: w' @ Sigma @ w
        weights = np.array([0.5, 0.3, 0.2])
        # Use Sigma @ w first (works), then left multiply
        variance = weights @ (Sigma @ weights)

        # Manual calculation
        expected = weights @ cov @ weights
        assert variance == pytest.approx(expected)

    def test_distance_matrix(self):
        """Distance/cost matrix for routing."""
        distances = np.array(
            [
                [0, 10, 20, 15],
                [10, 0, 25, 30],
                [20, 25, 0, 35],
                [15, 30, 35, 0],
            ]
        )
        D = MatrixParameter("D", distances, symmetric=True)

        # Access specific distance
        assert D[0, 1] == 10
        assert D[2, 3] == 35

        # Update distances
        new_distances = distances.copy()
        new_distances[0, 1] = 12
        new_distances[1, 0] = 12
        D.set(new_distances)

        assert D[0, 1] == 12

    def test_constraint_matrix(self):
        """Time-varying constraint matrix."""
        # A @ x <= b where A changes over time
        A = MatrixParameter("A", [[1, 2], [3, 4], [5, 6]])

        # Simulate constraint evaluation
        x = np.array([1, 1])
        lhs = A @ x

        np.testing.assert_array_equal(lhs, [3, 7, 11])

        # Update A for next time period
        A.set([[2, 1], [4, 3], [6, 5]])
        lhs_new = A @ x

        np.testing.assert_array_equal(
            lhs_new, [3, 7, 11]
        )  # Same result, different matrix

    def test_large_covariance_update(self):
        """Update large covariance matrix efficiently."""
        n = 100
        # Create random positive semi-definite matrix
        rng = np.random.default_rng(42)
        L = rng.standard_normal((n, n))
        cov1 = L @ L.T / n

        Sigma = MatrixParameter("Sigma", cov1, symmetric=True)
        assert Sigma.shape == (n, n)

        # Update with new covariance
        L2 = rng.standard_normal((n, n))
        cov2 = L2 @ L2.T / n
        Sigma.set(cov2)

        np.testing.assert_allclose(Sigma.to_numpy(), cov2)
