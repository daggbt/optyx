"""Tests for sparse gradient and Jacobian compilation (Issue #95)."""

import numpy as np
import pytest
from scipy import sparse

from optyx import Variable, sin, exp
from optyx.core.autodiff import (
    compile_jacobian,
    compile_sparse_jacobian,
)
from optyx.core.compiler import (
    compile_sparse_gradient,
    compile_gradient_with_sparsity,
    compile_gradient,
)


class TestCompileSparseGradient:
    """Tests for compile_sparse_gradient function."""

    def test_constant_linear_gradient(self):
        """Linear expression should return constant sparse gradient."""
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")
        expr = 2 * x + 3 * z  # y has zero gradient

        grad_fn = compile_sparse_gradient(expr, [x, y, z])
        result = grad_fn(np.array([1.0, 2.0, 3.0]))

        assert sparse.issparse(result)
        assert result.shape == (1, 3)
        dense = result.toarray().flatten()
        np.testing.assert_array_almost_equal(dense, [2.0, 0.0, 3.0])

    def test_constant_gradient_returns_same_object(self):
        """Constant sparse gradient should return same object each call."""
        x = Variable("x")
        y = Variable("y")
        expr = 5 * x + 2 * y

        grad_fn = compile_sparse_gradient(expr, [x, y])
        r1 = grad_fn(np.array([1.0, 2.0]))
        r2 = grad_fn(np.array([3.0, 4.0]))
        assert r1 is r2

    def test_nonlinear_sparse_gradient(self):
        """Non-linear expression with sparse gradient."""
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")
        expr = x**2 + z**3  # gradient: [2x, 0, 3z²]

        grad_fn = compile_sparse_gradient(expr, [x, y, z])

        result = grad_fn(np.array([3.0, 0.0, 2.0]))
        assert sparse.issparse(result)
        dense = result.toarray().flatten()
        np.testing.assert_array_almost_equal(dense, [6.0, 0.0, 12.0])

    def test_nonlinear_sparse_gradient_varies_with_x(self):
        """Non-linear sparse gradient should give different values at different x."""
        x = Variable("x")
        y = Variable("y")
        expr = x**2  # gradient: [2x, 0]

        grad_fn = compile_sparse_gradient(expr, [x, y])

        r1 = grad_fn(np.array([1.0, 0.0]))
        r2 = grad_fn(np.array([5.0, 0.0]))

        assert r1.toarray()[0, 0] == pytest.approx(2.0)
        assert r2.toarray()[0, 0] == pytest.approx(10.0)

    def test_zero_gradient(self):
        """Expression independent of all variables."""
        x = Variable("x")
        y = Variable("y")
        from optyx.core.expressions import Constant

        expr = Constant(42.0)

        grad_fn = compile_sparse_gradient(expr, [x, y])
        result = grad_fn(np.array([1.0, 2.0]))

        assert sparse.issparse(result)
        assert result.nnz == 0
        assert result.shape == (1, 2)

    def test_sparse_format_is_csr(self):
        """Output should be in CSR format."""
        x = Variable("x")
        y = Variable("y")
        expr = x + y

        grad_fn = compile_sparse_gradient(expr, [x, y])
        result = grad_fn(np.array([1.0, 2.0]))
        assert isinstance(result, sparse.csr_matrix)

    def test_large_sparse_gradient(self):
        """Sparse gradient with many variables but few non-zero."""
        variables = [Variable(f"x{i}") for i in range(100)]
        # Only depends on variables 0, 50, 99
        expr = 2 * variables[0] + 3 * variables[50] + variables[99]

        grad_fn = compile_sparse_gradient(expr, variables)
        x = np.zeros(100)
        result = grad_fn(x)

        assert sparse.issparse(result)
        assert result.shape == (1, 100)
        assert result.nnz == 3
        dense = result.toarray().flatten()
        assert dense[0] == pytest.approx(2.0)
        assert dense[50] == pytest.approx(3.0)
        assert dense[99] == pytest.approx(1.0)

    def test_matches_dense_gradient(self):
        """Sparse gradient should match dense gradient numerically."""
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")
        expr = x**2 * sin(z) + 3 * y

        variables = [x, y, z]
        sparse_fn = compile_sparse_gradient(expr, variables)
        dense_fn = compile_gradient(expr, variables)

        for _ in range(5):
            point = np.random.randn(3)
            sparse_result = sparse_fn(point).toarray().flatten()
            dense_result = dense_fn(point)
            np.testing.assert_array_almost_equal(sparse_result, dense_result)

    def test_transcendental_sparse_gradient(self):
        """Transcendental function with sparse gradient."""
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")
        expr = sin(x) + exp(z)  # gradient: [cos(x), 0, exp(z)]

        grad_fn = compile_sparse_gradient(expr, [x, y, z])
        result = grad_fn(np.array([0.0, 0.0, 0.0]))

        dense = result.toarray().flatten()
        np.testing.assert_array_almost_equal(dense, [1.0, 0.0, 1.0])


class TestCompileGradientWithSparsity:
    """Tests for compile_gradient_with_sparsity function."""

    def test_sparse_below_threshold(self):
        """Returns sparse when density is below threshold."""
        variables = [Variable(f"x{i}") for i in range(10)]
        expr = variables[0] + variables[1]  # 2/10 = 0.2 density

        grad_fn = compile_gradient_with_sparsity(expr, variables)
        result = grad_fn(np.zeros(10))

        # Should be sparse (density 0.2 < default threshold 0.5)
        assert sparse.issparse(result)

    def test_dense_above_threshold(self):
        """Returns dense when density is above threshold."""
        x = Variable("x")
        y = Variable("y")
        expr = x + y  # 2/2 = 1.0 density

        grad_fn = compile_gradient_with_sparsity(expr, [x, y])
        result = grad_fn(np.array([1.0, 2.0]))

        # Should be dense (density 1.0 > default threshold 0.5)
        assert isinstance(result, np.ndarray)

    def test_custom_threshold(self):
        """Custom threshold controls sparse/dense selection."""
        variables = [Variable(f"x{i}") for i in range(10)]
        expr = sum(variables[:4], variables[0])  # 4/10 = 0.4 density

        # With threshold 0.3 → should be dense (0.4 > 0.3)
        grad_fn_dense = compile_gradient_with_sparsity(
            expr, variables, density_threshold=0.3
        )
        result_dense = grad_fn_dense(np.zeros(10))
        assert isinstance(result_dense, np.ndarray)
        assert not sparse.issparse(result_dense)

        # With threshold 0.5 → should be sparse (0.4 < 0.5)
        grad_fn_sparse = compile_gradient_with_sparsity(
            expr, variables, density_threshold=0.5
        )
        result_sparse = grad_fn_sparse(np.zeros(10))
        assert sparse.issparse(result_sparse)

    def test_numerical_correctness(self):
        """Sparse vs dense outputs should agree numerically."""
        variables = [Variable(f"x{i}") for i in range(20)]
        expr = variables[0] ** 2 + 3 * variables[10]

        sparse_fn = compile_gradient_with_sparsity(
            expr, variables, density_threshold=1.0
        )
        dense_fn = compile_gradient_with_sparsity(
            expr, variables, density_threshold=0.0
        )

        point = np.random.randn(20)
        sparse_result = sparse_fn(point)
        dense_result = dense_fn(point)

        if sparse.issparse(sparse_result):
            sparse_result = sparse_result.toarray().flatten()
        if sparse.issparse(dense_result):
            dense_result = dense_result.toarray().flatten()

        np.testing.assert_array_almost_equal(sparse_result, dense_result)


class TestCompileSparseJacobian:
    """Tests for compile_sparse_jacobian function."""

    def test_linear_constraints_sparse(self):
        """Linear constraints produce constant sparse Jacobian."""
        n = 10
        variables = [Variable(f"x{i}") for i in range(n)]

        # Two sparse constraints: x0 + x1, x8 + x9 → density = 4/20 = 0.2
        exprs = [variables[0] + variables[1], variables[8] + variables[9]]

        jac_fn = compile_sparse_jacobian(exprs, variables)
        result = jac_fn(np.zeros(n))

        assert sparse.issparse(result)
        dense = result.toarray()
        expected = np.zeros((2, n))
        expected[0, 0] = 1.0
        expected[0, 1] = 1.0
        expected[1, 8] = 1.0
        expected[1, 9] = 1.0
        np.testing.assert_array_almost_equal(dense, expected)

    def test_constant_jacobian_same_object(self):
        """All-constant sparse Jacobian returns same object."""
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")
        exprs = [x + y, y + z]

        jac_fn = compile_sparse_jacobian(exprs, [x, y, z])
        r1 = jac_fn(np.array([1.0, 2.0, 3.0]))
        r2 = jac_fn(np.array([4.0, 5.0, 6.0]))
        assert r1 is r2

    def test_nonlinear_sparse_jacobian(self):
        """Non-linear sparse Jacobian varies with x."""
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")

        # x² only depends on x, z³ only depends on z
        exprs = [x**2, z**3]

        jac_fn = compile_sparse_jacobian(exprs, [x, y, z])

        result = jac_fn(np.array([3.0, 0.0, 2.0]))
        assert sparse.issparse(result)
        dense = result.toarray()
        expected = np.array(
            [
                [6.0, 0.0, 0.0],  # d(x²)/dx=2x, 0, 0
                [0.0, 0.0, 12.0],  # 0, 0, d(z³)/dz=3z²
            ]
        )
        np.testing.assert_array_almost_equal(dense, expected)

    def test_matches_dense_jacobian(self):
        """Sparse Jacobian matches dense Jacobian numerically."""
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")
        variables = [x, y, z]

        exprs = [x**2 + y, sin(z), 2 * x + 3 * z]

        sparse_fn = compile_sparse_jacobian(exprs, variables, density_threshold=1.0)
        dense_fn = compile_jacobian(exprs, variables)

        for _ in range(5):
            point = np.random.randn(3)
            sparse_result = sparse_fn(point)
            dense_result = dense_fn(point)

            if sparse.issparse(sparse_result):
                sparse_result = sparse_result.toarray()

            np.testing.assert_array_almost_equal(sparse_result, dense_result)

    def test_high_density_falls_back_to_dense(self):
        """High density Jacobian should fall back to dense output."""
        x = Variable("x")
        y = Variable("y")

        # Both expressions depend on both variables → density = 1.0
        exprs = [x + y, x * y]

        jac_fn = compile_sparse_jacobian(exprs, [x, y], density_threshold=0.5)
        result = jac_fn(np.array([1.0, 2.0]))

        # Should be dense (falls back)
        assert isinstance(result, np.ndarray)

    def test_empty_expressions(self):
        """Empty expression list produces empty sparse matrix."""
        x = Variable("x")
        jac_fn = compile_sparse_jacobian([], [x])
        result = jac_fn(np.array([1.0]))
        assert sparse.issparse(result)
        assert result.shape == (0, 1)

    def test_large_sparse_jacobian(self):
        """Large sparse Jacobian with few non-zeros per row."""
        n = 50
        variables = [Variable(f"x{i}") for i in range(n)]

        # Chain constraints: x_i + x_{i+1} for i in 0..n-2
        # Each row has exactly 2 non-zeros → density = 2/n = 4%
        exprs = [variables[i] + variables[i + 1] for i in range(n - 1)]

        jac_fn = compile_sparse_jacobian(exprs, variables)
        result = jac_fn(np.zeros(n))

        assert sparse.issparse(result)
        assert result.shape == (n - 1, n)
        # Each row should have exactly 2 non-zeros
        assert result.nnz == 2 * (n - 1)

        # Verify structure: row i has nonzeros at columns i and i+1
        dense = result.toarray()
        for i in range(n - 1):
            assert dense[i, i] == pytest.approx(1.0)
            assert dense[i, i + 1] == pytest.approx(1.0)
            # All other columns should be zero
            assert np.sum(np.abs(dense[i, :])) == pytest.approx(2.0)

    def test_mixed_constant_and_variable_rows(self):
        """Jacobian with some constant and some variable rows."""
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")
        variables = [x, y, z]

        # Row 0: linear (constant gradient [2, 0, 3])
        # Row 1: nonlinear (variable gradient [2x, 0, 0])
        exprs = [2 * x + 3 * z, x**2]

        jac_fn = compile_sparse_jacobian(exprs, variables)
        result = jac_fn(np.array([4.0, 0.0, 0.0]))

        assert sparse.issparse(result)
        dense = result.toarray()
        expected = np.array(
            [
                [2.0, 0.0, 3.0],
                [8.0, 0.0, 0.0],
            ]
        )
        np.testing.assert_array_almost_equal(dense, expected)

    def test_onnz_memory(self):
        """Sparse Jacobian should use O(nnz) storage, not O(m*n)."""
        n = 100
        variables = [Variable(f"x{i}") for i in range(n)]

        # n-1 chain constraints, each with 2 nonzeros
        exprs = [variables[i] + variables[i + 1] for i in range(n - 1)]

        jac_fn = compile_sparse_jacobian(exprs, variables)
        result = jac_fn(np.zeros(n))

        # CSR storage: nnz data values + nnz column indices + (m+1) row pointers
        # Should be much less than m*n = 99*100 = 9900
        total_stored = result.nnz  # data + indices
        assert total_stored == 2 * (n - 1)  # 198 vs 9900 dense


class TestSparseJacobianFormats:
    """Test sparse output format properties."""

    def test_csr_format(self):
        """Sparse Jacobian should be in CSR format."""
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")
        exprs = [x + y, z]

        jac_fn = compile_sparse_jacobian(exprs, [x, y, z])
        result = jac_fn(np.zeros(3))
        assert isinstance(result, sparse.csr_matrix)

    def test_sparse_gradient_csr_format(self):
        """Sparse gradient should be in CSR format."""
        x = Variable("x")
        y = Variable("y")
        expr = x + y

        grad_fn = compile_sparse_gradient(expr, [x, y])
        result = grad_fn(np.zeros(2))
        assert isinstance(result, sparse.csr_matrix)


class TestSparseCompilationEdgeCases:
    """Edge cases for sparse compilation."""

    def test_single_expression_single_variable(self):
        """Simplest case: one expression, few variables to ensure sparse."""
        variables = [Variable(f"x{i}") for i in range(10)]
        expr = variables[0] ** 2  # density 1/10 = 0.1

        jac_fn = compile_sparse_jacobian([expr], variables)
        result = jac_fn(np.array([3.0] + [0.0] * 9))

        assert sparse.issparse(result)
        assert result.toarray()[0, 0] == pytest.approx(6.0)

    def test_zero_row_in_jacobian(self):
        """Expression with zero gradient (constant)."""
        x = Variable("x")
        y = Variable("y")
        from optyx.core.expressions import Constant

        exprs = [Constant(5.0), x + y]

        jac_fn = compile_sparse_jacobian(exprs, [x, y])
        result = jac_fn(np.array([1.0, 2.0]))

        assert sparse.issparse(result)
        dense = result.toarray()
        np.testing.assert_array_almost_equal(dense[0, :], [0.0, 0.0])
        np.testing.assert_array_almost_equal(dense[1, :], [1.0, 1.0])

    def test_all_zero_jacobian(self):
        """All expressions are constants → all-zero Jacobian."""
        x = Variable("x")
        from optyx.core.expressions import Constant

        exprs = [Constant(1.0), Constant(2.0)]

        jac_fn = compile_sparse_jacobian(exprs, [x])
        result = jac_fn(np.array([1.0]))

        assert sparse.issparse(result)
        assert result.nnz == 0
        assert result.shape == (2, 1)
