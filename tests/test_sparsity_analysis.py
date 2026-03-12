"""Tests for sparsity analysis (Issue #94)."""

from __future__ import annotations

import numpy as np
import pytest

from optyx import Variable, VectorVariable
from optyx.core.autodiff import (
    SparsityPattern,
    analyze_gradient_sparsity,
    analyze_jacobian_sparsity,
)
from optyx.core.expressions import Constant


class TestSparsityPattern:
    """Test SparsityPattern dataclass properties."""

    def test_nnz_count(self):
        sp = SparsityPattern(
            nnz_indices=np.array([0, 2, 4], dtype=np.intp),
            size=5,
            is_constant=False,
            constant_values=None,
        )
        assert sp.nnz == 3

    def test_density(self):
        sp = SparsityPattern(
            nnz_indices=np.array([0, 2], dtype=np.intp),
            size=10,
            is_constant=False,
            constant_values=None,
        )
        assert sp.density == pytest.approx(0.2)

    def test_is_dense(self):
        sp = SparsityPattern(
            nnz_indices=np.array([0, 1, 2], dtype=np.intp),
            size=3,
            is_constant=False,
            constant_values=None,
        )
        assert sp.is_dense

    def test_not_dense(self):
        sp = SparsityPattern(
            nnz_indices=np.array([0, 2], dtype=np.intp),
            size=3,
            is_constant=False,
            constant_values=None,
        )
        assert not sp.is_dense

    def test_empty_pattern(self):
        sp = SparsityPattern(
            nnz_indices=np.array([], dtype=np.intp),
            size=5,
            is_constant=False,
            constant_values=None,
        )
        assert sp.nnz == 0
        assert sp.density == 0.0
        assert not sp.is_dense


class TestAnalyzeGradientSparsity:
    """Test analyze_gradient_sparsity for various expression types."""

    def test_constant_expression(self):
        """Constant expression has no non-zero gradients."""
        x = Variable("x")
        y = Variable("y")
        expr = Constant(5.0)
        sp = analyze_gradient_sparsity(expr, [x, y])
        assert sp.nnz == 0
        assert sp.size == 2
        assert not sp.is_dense

    def test_single_variable(self):
        """Expression depending on one variable."""
        x = Variable("x")
        y = Variable("y")
        expr = 2.0 * x + 3.0
        sp = analyze_gradient_sparsity(expr, [x, y])
        assert sp.nnz == 1
        np.testing.assert_array_equal(sp.nnz_indices, [0])
        assert sp.is_constant
        np.testing.assert_array_almost_equal(sp.constant_values, [2.0])

    def test_two_variables(self):
        """Expression depending on two variables."""
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")
        expr = x + y
        sp = analyze_gradient_sparsity(expr, [x, y, z])
        assert sp.nnz == 2
        np.testing.assert_array_equal(sp.nnz_indices, [0, 1])
        assert not sp.is_dense

    def test_all_variables(self):
        """Expression depending on all variables."""
        x = Variable("x")
        y = Variable("y")
        expr = x * y
        sp = analyze_gradient_sparsity(expr, [x, y])
        assert sp.nnz == 2
        assert sp.is_dense
        # x*y has non-constant gradients (grad_x = y, grad_y = x)
        assert not sp.is_constant

    def test_linear_expression_constant_gradients(self):
        """Linear expression has constant gradients."""
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")
        expr = 3.0 * x - 2.0 * y + 5.0
        sp = analyze_gradient_sparsity(expr, [x, y, z])
        assert sp.nnz == 2
        np.testing.assert_array_equal(sp.nnz_indices, [0, 1])
        assert sp.is_constant
        np.testing.assert_array_almost_equal(sp.constant_values, [3.0, -2.0])

    def test_quadratic_expression_nonconstant_gradients(self):
        """Quadratic expression has non-constant gradients."""
        x = Variable("x")
        y = Variable("y")
        expr = x**2 + y
        sp = analyze_gradient_sparsity(expr, [x, y])
        assert sp.nnz == 2
        # grad_x = 2*x (not constant), grad_y = 1 (constant)
        # But is_constant should be False since not ALL are constant
        assert not sp.is_constant

    def test_vector_linear_combination(self):
        """LinearCombination (c @ x) has constant gradient."""
        x = VectorVariable("x", 5)
        variables = list(x._variables)
        c = np.array([1.0, 0.0, 3.0, 0.0, 5.0])
        expr = c @ x
        sp = analyze_gradient_sparsity(expr, variables)
        # All 5 variables appear in c @ x even if coefficient is 0
        # (because LinearCombination stores all variables)
        assert sp.size == 5
        assert sp.is_constant

    def test_dot_product_dense(self):
        """DotProduct x.dot(x) depends on all variables."""
        x = VectorVariable("x", 4)
        variables = list(x._variables)
        expr = x.dot(x)
        sp = analyze_gradient_sparsity(expr, variables)
        assert sp.nnz == 4
        assert sp.is_dense
        # grad = 2*x, not constant
        assert not sp.is_constant

    def test_vector_sum(self):
        """VectorSum depends on all variables with constant gradient."""
        x = VectorVariable("x", 3)
        variables = list(x._variables)
        expr = x.sum()
        sp = analyze_gradient_sparsity(expr, variables)
        assert sp.nnz == 3
        assert sp.is_dense
        assert sp.is_constant
        np.testing.assert_array_almost_equal(sp.constant_values, [1.0, 1.0, 1.0])

    def test_sparse_constraint(self):
        """Constraint that only involves a subset of variables."""
        variables = [Variable(f"x{i}") for i in range(10)]
        # Constraint: x0 + x5 <= 10
        expr = variables[0] + variables[5]
        sp = analyze_gradient_sparsity(expr, variables)
        assert sp.nnz == 2
        assert sp.density == pytest.approx(0.2)
        np.testing.assert_array_equal(sp.nnz_indices, [0, 5])
        assert sp.is_constant
        np.testing.assert_array_almost_equal(sp.constant_values, [1.0, 1.0])


class TestAnalyzeJacobianSparsity:
    """Test analyze_jacobian_sparsity for multi-row Jacobians."""

    def test_single_row(self):
        """Single expression Jacobian."""
        x = Variable("x")
        y = Variable("y")
        patterns = analyze_jacobian_sparsity([x + y], [x, y])
        assert len(patterns) == 1
        assert patterns[0].nnz == 2

    def test_diagonal_jacobian(self):
        """Each constraint depends on exactly one variable."""
        variables = [Variable(f"x{i}") for i in range(3)]
        exprs = [
            2.0 * variables[0],
            3.0 * variables[1],
            4.0 * variables[2],
        ]
        patterns = analyze_jacobian_sparsity(exprs, variables)
        assert len(patterns) == 3
        for i, sp in enumerate(patterns):
            assert sp.nnz == 1
            np.testing.assert_array_equal(sp.nnz_indices, [i])
            assert sp.is_constant

    def test_sparse_jacobian(self):
        """Mixed sparse constraint system."""
        variables = [Variable(f"x{i}") for i in range(5)]
        exprs = [
            variables[0] + variables[1],  # depends on x0, x1
            variables[2] * variables[3],  # depends on x2, x3
            variables[4],  # depends on x4
        ]
        patterns = analyze_jacobian_sparsity(exprs, variables)
        assert len(patterns) == 3
        np.testing.assert_array_equal(patterns[0].nnz_indices, [0, 1])
        np.testing.assert_array_equal(patterns[1].nnz_indices, [2, 3])
        np.testing.assert_array_equal(patterns[2].nnz_indices, [4])

    def test_dense_jacobian(self):
        """All constraints depend on all variables."""
        x = Variable("x")
        y = Variable("y")
        exprs = [x + y, x * y]
        patterns = analyze_jacobian_sparsity(exprs, [x, y])
        assert all(sp.is_dense for sp in patterns)


class TestSparsityIntegration:
    """Test that sparsity analysis integrates correctly with compile_jacobian."""

    def test_sparse_jacobian_correctness(self):
        """Verify compile_jacobian produces correct results with sparse rows."""
        from optyx.core.autodiff import compile_jacobian

        variables = [Variable(f"x{i}") for i in range(5)]
        # Sparse constraint: only x0 + x3
        expr = variables[0] + variables[3]
        jac_fn = compile_jacobian([expr], variables)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = jac_fn(x)
        expected = np.array([[1.0, 0.0, 0.0, 1.0, 0.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_sparse_nonlinear_jacobian_correctness(self):
        """Verify sparse nonlinear Jacobian row is correct."""
        from optyx.core.autodiff import compile_jacobian

        variables = [Variable(f"x{i}") for i in range(5)]
        # Nonlinear constraint: x1^2 + x3
        expr = variables[1] ** 2 + variables[3]
        jac_fn = compile_jacobian([expr], variables)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = jac_fn(x)
        # grad = [0, 2*x1, 0, 1, 0] = [0, 4, 0, 1, 0]
        expected = np.array([[0.0, 4.0, 0.0, 1.0, 0.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_mixed_sparse_dense_rows(self):
        """Jacobian with both sparse and dense rows."""
        from optyx.core.autodiff import compile_jacobian

        variables = [Variable(f"x{i}") for i in range(3)]
        exprs = [
            variables[0],  # sparse: only x0
            variables[0] + variables[1] + variables[2],  # dense: all vars
        ]
        jac_fn = compile_jacobian(exprs, variables)
        x = np.array([1.0, 2.0, 3.0])
        result = jac_fn(x)
        expected = np.array(
            [
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ]
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_large_sparse_system(self):
        """Large system where each constraint is very sparse."""
        from optyx.core.autodiff import compile_jacobian

        n = 100
        variables = [Variable(f"x{i}") for i in range(n)]
        # Each constraint depends on only 2 consecutive variables
        exprs = [variables[i] + variables[(i + 1) % n] for i in range(n)]
        jac_fn = compile_jacobian(exprs, variables)
        x = np.ones(n)
        result = jac_fn(x)
        # Each row should have exactly 2 non-zeros
        for i in range(n):
            row = result[i]
            assert np.count_nonzero(row) == 2
            assert row[i] == 1.0
            assert row[(i + 1) % n] == 1.0
