"""Tests for matrix-form linear constraints via subject_to(A @ x <= b)."""

import numpy as np
import pytest
from scipy import sparse as sp
from scipy.optimize import linprog

from optyx.analysis import LinearProgramExtractor
from optyx import Problem, VectorVariable, as_matrix
from optyx.core.errors import DimensionMismatchError


class TestMatrixConstraintValidation:
    """Validation for direct matrix constraints."""

    def test_dimension_mismatch_columns(self):
        x = VectorVariable("x", 3, lb=0)
        A = np.ones((2, 4))  # 4 columns != 3 variables
        b = np.ones(2)
        prob = Problem().minimize(np.ones(3) @ x)
        with pytest.raises(DimensionMismatchError, match=r"\(2, 4\).*(3,)"):
            prob.subject_to(A @ x <= b)

    def test_dimension_mismatch_rows(self):
        x = VectorVariable("x", 3, lb=0)
        A = np.ones((2, 3))
        b = np.ones(5)  # 5 elements != 2 rows
        prob = Problem().minimize(np.ones(3) @ x)
        with pytest.raises(ValueError, match="2 rows.*5 elements"):
            prob.subject_to(A @ x <= b)

    def test_sparse_dimension_mismatch(self):
        x = VectorVariable("x", 3, lb=0)
        A = as_matrix(sp.csr_matrix(np.ones((2, 5))))  # 5 columns != 3
        b = np.ones(2)
        prob = Problem().minimize(np.ones(3) @ x)
        with pytest.raises(DimensionMismatchError, match=r"\(2, 5\).*(3,)"):
            prob.subject_to(A @ x <= b)


class TestAsMatrixStorage:
    """Storage policy overrides for as_matrix()."""

    def test_force_sparse_from_dense(self):
        wrapped = as_matrix(np.eye(8), storage="sparse")
        assert sp.issparse(wrapped.data)
        assert wrapped.storage == "sparse"

    def test_force_dense_from_sparse(self):
        wrapped = as_matrix(sp.eye(8, format="csr"), storage="dense")
        assert isinstance(wrapped.data, np.ndarray)
        assert wrapped.storage == "dense"

    def test_auto_keeps_small_dense_matrices_dense(self):
        wrapped = as_matrix(np.eye(4), storage="auto")
        assert isinstance(wrapped.data, np.ndarray)
        assert wrapped.storage == "dense"

    def test_auto_converts_large_sparse_like_dense_matrices(self):
        wrapped = as_matrix(np.eye(64), storage="auto")
        assert sp.issparse(wrapped.data)
        assert wrapped.storage == "sparse"

    def test_invalid_storage_raises(self):
        with pytest.raises(ValueError, match="storage"):
            as_matrix(np.eye(2), storage="invalid")


class TestSubjectToMatrixDense:
    """Dense matrix constraints."""

    def test_le_constraint(self):
        """min c'x s.t. Ax <= b, x >= 0."""
        x = VectorVariable("x", 3, lb=0)
        A = np.array([[1, 2, 0], [0, 1, 3]])
        b = np.array([10.0, 12.0])
        c = np.array([1.0, 1.0, 1.0])

        prob = Problem().minimize(c @ x)
        prob.subject_to(A @ x <= b)

        assert prob.n_constraints == 2
        assert len(prob.variables) == 3

        sol = prob.solve()
        assert sol.status.value == "optimal"
        assert sol.objective_value == pytest.approx(0.0, abs=1e-8)

    def test_eq_constraint(self):
        """min c'x s.t. Ax == b, x >= 0."""
        x = VectorVariable("x", 2, lb=0)
        A = np.array([[1.0, 1.0]])
        b = np.array([10.0])
        c = np.array([3.0, 1.0])

        prob = Problem().minimize(c @ x)
        prob.subject_to((A @ x).eq(b))

        sol = prob.solve()
        assert sol.status.value == "optimal"
        # min 3x0 + x1 s.t. x0+x1=10 → x0=0, x1=10, value=10
        assert sol.objective_value == pytest.approx(10.0, abs=1e-8)
        assert sol.values["x[0]"] == pytest.approx(0.0, abs=1e-8)
        assert sol.values["x[1]"] == pytest.approx(10.0, abs=1e-8)

    def test_ge_constraint(self):
        """min c'x s.t. Ax >= b, x >= 0, x <= 100."""
        x = VectorVariable("x", 2, lb=0, ub=100)
        A = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([5.0, 3.0])
        c = np.array([1.0, 1.0])

        prob = Problem().minimize(c @ x)
        prob.subject_to(A @ x >= b)

        sol = prob.solve()
        assert sol.status.value == "optimal"
        assert sol.objective_value == pytest.approx(8.0, abs=1e-8)

    def test_method_chaining(self):
        """Direct matrix constraints return self for fluent API."""
        x = VectorVariable("x", 2, lb=0)
        A = np.eye(2)
        b = np.ones(2)
        c = np.ones(2)

        sol = Problem().minimize(c @ x).subject_to(A @ x <= b).solve()
        assert sol.status.value == "optimal"

    def test_b_as_list(self):
        """b can be a Python list."""
        x = VectorVariable("x", 2, lb=0)
        A = np.eye(2)
        b = [5.0, 3.0]
        c = np.ones(2)

        prob = Problem().minimize(c @ x).subject_to(A @ x <= b)
        sol = prob.solve()
        assert sol.status.value == "optimal"


class TestSubjectToMatrixSparse:
    """Sparse matrix constraints."""

    def test_sparse_csr_le(self):
        """CSR sparse matrix with <= constraints."""
        n = 100
        x = VectorVariable("x", n, lb=0)
        A = as_matrix(sp.eye(n, format="csr"))
        b = np.ones(n) * 10
        c = np.ones(n)

        prob = Problem().minimize(c @ x)
        prob.subject_to(A @ x <= b)

        sol = prob.solve()
        assert sol.status.value == "optimal"
        assert sol.objective_value == pytest.approx(0.0, abs=1e-6)

    def test_sparse_csc_eq(self):
        """CSC sparse matrix with == constraints."""
        x = VectorVariable("x", 3, lb=0)
        A = as_matrix(sp.csc_matrix(np.array([[1, 1, 1]])))
        b = np.array([6.0])
        c = np.array([2.0, 1.0, 3.0])

        prob = Problem().minimize(c @ x)
        prob.subject_to((A @ x).eq(b))

        sol = prob.solve()
        assert sol.status.value == "optimal"
        # min 2x0 + x1 + 3x2 s.t. x0+x1+x2=6 → x1=6, value=6
        assert sol.objective_value == pytest.approx(6.0, abs=1e-8)

    def test_sparse_random(self):
        """Random sparse matrix constraint produces correct solution."""
        rng = np.random.default_rng(42)
        n = 50
        m = 30
        x = VectorVariable("x", n, lb=0, ub=10)
        A_sparse = sp.random(m, n, density=0.1, format="csr", random_state=rng)
        A = as_matrix(A_sparse)
        b = A_sparse @ np.ones(n) * 5  # feasible b
        c = rng.standard_normal(n)

        prob = Problem().minimize(c @ x)
        prob.subject_to(A @ x <= b)

        sol = prob.solve()
        assert sol.status.value == "optimal"

        # Verify against direct scipy linprog
        bounds = [(0, 10)] * n
        ref = linprog(c, A_ub=A_sparse, b_ub=b, bounds=bounds, method="highs")
        assert ref.success
        assert sol.objective_value == pytest.approx(ref.fun, rel=1e-6)

    def test_sparse_ge(self):
        """Sparse >= constraint."""
        x = VectorVariable("x", 3, lb=0, ub=100)
        A = as_matrix(sp.csr_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])))
        b = np.array([2.0, 3.0, 4.0])
        c = np.array([1.0, 1.0, 1.0])

        prob = Problem().minimize(c @ x)
        prob.subject_to(A @ x >= b)

        sol = prob.solve()
        assert sol.status.value == "optimal"
        assert sol.objective_value == pytest.approx(9.0, abs=1e-8)


class TestMixedConstraints:
    """Matrix constraints combined with scalar expression constraints."""

    def test_matrix_plus_expression_constraints(self):
        """Matrix constraints and expression constraints work together."""
        x = VectorVariable("x", 2, lb=0)

        prob = Problem().minimize(x[0] + x[1])
        # Matrix constraint: x0 + x1 >= 10
        A = np.array([[1.0, 1.0]])
        b = np.array([10.0])
        prob.subject_to(A @ x >= b)
        # Expression constraint: x0 <= 7
        prob.subject_to(x[0] <= 7)

        sol = prob.solve()
        assert sol.status.value == "optimal"
        assert sol.objective_value == pytest.approx(10.0, abs=1e-8)
        # x0 <= 7 and x0+x1 >= 10, so x0+x1 = 10 at optimum
        assert sol.values["x[0]"] + sol.values["x[1]"] == pytest.approx(10.0, abs=1e-8)
        assert sol.values["x[0]"] <= 7.0 + 1e-8

    def test_multiple_matrix_constraints(self):
        """Multiple matrix constraints added through subject_to()."""
        x = VectorVariable("x", 2, lb=0, ub=100)

        prob = Problem().minimize(x[0] + x[1])
        prob.subject_to(np.array([[1, 0]]) @ x >= np.array([3.0]))
        prob.subject_to(np.array([[0, 1]]) @ x >= np.array([5.0]))

        sol = prob.solve()
        assert sol.status.value == "optimal"
        assert sol.objective_value == pytest.approx(8.0, abs=1e-8)

    def test_le_and_eq_matrix_constraints(self):
        """Mix of <= and == matrix constraints."""
        x = VectorVariable("x", 3, lb=0)

        prob = Problem().minimize(x[0] + x[1] + x[2])
        # x0 + x1 + x2 == 10
        prob.subject_to((np.array([[1, 1, 1]]) @ x).eq(np.array([10.0])))
        # x0 <= 3
        prob.subject_to(np.array([[1, 0, 0]]) @ x <= np.array([3.0]))

        sol = prob.solve()
        assert sol.status.value == "optimal"
        assert sol.objective_value == pytest.approx(10.0, abs=1e-8)


class TestLPDataExtraction:
    """Verify LP data extraction with matrix constraints."""

    def test_sparse_preserved_in_lp_data(self):
        """Sparse matrices are preserved (not densified) in LPData."""
        n = 100
        x = VectorVariable("x", n, lb=0)
        A_sparse = sp.eye(n, format="csr") * 2
        A = as_matrix(A_sparse)
        b = np.ones(n) * 10
        c = np.ones(n)

        prob = Problem().minimize(c @ x)
        prob.subject_to(A @ x <= b)

        extractor = LinearProgramExtractor()
        lp_data = extractor.extract(prob)

        assert sp.issparse(lp_data.A_ub)
        assert lp_data.A_ub.shape == (n, n)
        np.testing.assert_array_almost_equal(lp_data.b_ub, b)

    def test_dense_stays_dense(self):
        """Dense matrix constraints produce dense LPData."""
        x = VectorVariable("x", 3, lb=0)
        A = np.eye(3)
        b = np.ones(3) * 5
        c = np.ones(3)

        prob = Problem().minimize(c @ x)
        prob.subject_to(A @ x <= b)

        extractor = LinearProgramExtractor()
        lp_data = extractor.extract(prob)

        assert isinstance(lp_data.A_ub, np.ndarray)

    def test_n_constraints_with_matrix(self):
        """n_constraints counts matrix constraint rows."""
        x = VectorVariable("x", 5, lb=0)
        prob = Problem().minimize(np.ones(5) @ x)
        prob.subject_to(np.eye(5) @ x <= np.ones(5) * 10)
        prob.subject_to(x[0] >= 1)
        assert prob.n_constraints == 6  # 5 matrix + 1 expression

    def test_is_linear_with_matrix_constraints(self):
        """Matrix constraints don't affect linearity detection."""
        x = VectorVariable("x", 3, lb=0)
        prob = Problem().minimize(np.ones(3) @ x)
        prob.subject_to(np.eye(3) @ x <= np.ones(3))
        assert prob._is_linear_problem()


class TestLargeScale:
    """Performance-oriented tests for large-scale LPs."""

    def test_large_sparse_lp(self):
        """n=1000 sparse LP solves correctly."""
        rng = np.random.default_rng(123)
        n = 1000
        m = 500
        x = VectorVariable("x", n, lb=0, ub=100)
        A_sparse = sp.random(m, n, density=0.01, format="csr", random_state=rng)
        A = as_matrix(A_sparse)
        b = np.abs(A_sparse @ np.ones(n)) + 1
        c = rng.standard_normal(n)

        prob = Problem().minimize(c @ x)
        prob.subject_to(A @ x <= b)

        sol = prob.solve()
        assert sol.status.value == "optimal"

        # Verify against scipy
        bounds = [(0, 100)] * n
        ref = linprog(c, A_ub=A_sparse, b_ub=b, bounds=bounds, method="highs")
        assert ref.success
        assert sol.objective_value == pytest.approx(ref.fun, rel=1e-4)

    def test_warm_solve_uses_cache(self):
        """Second solve reuses LP cache."""
        x = VectorVariable("x", 10, lb=0)
        A = as_matrix(sp.eye(10, format="csr"))
        b = np.ones(10) * 5
        c = np.ones(10)

        prob = Problem().minimize(c @ x)
        prob.subject_to(A @ x <= b)

        sol1 = prob.solve()
        assert prob._lp_cache is not None
        sol2 = prob.solve()
        assert sol1.objective_value == sol2.objective_value
