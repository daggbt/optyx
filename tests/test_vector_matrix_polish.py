"""Tests for vector/matrix polishing features (Issue #99).

Tests cover:
1. MatrixVariable.diagonal(offset=k) — main, super, sub-diagonals
2. Per-element bounds arrays on VectorVariable constructor
3. Fancy indexing x[[0, 2]] on VectorVariable
"""

from __future__ import annotations

import numpy as np
import pytest

from optyx import Problem, Variable, VectorVariable
from optyx.core.matrices import MatrixVariable


# ============================================================
# 1. MatrixVariable.diagonal(offset=k)
# ============================================================


class TestMatrixDiagonal:
    """Tests for MatrixVariable.diagonal() with offset."""

    def test_main_diagonal(self):
        """diagonal() with default offset returns main diagonal."""
        A = MatrixVariable("A", 3, 3)
        d = A.diagonal()
        assert isinstance(d, VectorVariable)
        assert len(d) == 3
        assert d[0] is A[0, 0]
        assert d[1] is A[1, 1]
        assert d[2] is A[2, 2]

    def test_main_diagonal_explicit_zero(self):
        """diagonal(0) same as diagonal()."""
        A = MatrixVariable("A", 3, 3)
        d = A.diagonal(0)
        assert len(d) == 3
        assert d[0] is A[0, 0]
        assert d[1] is A[1, 1]
        assert d[2] is A[2, 2]

    def test_super_diagonal(self):
        """diagonal(1) returns first super-diagonal."""
        A = MatrixVariable("A", 3, 3)
        d = A.diagonal(1)
        assert len(d) == 2
        assert d[0] is A[0, 1]
        assert d[1] is A[1, 2]

    def test_sub_diagonal(self):
        """diagonal(-1) returns first sub-diagonal."""
        A = MatrixVariable("A", 3, 3)
        d = A.diagonal(-1)
        assert len(d) == 2
        assert d[0] is A[1, 0]
        assert d[1] is A[2, 1]

    def test_super_diagonal_offset_2(self):
        """diagonal(2) on 4x4 returns 2 elements."""
        A = MatrixVariable("A", 4, 4)
        d = A.diagonal(2)
        assert len(d) == 2
        assert d[0] is A[0, 2]
        assert d[1] is A[1, 3]

    def test_sub_diagonal_offset_minus2(self):
        """diagonal(-2) on 4x4 returns 2 elements."""
        A = MatrixVariable("A", 4, 4)
        d = A.diagonal(-2)
        assert len(d) == 2
        assert d[0] is A[2, 0]
        assert d[1] is A[3, 1]

    def test_single_element_diagonal(self):
        """diagonal with max offset returns single element."""
        A = MatrixVariable("A", 3, 3)
        d = A.diagonal(2)
        assert len(d) == 1
        assert d[0] is A[0, 2]

    def test_rectangular_main_diagonal(self):
        """diagonal() on non-square matrix."""
        A = MatrixVariable("A", 2, 4)
        d = A.diagonal()
        assert len(d) == 2  # min(2, 4)
        assert d[0] is A[0, 0]
        assert d[1] is A[1, 1]

    def test_rectangular_tall_diagonal(self):
        """diagonal() on tall matrix."""
        A = MatrixVariable("A", 4, 2)
        d = A.diagonal()
        assert len(d) == 2  # min(4, 2)
        assert d[0] is A[0, 0]
        assert d[1] is A[1, 1]

    def test_out_of_bounds_offset_raises(self):
        """diagonal with too-large offset raises error."""
        A = MatrixVariable("A", 3, 3)
        with pytest.raises(Exception):
            A.diagonal(3)

    def test_out_of_bounds_negative_offset_raises(self):
        """diagonal with too-negative offset raises error."""
        A = MatrixVariable("A", 3, 3)
        with pytest.raises(Exception):
            A.diagonal(-3)

    def test_diagonal_in_constraint(self):
        """diagonal can be used in optimization constraints."""
        A = MatrixVariable("A", 2, 2, lb=0, ub=10)
        prob = Problem()
        diag = A.diagonal()
        prob.minimize(diag.sum())
        prob.subject_to(diag[0] >= 1)
        prob.subject_to(diag[1] >= 2)
        sol = prob.solve()
        assert sol.is_optimal
        assert abs(sol[A[0, 0]] - 1.0) < 1e-6
        assert abs(sol[A[1, 1]] - 2.0) < 1e-6

    def test_symmetric_diagonal(self):
        """diagonal works on symmetric matrices."""
        A = MatrixVariable("A", 3, 3, symmetric=True)
        d = A.diagonal()
        assert len(d) == 3


# ============================================================
# 2. Per-Element Bounds Arrays on VectorVariable
# ============================================================


class TestPerElementBounds:
    """Tests for VectorVariable with per-element lb/ub arrays."""

    def test_numpy_array_bounds(self):
        """Accept numpy arrays for lb and ub."""
        lb = np.array([0.0, 0.5, 0.2])
        ub = np.array([1.0, 2.0, 1.5])
        x = VectorVariable("x", 3, lb=lb, ub=ub)
        assert x[0].lb == 0.0
        assert x[0].ub == 1.0
        assert x[1].lb == 0.5
        assert x[1].ub == 2.0
        assert x[2].lb == 0.2
        assert x[2].ub == 1.5

    def test_list_bounds(self):
        """Accept plain lists for lb and ub."""
        x = VectorVariable("x", 3, lb=[1.0, 2.0, 3.0], ub=[10.0, 20.0, 30.0])
        assert x[0].lb == 1.0
        assert x[1].lb == 2.0
        assert x[2].lb == 3.0
        assert x[0].ub == 10.0
        assert x[1].ub == 20.0
        assert x[2].ub == 30.0

    def test_scalar_bounds_still_work(self):
        """Scalar bounds apply to all elements."""
        x = VectorVariable("x", 3, lb=0.0, ub=1.0)
        for i in range(3):
            assert x[i].lb == 0.0
            assert x[i].ub == 1.0

    def test_none_bounds_still_work(self):
        """None bounds leave variables unbounded."""
        x = VectorVariable("x", 3)
        for i in range(3):
            assert x[i].lb is None
            assert x[i].ub is None

    def test_mixed_scalar_and_array_bounds(self):
        """Scalar lb with array ub, and vice versa."""
        x = VectorVariable("x", 3, lb=0.0, ub=[10.0, 20.0, 30.0])
        assert x[0].lb == 0.0
        assert x[0].ub == 10.0
        assert x[1].lb == 0.0
        assert x[1].ub == 20.0
        assert x[2].lb == 0.0
        assert x[2].ub == 30.0

    def test_wrong_size_bounds_raises(self):
        """Mismatched bounds array size raises error."""
        with pytest.raises(Exception):
            VectorVariable("x", 3, lb=[0.0, 1.0])  # size 2 != 3

    def test_wrong_size_ub_raises(self):
        """Mismatched ub array size raises error."""
        with pytest.raises(Exception):
            VectorVariable("x", 3, ub=np.array([1.0, 2.0, 3.0, 4.0]))  # size 4 != 3

    def test_per_element_bounds_solve(self):
        """Per-element bounds are respected by the solver."""
        lb = np.array([1.0, 2.0, 3.0])
        ub = np.array([10.0, 20.0, 30.0])
        x = VectorVariable("x", 3, lb=lb, ub=ub)

        prob = Problem()
        prob.minimize(x.sum())
        sol = prob.solve()

        assert sol.is_optimal
        # Minimum should be at lower bounds
        assert abs(sol[x[0]] - 1.0) < 1e-6
        assert abs(sol[x[1]] - 2.0) < 1e-6
        assert abs(sol[x[2]] - 3.0) < 1e-6

    def test_per_element_bounds_maximize(self):
        """Per-element bounds work with maximize."""
        ub = np.array([5.0, 10.0, 15.0])
        x = VectorVariable("x", 3, lb=0.0, ub=ub)

        prob = Problem()
        prob.maximize(x.sum())
        sol = prob.solve()

        assert sol.is_optimal
        assert abs(sol[x[0]] - 5.0) < 1e-6
        assert abs(sol[x[1]] - 10.0) < 1e-6
        assert abs(sol[x[2]] - 15.0) < 1e-6

    def test_per_element_bounds_nlp(self):
        """Per-element bounds in NLP context."""
        lb = np.array([0.0, -1.0])
        ub = np.array([2.0, 3.0])
        x = VectorVariable("x", 2, lb=lb, ub=ub)

        prob = Problem()
        prob.minimize(x.dot(x))  # min x0^2 + x1^2
        sol = prob.solve()

        assert sol.is_optimal
        # Minimum of sum of squares with these bounds is at (0, 0)
        assert abs(sol[x[0]] - 0.0) < 1e-4
        assert abs(sol[x[1]] - 0.0) < 1e-4


# ============================================================
# 3. Fancy Indexing on VectorVariable
# ============================================================


class TestFancyIndexing:
    """Tests for fancy indexing on VectorVariable."""

    def test_list_indexing(self):
        """x[[0, 2]] returns VectorVariable with those elements."""
        x = VectorVariable("x", 5, lb=0, ub=10)
        sub = x[[0, 2]]
        assert isinstance(sub, VectorVariable)
        assert len(sub) == 2
        assert sub[0] is x[0]
        assert sub[1] is x[2]

    def test_list_indexing_order(self):
        """Fancy indexing preserves the requested order."""
        x = VectorVariable("x", 5)
        sub = x[[4, 1, 3]]
        assert len(sub) == 3
        assert sub[0] is x[4]
        assert sub[1] is x[1]
        assert sub[2] is x[3]

    def test_numpy_array_indexing(self):
        """x[np.array([0, 2, 4])] works."""
        x = VectorVariable("x", 5)
        sub = x[np.array([0, 2, 4])]
        assert isinstance(sub, VectorVariable)
        assert len(sub) == 3
        assert sub[0] is x[0]
        assert sub[1] is x[2]
        assert sub[2] is x[4]

    def test_tuple_indexing(self):
        """x[(0, 2)] works as fancy indexing."""
        x = VectorVariable("x", 5)
        sub = x[(0, 2)]
        assert isinstance(sub, VectorVariable)
        assert len(sub) == 2

    def test_boolean_indexing(self):
        """Boolean array indexing works."""
        x = VectorVariable("x", 4)
        mask = np.array([True, False, True, False])
        sub = x[mask]
        assert isinstance(sub, VectorVariable)
        assert len(sub) == 2
        assert sub[0] is x[0]
        assert sub[1] is x[2]

    def test_negative_fancy_indexing(self):
        """Negative indices work in fancy indexing."""
        x = VectorVariable("x", 5)
        sub = x[[-1, -3]]
        assert len(sub) == 2
        assert sub[0] is x[4]
        assert sub[1] is x[2]

    def test_single_element_fancy(self):
        """Fancy index with single element returns VectorVariable."""
        x = VectorVariable("x", 5)
        sub = x[[3]]
        assert isinstance(sub, VectorVariable)
        assert len(sub) == 1
        assert sub[0] is x[3]

    def test_empty_fancy_raises(self):
        """Empty fancy index raises error."""
        x = VectorVariable("x", 5)
        with pytest.raises(IndexError):
            x[[]]

    def test_out_of_range_fancy_raises(self):
        """Out-of-range fancy index raises error."""
        x = VectorVariable("x", 5)
        with pytest.raises(IndexError):
            x[[0, 10]]

    def test_boolean_wrong_size_raises(self):
        """Boolean mask with wrong size raises error."""
        x = VectorVariable("x", 4)
        with pytest.raises(IndexError):
            x[np.array([True, False])]

    def test_fancy_indexing_in_optimization(self):
        """Fancy-indexed sub-vector can be used in constraints."""
        x = VectorVariable("x", 5, lb=0, ub=10)
        prob = Problem()
        prob.minimize(x.sum())

        # Constrain only elements 0, 2, 4
        selected = x[[0, 2, 4]]
        for i in range(len(selected)):
            prob.subject_to(selected[i] >= 3)

        sol = prob.solve()
        assert sol.is_optimal
        # x[0], x[2], x[4] >= 3, others at lb=0
        assert abs(sol[x[0]] - 3.0) < 1e-6
        assert abs(sol[x[1]] - 0.0) < 1e-6
        assert abs(sol[x[2]] - 3.0) < 1e-6
        assert abs(sol[x[3]] - 0.0) < 1e-6
        assert abs(sol[x[4]] - 3.0) < 1e-6

    def test_fancy_indexing_with_per_element_bounds(self):
        """Fancy indexing combined with per-element bounds."""
        lb = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        x = VectorVariable("x", 5, lb=lb, ub=100.0)

        sub = x[[1, 3]]
        assert sub[0] is x[1]
        assert sub[1] is x[3]

        # Each element retains its per-element bound
        assert sub[0].lb == 2.0
        assert sub[1].lb == 4.0

    def test_fancy_indexing_nlp(self):
        """Fancy indexing works in NLP solve context."""
        x = VectorVariable("x", 4, lb=-10, ub=10)
        prob = Problem()

        # Only penalize elements 0 and 2
        sub = x[[0, 2]]
        prob.minimize(sub.dot(sub))

        sol = prob.solve()
        assert sol.is_optimal
        assert abs(sol[x[0]]) < 1e-4
        assert abs(sol[x[2]]) < 1e-4


# ============================================================
# 4. Integration: combining features
# ============================================================


class TestCombinedFeatures:
    """Tests combining multiple features together."""

    def test_diagonal_with_per_element_bounds(self):
        """MatrixVariable diagonal works when underlying vars have bounds."""
        A = MatrixVariable("A", 3, 3, lb=0, ub=10)
        d = A.diagonal()
        prob = Problem()
        prob.minimize(d.sum())
        prob.subject_to(d[0] >= 1)
        sol = prob.solve()
        assert sol.is_optimal
        assert abs(sol[A[0, 0]] - 1.0) < 1e-6
        assert abs(sol[A[1, 1]] - 0.0) < 1e-6
        assert abs(sol[A[2, 2]] - 0.0) < 1e-6

    def test_diagonal_fancy_index(self):
        """Can fancy-index from a diagonal."""
        A = MatrixVariable("A", 4, 4)
        d = A.diagonal()
        sub = d[[0, 3]]
        assert sub[0] is A[0, 0]
        assert sub[1] is A[3, 3]
