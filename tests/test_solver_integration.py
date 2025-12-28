"""Tests for solver integration with VectorVariable and MatrixVariable."""

import numpy as np
import pytest

from optyx import (
    Variable,
    VectorVariable,
    MatrixVariable,
    Problem,
)


# =============================================================================
# VectorVariable Solver Integration Tests
# =============================================================================


class TestVectorVariableSolverIntegration:
    """Tests for VectorVariable with Problem.solve()."""

    def test_vector_minimize_sum_of_squares(self):
        """Minimize sum of squares with vector variable."""

        x = VectorVariable("x", 3, lb=-10, ub=10)
        # Minimize x[0]^2 + x[1]^2 + x[2]^2
        obj = sum(x[i] ** 2 for i in range(3))

        prob = Problem().minimize(obj)
        sol = prob.solve()

        assert sol.is_optimal
        # Optimal is x = [0, 0, 0]
        x_vals = sol[x]
        assert isinstance(x_vals, np.ndarray)
        assert x_vals.shape == (3,)
        np.testing.assert_array_almost_equal(x_vals, [0, 0, 0], decimal=3)

    def test_vector_with_equality_constraint(self):
        """Vector variable with sum constraint."""
        from optyx import Constraint

        x = VectorVariable("x", 3, lb=0, ub=10)
        # Minimize x[0]^2 + x[1]^2 + x[2]^2 subject to sum(x) = 3
        obj = sum(x[i] ** 2 for i in range(3))
        # Use Constraint() for equality
        constraint = Constraint(x[0] + x[1] + x[2] - 3, "==")

        prob = Problem().minimize(obj).subject_to(constraint)
        sol = prob.solve()

        assert sol.is_optimal
        x_vals = sol[x]
        # Optimal is x = [1, 1, 1] (equal distribution minimizes sum of squares)
        np.testing.assert_array_almost_equal(x_vals, [1, 1, 1], decimal=3)

    def test_vector_solution_access_by_name(self):
        """Can access vector elements by name string."""
        x = VectorVariable("x", 2, lb=0, ub=10)
        prob = Problem().minimize(x[0] + x[1])
        sol = prob.solve()

        assert sol.is_optimal
        # Access by name string
        assert sol["x[0]"] == pytest.approx(0.0, abs=1e-3)
        assert sol["x[1]"] == pytest.approx(0.0, abs=1e-3)

    def test_vector_solution_get_with_default(self):
        """Solution.get() works with VectorVariable."""
        x = VectorVariable("x", 2, lb=0, ub=10)
        prob = Problem().minimize(x[0] + x[1])
        sol = prob.solve()

        # Get vector with default
        result = sol.get(x)
        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_vector_deterministic_ordering(self):
        """Vector elements are ordered numerically, not lexicographically."""
        # Use more than 10 elements to test natural sorting
        x = VectorVariable("x", 12, lb=0, ub=1)
        # Make optimal solution have different values for each element
        # Minimize -sum(i * x[i]) to make x[i] = 1 for all i
        obj = sum(x[i] for i in range(12))

        prob = Problem().minimize(obj)
        variables = prob.variables

        # Verify natural sorting: x[0], x[1], ..., x[9], x[10], x[11]
        expected_order = [f"x[{i}]" for i in range(12)]
        actual_order = [v.name for v in variables]
        assert actual_order == expected_order

    def test_large_vector_variable(self):
        """Solver handles large vector variables."""
        n = 100
        x = VectorVariable("x", n, lb=0, ub=1)
        obj = sum(x[i] ** 2 for i in range(n))

        prob = Problem().minimize(obj)
        sol = prob.solve()

        assert sol.is_optimal
        x_vals = sol[x]
        assert x_vals.shape == (n,)
        # All should be 0
        np.testing.assert_array_almost_equal(x_vals, np.zeros(n), decimal=3)


# =============================================================================
# MatrixVariable Solver Integration Tests
# =============================================================================


class TestMatrixVariableSolverIntegration:
    """Tests for MatrixVariable with Problem.solve()."""

    def test_matrix_minimize_frobenius_norm(self):
        """Minimize Frobenius norm (sum of squares) of matrix."""
        A = MatrixVariable("A", 2, 2, lb=-10, ub=10)
        # Minimize sum of squares
        obj = sum(A[i, j] ** 2 for i in range(2) for j in range(2))

        prob = Problem().minimize(obj)
        sol = prob.solve()

        assert sol.is_optimal
        A_vals = sol[A]
        assert isinstance(A_vals, np.ndarray)
        assert A_vals.shape == (2, 2)
        np.testing.assert_array_almost_equal(A_vals, np.zeros((2, 2)), decimal=3)

    def test_matrix_with_trace_constraint(self):
        """Matrix variable with trace constraint."""
        from optyx import Constraint

        A = MatrixVariable("A", 2, 2, lb=0, ub=10)
        # Minimize sum of squares subject to trace = 2
        obj = sum(A[i, j] ** 2 for i in range(2) for j in range(2))
        # Use Constraint() for equality
        trace_constraint = Constraint(A[0, 0] + A[1, 1] - 2, "==")

        prob = Problem().minimize(obj).subject_to(trace_constraint)
        sol = prob.solve()

        assert sol.is_optimal
        A_vals = sol[A]
        # Optimal: diagonal = [1, 1], off-diagonal = [0, 0]
        assert A_vals[0, 0] == pytest.approx(1.0, abs=0.1)
        assert A_vals[1, 1] == pytest.approx(1.0, abs=0.1)
        assert A_vals[0, 1] == pytest.approx(0.0, abs=0.1)
        assert A_vals[1, 0] == pytest.approx(0.0, abs=0.1)

    def test_matrix_row_major_ordering(self):
        """Matrix elements are flattened in row-major order."""
        A = MatrixVariable("A", 2, 3, lb=0, ub=10)
        obj = sum(A[i, j] for i in range(2) for j in range(3))

        prob = Problem().minimize(obj)
        variables = prob.variables

        # Verify row-major order: A[0,0], A[0,1], A[0,2], A[1,0], A[1,1], A[1,2]
        expected_order = [
            "A[0,0]",
            "A[0,1]",
            "A[0,2]",
            "A[1,0]",
            "A[1,1]",
            "A[1,2]",
        ]
        actual_order = [v.name for v in variables]
        assert actual_order == expected_order

    def test_matrix_solution_access_by_name(self):
        """Can access matrix elements by name string."""
        A = MatrixVariable("A", 2, 2, lb=0, ub=10)
        obj = sum(A[i, j] for i in range(2) for j in range(2))

        prob = Problem().minimize(obj)
        sol = prob.solve()

        assert sol.is_optimal
        assert sol["A[0,0]"] == pytest.approx(0.0, abs=1e-3)
        assert sol["A[1,1]"] == pytest.approx(0.0, abs=1e-3)

    def test_symmetric_matrix(self):
        """Symmetric matrix works with solver."""
        S = MatrixVariable("S", 2, 2, lb=0, ub=10, symmetric=True)
        # S[0,1] and S[1,0] are the same variable
        obj = S[0, 0] ** 2 + S[1, 1] ** 2 + S[0, 1] ** 2

        prob = Problem().minimize(obj)
        sol = prob.solve()

        assert sol.is_optimal
        S_vals = sol[S]
        # Should be symmetric
        assert S_vals[0, 1] == pytest.approx(S_vals[1, 0], abs=1e-6)


# =============================================================================
# Mixed Variable Types Tests
# =============================================================================


class TestMixedVariableTypes:
    """Tests for problems with scalar, vector, and matrix variables."""

    def test_scalar_and_vector(self):
        """Problem with both scalar and vector variables."""
        t = Variable("t", lb=0, ub=10)
        x = VectorVariable("x", 3, lb=0, ub=10)

        # Minimize t + sum(x) subject to t >= x[i]
        obj = t + sum(x[i] for i in range(3))
        constraints = [t >= x[i] for i in range(3)]

        prob = Problem().minimize(obj).subject_to(constraints)
        sol = prob.solve()

        assert sol.is_optimal
        # Optimal: all = 0
        assert sol[t] == pytest.approx(0.0, abs=0.1)
        np.testing.assert_array_almost_equal(sol[x], [0, 0, 0], decimal=2)

    def test_vector_and_matrix(self):
        """Problem with both vector and matrix variables."""
        x = VectorVariable("x", 2, lb=0, ub=10)
        A = MatrixVariable("A", 2, 2, lb=0, ub=10)

        obj = sum(x[i] for i in range(2)) + sum(
            A[i, j] for i in range(2) for j in range(2)
        )

        prob = Problem().minimize(obj)
        sol = prob.solve()

        assert sol.is_optimal
        np.testing.assert_array_almost_equal(sol[x], [0, 0], decimal=3)
        np.testing.assert_array_almost_equal(sol[A], np.zeros((2, 2)), decimal=3)

    def test_variable_ordering_mixed(self):
        """Mixed variables are ordered correctly."""
        x = VectorVariable("x", 2)
        A = MatrixVariable("A", 2, 2)
        z = Variable("z")

        obj = z + sum(x[i] for i in range(2)) + A[0, 0]

        prob = Problem().minimize(obj)
        variables = prob.variables

        # Natural sort: A[0,0], A[0,1], ..., x[0], x[1], z
        names = [v.name for v in variables]
        # A comes before x, z comes after x
        assert names[0].startswith("A")
        assert "z" in names


# =============================================================================
# Solution Access Tests
# =============================================================================


class TestSolutionAccess:
    """Tests for Solution class access patterns."""

    def test_solution_getitem_scalar(self):
        """Solution[scalar_var] returns float."""
        x = Variable("x", lb=0, ub=10)
        prob = Problem().minimize(x)
        sol = prob.solve()

        assert isinstance(sol[x], float)

    def test_solution_getitem_vector(self):
        """Solution[vector_var] returns 1D array."""
        x = VectorVariable("x", 5, lb=0, ub=10)
        obj = sum(x[i] for i in range(5))
        prob = Problem().minimize(obj)
        sol = prob.solve()

        result = sol[x]
        assert isinstance(result, np.ndarray)
        assert result.ndim == 1
        assert result.shape == (5,)

    def test_solution_getitem_matrix(self):
        """Solution[matrix_var] returns 2D array."""
        A = MatrixVariable("A", 3, 4, lb=0, ub=10)
        obj = sum(A[i, j] for i in range(3) for j in range(4))
        prob = Problem().minimize(obj)
        sol = prob.solve()

        result = sol[A]
        assert isinstance(result, np.ndarray)
        assert result.ndim == 2
        assert result.shape == (3, 4)

    def test_solution_get_missing_returns_default(self):
        """Solution.get() returns default for missing variable."""
        x = Variable("x", lb=0, ub=10)
        prob = Problem().minimize(x)
        sol = prob.solve()

        # Create a variable not in the problem
        y = Variable("y")
        assert sol.get(y, default=-999.0) == -999.0

    def test_solution_getitem_string(self):
        """Solution[string_name] returns float."""
        x = VectorVariable("x", 2, lb=0, ub=10)
        obj = sum(x[i] for i in range(2))
        prob = Problem().minimize(obj)
        sol = prob.solve()

        assert isinstance(sol["x[0]"], float)
        assert isinstance(sol["x[1]"], float)


# =============================================================================
# Natural Sorting Tests
# =============================================================================


class TestNaturalSorting:
    """Tests for natural variable ordering."""

    def test_vector_natural_sort(self):
        """VectorVariable elements sort numerically."""
        x = VectorVariable("x", 15)  # x[0] through x[14]
        obj = sum(x[i] for i in range(15))
        prob = Problem().minimize(obj)

        names = [v.name for v in prob.variables]
        expected = [f"x[{i}]" for i in range(15)]
        assert names == expected

    def test_matrix_natural_sort(self):
        """MatrixVariable elements sort row-major numerically."""
        A = MatrixVariable("A", 3, 12)  # Columns 0-11
        obj = sum(A[i, j] for i in range(3) for j in range(12))
        prob = Problem().minimize(obj)

        names = [v.name for v in prob.variables]
        expected = [f"A[{i},{j}]" for i in range(3) for j in range(12)]
        assert names == expected

    def test_multiple_variables_natural_sort(self):
        """Multiple variables with indices sort correctly."""
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        obj = sum(x[i] for i in range(3)) + sum(y[i] for i in range(3))
        prob = Problem().minimize(obj)

        names = [v.name for v in prob.variables]
        # x comes before y alphabetically
        assert names == ["x[0]", "x[1]", "x[2]", "y[0]", "y[1]", "y[2]"]
