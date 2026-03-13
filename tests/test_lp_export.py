"""Tests for LP format export (Issue #106)."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from optyx import (
    BinaryVariable,
    IntegerVariable,
    Problem,
    Variable,
    VectorVariable,
    VariableDict,
    quadratic_form,
    sin,
    exp,
)
from optyx.io import write_lp, format_lp
from optyx.analysis import extract_quadratic_coefficients
from optyx.core.errors import InvalidOperationError, NoObjectiveError


# =============================================================================
# Test Problem.write() and Problem.to_lp()
# =============================================================================


class TestProblemWrite:
    """Tests for Problem.write() file output."""

    def test_write_creates_file(self, tmp_path):
        x = Variable("x", lb=0)
        prob = Problem("test")
        prob.minimize(x)
        filepath = str(tmp_path / "model.lp")
        prob.write(filepath)
        assert os.path.exists(filepath)

    def test_write_content_matches_to_lp(self, tmp_path):
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)
        prob = Problem("test")
        prob.minimize(x + y)
        prob.subject_to(x + y >= 1)
        filepath = str(tmp_path / "model.lp")
        prob.write(filepath)
        with open(filepath) as f:
            content = f.read()
        assert content == prob.to_lp()

    def test_to_lp_returns_string(self):
        x = Variable("x", lb=0)
        prob = Problem()
        prob.minimize(x)
        result = prob.to_lp()
        assert isinstance(result, str)
        assert "Minimize" in result
        assert "End" in result


# =============================================================================
# Test LP Format Structure
# =============================================================================


class TestLPFormatStructure:
    """Tests for the overall LP format structure."""

    def test_model_name_in_comment(self):
        prob = Problem("my_model")
        prob.minimize(Variable("x", lb=0))
        lp = prob.to_lp()
        assert lp.startswith("\\ Model my_model")

    def test_default_model_name(self):
        prob = Problem()
        prob.minimize(Variable("x", lb=0))
        lp = prob.to_lp()
        assert "\\ Model optyx_model" in lp

    def test_sections_order(self):
        x = Variable("x", lb=0)
        y = IntegerVariable("y", lb=0, ub=5)
        prob = Problem()
        prob.minimize(x + y)
        prob.subject_to(x + y >= 1)
        lp = prob.to_lp()
        lines = lp.split("\n")

        # Find section positions
        minimize_idx = next(i for i, l in enumerate(lines) if l.strip() == "Minimize")
        subject_idx = next(i for i, l in enumerate(lines) if l.strip() == "Subject To")
        bounds_idx = next(i for i, l in enumerate(lines) if l.strip() == "Bounds")
        generals_idx = next(i for i, l in enumerate(lines) if l.strip() == "Generals")
        end_idx = next(i for i, l in enumerate(lines) if l.strip() == "End")

        assert minimize_idx < subject_idx < bounds_idx < generals_idx < end_idx

    def test_end_keyword(self):
        prob = Problem()
        prob.minimize(Variable("x", lb=0))
        lp = prob.to_lp()
        assert lp.strip().endswith("End")


# =============================================================================
# Test Linear Objectives
# =============================================================================


class TestLinearObjective:
    """Tests for linear objective formatting."""

    def test_minimize_single_variable(self):
        x = Variable("x", lb=0)
        prob = Problem()
        prob.minimize(x)
        lp = prob.to_lp()
        assert "Minimize" in lp
        assert "obj: x" in lp

    def test_maximize(self):
        x = Variable("x", lb=0)
        prob = Problem()
        prob.maximize(x)
        lp = prob.to_lp()
        assert "Maximize" in lp

    def test_coefficient_formatting(self):
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)
        prob = Problem()
        prob.minimize(2 * x + 3 * y)
        lp = prob.to_lp()
        assert "2 x" in lp
        assert "3 y" in lp

    def test_negative_coefficient(self):
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)
        prob = Problem()
        prob.minimize(x - 2 * y)
        lp = prob.to_lp()
        assert "- 2 y" in lp

    def test_unit_coefficient_omitted(self):
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)
        prob = Problem()
        prob.minimize(x + y)
        lp = prob.to_lp()
        # "1 x" should not appear — just "x"
        assert "1 x" not in lp
        assert "obj: x + y" in lp

    def test_negative_unit_coefficient(self):
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)
        prob = Problem()
        prob.minimize(x - y)
        lp = prob.to_lp()
        assert "obj: x - y" in lp

    def test_vector_sum_objective(self):
        x = VectorVariable("x", 3, lb=0)
        prob = Problem()
        prob.minimize(x.sum())
        lp = prob.to_lp()
        assert "x[0]" in lp
        assert "x[1]" in lp
        assert "x[2]" in lp

    def test_linear_combination_objective(self):
        x = VectorVariable("x", 3, lb=0)
        c = np.array([1.0, 2.0, 3.0])
        prob = Problem()
        prob.minimize(c @ x)
        lp = prob.to_lp()
        assert "x[0]" in lp
        assert "2 x[1]" in lp
        assert "3 x[2]" in lp


# =============================================================================
# Test Quadratic Objectives
# =============================================================================


class TestQuadraticObjective:
    """Tests for quadratic objective formatting."""

    def test_simple_quadratic(self):
        x = Variable("x", lb=0)
        prob = Problem()
        prob.minimize(x**2)
        lp = prob.to_lp()
        assert "[" in lp
        assert "] / 2" in lp
        assert "x ^2" in lp

    def test_cross_term(self):
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)
        prob = Problem()
        prob.minimize(x * y)
        lp = prob.to_lp()
        assert "x * y" in lp

    def test_mixed_linear_quadratic(self):
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)
        prob = Problem()
        prob.minimize(x + y + x**2 + x * y + y**2)
        lp = prob.to_lp()
        # Should have both linear and quadratic parts
        assert "x + y" in lp or ("x" in lp and "y" in lp)
        assert "[" in lp
        assert "] / 2" in lp

    def test_dot_product_self(self):
        x = VectorVariable("x", 3, lb=0)
        prob = Problem()
        prob.minimize(x.dot(x))
        lp = prob.to_lp()
        assert "x[0] ^2" in lp
        assert "x[1] ^2" in lp
        assert "x[2] ^2" in lp

    def test_quadratic_form(self):
        x = VectorVariable("x", 2, lb=0)
        Q = np.array([[2.0, 1.0], [1.0, 3.0]])
        prob = Problem()
        prob.minimize(quadratic_form(x, Q))
        lp = prob.to_lp()
        assert "[" in lp
        assert "x[0] ^2" in lp
        assert "x[0] * x[1]" in lp
        assert "x[1] ^2" in lp

    def test_quadratic_form_coefficients(self):
        """Verify quadratic coefficients are correct in LP format."""
        x = VectorVariable("x", 2, lb=0)
        Q = np.array([[2.0, 0.5], [0.5, 3.0]])
        prob = Problem()
        prob.minimize(quadratic_form(x, Q))
        lp = prob.to_lp()
        # LP format: [ 2*coeff terms ] / 2
        # x[0]^2 coeff = 2.0, doubled = 4.0
        assert "4 x[0] ^2" in lp
        # x[0]*x[1] coeff = 0.5+0.5 = 1.0, doubled = 2.0
        assert "2 x[0] * x[1]" in lp
        # x[1]^2 coeff = 3.0, doubled = 6.0
        assert "6 x[1] ^2" in lp


# =============================================================================
# Test Constraints
# =============================================================================


class TestConstraints:
    """Tests for constraint formatting."""

    def test_le_constraint(self):
        x = Variable("x", lb=0)
        prob = Problem()
        prob.minimize(x)
        prob.subject_to(x <= 10)
        lp = prob.to_lp()
        assert "<=" in lp
        assert "10" in lp

    def test_ge_constraint(self):
        x = Variable("x", lb=0)
        prob = Problem()
        prob.minimize(x)
        prob.subject_to(x >= 1)
        lp = prob.to_lp()
        assert ">=" in lp
        assert "1" in lp

    def test_eq_constraint(self):
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)
        prob = Problem()
        prob.minimize(x + y)
        prob.subject_to((x + y).eq(1))
        lp = prob.to_lp()
        assert "==" in lp

    def test_multiple_constraints(self):
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)
        prob = Problem()
        prob.minimize(x + y)
        prob.subject_to(x + y >= 1)
        prob.subject_to(x - y <= 5)
        prob.subject_to((x + y).eq(3))
        lp = prob.to_lp()
        assert "c0:" in lp
        assert "c1:" in lp
        assert "c2:" in lp

    def test_named_constraint(self):
        from optyx.constraints import Constraint

        x = Variable("x", lb=0)
        y = Variable("y", lb=0)
        prob = Problem()
        prob.minimize(x + y)
        c = Constraint(expr=x + y - 1, sense=">=", name="demand")
        prob.subject_to(c)
        lp = prob.to_lp()
        assert "demand:" in lp

    def test_no_constraints(self):
        x = Variable("x", lb=0)
        prob = Problem()
        prob.minimize(x)
        lp = prob.to_lp()
        assert "Subject To" not in lp

    def test_constraint_with_constant(self):
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)
        prob = Problem()
        prob.minimize(x)
        prob.subject_to(2 * x + 3 * y >= 6)
        lp = prob.to_lp()
        assert "2 x" in lp
        assert "3 y" in lp
        assert ">=" in lp
        assert "6" in lp


# =============================================================================
# Test Matrix Constraints
# =============================================================================


class TestMatrixConstraints:
    """Tests for matrix constraint formatting."""

    def test_dense_matrix_constraints(self):
        x = VectorVariable("x", 3, lb=0)
        A = np.array([[1, 2, 3], [4, 5, 6]])
        b = np.array([10, 20])
        prob = Problem()
        prob.minimize(x.sum())
        prob.subject_to_matrix(A, x, "<=", b)
        lp = prob.to_lp()
        assert "<=" in lp
        assert "10" in lp
        assert "20" in lp

    def test_sparse_matrix_constraints(self):
        from scipy import sparse as sp

        x = VectorVariable("x", 5, lb=0)
        A = sp.csr_matrix(np.array([[1, 0, 0, 2, 0], [0, 3, 0, 0, 4]]))
        b = np.array([5, 6])
        prob = Problem()
        prob.minimize(x.sum())
        prob.subject_to_matrix(A, x, ">=", b)
        lp = prob.to_lp()
        assert ">=" in lp
        # Zero coefficients should be omitted
        lines = lp.split("\n")
        constraint_lines = [l for l in lines if l.strip().startswith("c")]
        # First constraint should have x[0] and x[3]
        assert "x[0]" in constraint_lines[0]
        assert "x[3]" in constraint_lines[0]

    def test_mixed_expression_and_matrix_constraints(self):
        x = VectorVariable("x", 3, lb=0)
        A = np.array([[1, 1, 1]])
        b = np.array([10])
        prob = Problem()
        prob.minimize(x.sum())
        prob.subject_to(x[0] >= 1)
        prob.subject_to_matrix(A, x, "<=", b)
        lp = prob.to_lp()
        assert "c0:" in lp
        assert "c1:" in lp


# =============================================================================
# Test Bounds
# =============================================================================


class TestBounds:
    """Tests for variable bounds formatting."""

    def test_lower_bound_only(self):
        x = Variable("x", lb=0)
        prob = Problem()
        prob.minimize(x)
        lp = prob.to_lp()
        assert "0 <= x" in lp

    def test_upper_bound_only(self):
        x = Variable("x", ub=10)
        prob = Problem()
        prob.minimize(x)
        lp = prob.to_lp()
        assert "-inf <= x <= 10" in lp

    def test_both_bounds(self):
        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.minimize(x)
        lp = prob.to_lp()
        assert "0 <= x <= 10" in lp

    def test_free_variable(self):
        x = Variable("x")
        prob = Problem()
        prob.minimize(x)
        lp = prob.to_lp()
        assert "x free" in lp

    def test_binary_bounds(self):
        b = BinaryVariable("b")
        prob = Problem()
        prob.minimize(b)
        lp = prob.to_lp()
        assert "0 <= b <= 1" in lp


# =============================================================================
# Test Variable Types (Generals / Binaries)
# =============================================================================


class TestVariableTypes:
    """Tests for integer and binary variable type sections."""

    def test_integer_variable(self):
        x = IntegerVariable("x", lb=0, ub=10)
        prob = Problem()
        prob.minimize(x)
        lp = prob.to_lp()
        assert "Generals" in lp
        assert "  x" in lp

    def test_binary_variable(self):
        b = BinaryVariable("b")
        prob = Problem()
        prob.minimize(b)
        lp = prob.to_lp()
        assert "Binaries" in lp
        assert "  b" in lp

    def test_mixed_variable_types(self):
        x = Variable("x", lb=0)
        y = IntegerVariable("y", lb=0, ub=5)
        z = BinaryVariable("z")
        prob = Problem()
        prob.minimize(x + y + z)
        lp = prob.to_lp()
        assert "Generals" in lp
        assert "Binaries" in lp
        # x should not appear in Generals or Binaries
        generals_idx = lp.index("Generals")
        binaries_idx = lp.index("Binaries")
        generals_section = lp[generals_idx:binaries_idx]
        assert "  y" in generals_section
        binaries_section = lp[binaries_idx:]
        assert "  z" in binaries_section

    def test_no_generals_or_binaries_for_continuous(self):
        x = Variable("x", lb=0)
        prob = Problem()
        prob.minimize(x)
        lp = prob.to_lp()
        assert "Generals" not in lp
        assert "Binaries" not in lp

    def test_vector_integer_variables(self):
        x = VectorVariable("x", 3, lb=0, ub=10, domain="integer")
        prob = Problem()
        prob.minimize(x.sum())
        lp = prob.to_lp()
        assert "Generals" in lp
        assert "x[0]" in lp
        assert "x[1]" in lp
        assert "x[2]" in lp

    def test_vector_binary_variables(self):
        b = VectorVariable("b", 4, domain="binary")
        prob = Problem()
        prob.minimize(b.sum())
        lp = prob.to_lp()
        assert "Binaries" in lp


# =============================================================================
# Test Error Cases
# =============================================================================


class TestErrors:
    """Tests for error handling."""

    def test_no_objective_raises(self):
        prob = Problem()
        prob.subject_to(Variable("x", lb=0) >= 0)
        with pytest.raises(NoObjectiveError):
            prob.to_lp()

    def test_nonlinear_objective_raises(self):
        x = Variable("x", lb=0)
        prob = Problem()
        prob.minimize(sin(x))
        with pytest.raises(InvalidOperationError, match="LP format only supports"):
            prob.to_lp()

    def test_nonlinear_constraint_raises(self):
        x = Variable("x", lb=0)
        prob = Problem()
        prob.minimize(x)
        prob.subject_to(sin(x) <= 1)
        with pytest.raises(InvalidOperationError, match="not linear"):
            prob.to_lp()

    def test_exp_objective_raises(self):
        x = Variable("x", lb=0)
        prob = Problem()
        prob.minimize(exp(x))
        with pytest.raises(InvalidOperationError):
            prob.to_lp()


# =============================================================================
# Test extract_quadratic_coefficients utility
# =============================================================================


class TestExtractQuadraticCoefficients:
    """Tests for the extract_quadratic_coefficients analysis utility."""

    def test_simple_square(self):
        x = Variable("x")
        Q = extract_quadratic_coefficients(x**2, [x])
        assert Q.shape == (1, 1)
        assert Q[0, 0] == pytest.approx(1.0)

    def test_cross_term(self):
        x = Variable("x")
        y = Variable("y")
        Q = extract_quadratic_coefficients(x * y, [x, y])
        assert Q.shape == (2, 2)
        # Symmetric: Q[0,1] = Q[1,0] = 0.5
        assert Q[0, 1] == pytest.approx(0.5)
        assert Q[1, 0] == pytest.approx(0.5)

    def test_quadratic_form_extraction(self):
        x = VectorVariable("x", 2)
        Q_in = np.array([[2.0, 1.0], [1.0, 3.0]])
        expr = quadratic_form(x, Q_in)
        Q_out = extract_quadratic_coefficients(expr, list(x._variables))
        np.testing.assert_array_almost_equal(Q_out, Q_in)

    def test_dot_product_self(self):
        x = VectorVariable("x", 3)
        expr = x.dot(x)
        Q = extract_quadratic_coefficients(expr, list(x._variables))
        np.testing.assert_array_almost_equal(Q, np.eye(3))

    def test_mixed_linear_quadratic_extracts_only_quadratic(self):
        x = Variable("x")
        y = Variable("y")
        expr = x**2 + 3 * x + 2 * y
        Q = extract_quadratic_coefficients(expr, [x, y])
        assert Q[0, 0] == pytest.approx(1.0)
        assert Q[0, 1] == pytest.approx(0.0)
        assert Q[1, 1] == pytest.approx(0.0)

    def test_nonlinear_raises(self):
        x = Variable("x")
        from optyx.core.errors import NonLinearError
        with pytest.raises(NonLinearError):
            extract_quadratic_coefficients(sin(x), [x])

    def test_symmetry(self):
        x = Variable("x")
        y = Variable("y")
        Q = extract_quadratic_coefficients(2 * x * y + x**2, [x, y])
        np.testing.assert_array_almost_equal(Q, Q.T)


# =============================================================================
# Test VariableDict support
# =============================================================================


class TestVariableDictSupport:
    """Tests for VariableDict LP export."""

    def test_variable_dict_lp(self):
        buy = VariableDict("buy", ["ham", "egg"], lb=0)
        prob = Problem("diet")
        prob.minimize(buy["ham"] + 2 * buy["egg"])
        prob.subject_to(buy["ham"] + buy["egg"] >= 1)
        lp = prob.to_lp()
        assert "buy[ham]" in lp
        assert "buy[egg]" in lp

    def test_variable_dict_integer(self):
        x = VariableDict("x", ["a", "b"], lb=0, ub=10, domain="integer")
        prob = Problem()
        prob.minimize(x["a"] + x["b"])
        lp = prob.to_lp()
        assert "Generals" in lp


# =============================================================================
# Test Round-Trip Accuracy
# =============================================================================


class TestRoundTrip:
    """Tests verifying LP export matches the original model."""

    def test_lp_objective_coefficients(self):
        """Verify exported coefficients match the model."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)
        z = Variable("z", lb=0)
        prob = Problem()
        prob.minimize(5 * x - 3 * y + 7 * z)
        lp = prob.to_lp()
        assert "5 x" in lp
        assert "- 3 y" in lp
        assert "+ 7 z" in lp

    def test_constraint_rhs(self):
        """Verify constraint RHS values."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)
        prob = Problem()
        prob.minimize(x + y)
        prob.subject_to(x + y >= 10)
        prob.subject_to(x - y <= 3)
        lp = prob.to_lp()
        assert ">= 10" in lp
        assert "<= 3" in lp

    def test_write_and_read_back(self, tmp_path):
        """Verify file can be written and read back."""
        x = Variable("x", lb=0, ub=100)
        y = Variable("y", lb=-5, ub=50)
        prob = Problem("roundtrip")
        prob.minimize(2 * x + 3 * y)
        prob.subject_to(x + y >= 1)
        prob.subject_to(x - y <= 10)

        filepath = str(tmp_path / "roundtrip.lp")
        prob.write(filepath)

        with open(filepath) as f:
            content = f.read()

        assert "\\ Model roundtrip" in content
        assert "Minimize" in content
        assert "Subject To" in content
        assert "Bounds" in content
        assert "End" in content

    def test_maximize_objective(self):
        x = Variable("x", lb=0, ub=10)
        y = Variable("y", lb=0, ub=10)
        prob = Problem()
        prob.maximize(3 * x + 5 * y)
        prob.subject_to(x + y <= 10)
        lp = prob.to_lp()
        assert "Maximize" in lp
        assert "3 x" in lp
        assert "5 y" in lp
