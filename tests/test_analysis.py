from optyx.core.expressions import Variable
from optyx.core.errors import NonLinearError, NoObjectiveError
from optyx.constraints import Constraint
from optyx.problem import Problem
from optyx.analysis import (
    is_linear,
    is_quadratic,
    compute_degree,
    clear_degree_cache,
    extract_linear_coefficient,
    extract_constant_term,
    LinearProgramExtractor,
    is_simple_bound,
    classify_constraints,
)
import numpy as np
import pytest


def test_is_linear_simple():
    x = Variable("x")
    y = Variable("y")
    expr = 2 * x + 3 * y

    assert is_linear(expr)
    assert compute_degree(expr) == 1


def test_is_linear_nonlinear():
    x = Variable("x")
    expr = x**2

    assert not is_linear(expr)
    assert compute_degree(expr) == 2


def test_is_quadratic_simple():
    x = Variable("x")
    y = Variable("y")
    expr = x**2 + y

    assert is_quadratic(expr)
    assert compute_degree(expr) == 2


def test_is_quadratic_cubic():
    x = Variable("x")
    expr = x**3

    assert not is_quadratic(expr)
    assert compute_degree(expr) == 3


def test_is_linear_with_constants():
    x = Variable("x")
    expr = 2 * x + 5

    assert is_linear(expr)
    assert compute_degree(expr) == 1


def test_is_linear_with_products():
    x = Variable("x")
    y = Variable("y")
    expr = x * y

    assert not is_linear(expr)
    assert compute_degree(expr) is None


def test_division_by_constant_is_linear():
    x = Variable("x")
    expr = x / 2

    assert is_linear(expr)
    assert compute_degree(expr) == 1


def test_division_by_variable_not_polynomial():
    x = Variable("x")
    y = Variable("y")
    expr = x / y

    assert not is_linear(expr)
    assert compute_degree(expr) is None


# --- Performance optimization tests ---


def test_negation_preserves_degree():
    """Negation should preserve polynomial degree."""
    x = Variable("x")
    expr = -x
    assert is_linear(expr)
    assert compute_degree(expr) == 1

    expr2 = -(x**2)
    assert is_quadratic(expr2)
    assert compute_degree(expr2) == 2


def test_deeply_nested_expression():
    """Test early termination on deeply nested expressions."""
    x = Variable("x")
    # Build a deep chain: ((((x + x) + x) + x) + ...)
    expr = x
    for _ in range(100):
        expr = expr + x

    assert is_linear(expr)
    assert compute_degree(expr) == 1


def test_early_termination_on_nonpolynomial():
    """Non-polynomial at root should terminate immediately."""
    x = Variable("x")
    y = Variable("y")
    # x / y is non-polynomial - should terminate fast
    expr = x / y + (x + y + x + y)  # right side never evaluated

    assert not is_linear(expr)
    assert compute_degree(expr) is None


def test_power_zero_is_constant():
    """x**0 should have degree 0."""
    x = Variable("x")
    expr = x**0

    assert is_linear(expr)
    assert compute_degree(expr) == 0


def test_power_one_preserves_degree():
    """x**1 should have degree 1."""
    x = Variable("x")
    expr = x**1

    assert is_linear(expr)
    assert compute_degree(expr) == 1


def test_constant_expression():
    """Pure constants should be linear (degree 0)."""
    x = Variable("x")
    const_expr = x * 0 + 42  # Forces expression tree with constant result

    assert is_linear(const_expr)


def test_complex_quadratic():
    """Complex quadratic expression."""
    x = Variable("x")
    y = Variable("y")
    expr = 3 * x**2 + 2 * x + y**2 - 5 * y + 10

    assert is_quadratic(expr)
    assert not is_linear(expr)
    assert compute_degree(expr) == 2


def test_cache_clear():
    """Test that cache clearing works without error."""
    x = Variable("x")
    _ = compute_degree(x + x)
    clear_degree_cache()  # Should not raise
    _ = compute_degree(x + x)  # Should work after clear


# =============================================================================
# Issue #31: LP Coefficient Extraction Tests
# =============================================================================


class TestExtractLinearCoefficient:
    """Tests for extract_linear_coefficient function."""

    def test_simple_coefficient(self):
        """3*x should have coefficient 3."""
        x = Variable("x")
        assert extract_linear_coefficient(3 * x, x) == 3.0

    def test_coefficient_sum(self):
        """x + x + x should have coefficient 3."""
        x = Variable("x")
        expr = x + x + x
        assert extract_linear_coefficient(expr, x) == 3.0

    def test_coefficient_mixed(self):
        """2*x + 3*x should have coefficient 5."""
        x = Variable("x")
        expr = 2 * x + 3 * x
        assert extract_linear_coefficient(expr, x) == 5.0

    def test_coefficient_different_var(self):
        """Coefficient of y in 3*x + 2*y should be 2."""
        x = Variable("x")
        y = Variable("y")
        expr = 3 * x + 2 * y
        assert extract_linear_coefficient(expr, x) == 3.0
        assert extract_linear_coefficient(expr, y) == 2.0

    def test_coefficient_missing_var(self):
        """Coefficient of y in 3*x should be 0."""
        x = Variable("x")
        y = Variable("y")
        expr = 3 * x + 5
        assert extract_linear_coefficient(expr, y) == 0.0

    def test_coefficient_negation(self):
        """-x should have coefficient -1."""
        x = Variable("x")
        assert extract_linear_coefficient(-x, x) == -1.0

    def test_coefficient_subtraction(self):
        """x - 2*y should have coefficients 1 and -2."""
        x = Variable("x")
        y = Variable("y")
        expr = x - 2 * y
        assert extract_linear_coefficient(expr, x) == 1.0
        assert extract_linear_coefficient(expr, y) == -2.0

    def test_coefficient_division(self):
        """x/2 should have coefficient 0.5."""
        x = Variable("x")
        assert extract_linear_coefficient(x / 2, x) == 0.5

    def test_variable_order_independence(self):
        """Coefficient extraction should be independent of variable order in expression.

        Edge case: y + x vs x + y should give same results.
        """
        x = Variable("x")
        y = Variable("y")
        expr1 = 3 * x + 2 * y
        expr2 = 2 * y + 3 * x  # Different order

        assert extract_linear_coefficient(expr1, x) == 3.0
        assert extract_linear_coefficient(expr1, y) == 2.0
        assert extract_linear_coefficient(expr2, x) == 3.0
        assert extract_linear_coefficient(expr2, y) == 2.0

    def test_nonlinear_raises(self):
        """Non-linear expression should raise NonLinearError."""
        x = Variable("x")
        with pytest.raises(NonLinearError, match="linear"):
            extract_linear_coefficient(x**2, x)


class TestExtractConstantTerm:
    """Tests for extract_constant_term function."""

    def test_simple_constant(self):
        """2*x + 7 should have constant term 7."""
        x = Variable("x")
        expr = 2 * x + 7
        assert extract_constant_term(expr) == 7.0

    def test_negative_constant(self):
        """x - 3 should have constant term -3."""
        x = Variable("x")
        expr = x - 3
        assert extract_constant_term(expr) == -3.0

    def test_no_constant(self):
        """3*x + 2*y should have constant term 0."""
        x = Variable("x")
        y = Variable("y")
        expr = 3 * x + 2 * y
        assert extract_constant_term(expr) == 0.0

    def test_pure_constant(self):
        """Pure constant expression should return the constant."""
        x = Variable("x")
        expr = x * 0 + 42
        assert extract_constant_term(expr) == 42.0

    def test_scaled_constant(self):
        """2 * (x + 5) should have constant term 10."""
        x = Variable("x")
        expr = 2 * (x + 5)
        assert extract_constant_term(expr) == 10.0

    def test_nonlinear_raises(self):
        """Non-linear expression should raise NonLinearError."""
        x = Variable("x")
        with pytest.raises(NonLinearError, match="linear"):
            extract_constant_term(x**2 + 5)


class TestLinearProgramExtractor:
    """Tests for LinearProgramExtractor class."""

    def test_extract_objective_minimize(self):
        """Test objective extraction for minimization."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem()
        prob.minimize(3 * x + 2 * y + 5)

        extractor = LinearProgramExtractor()
        c, sense, variables = extractor.extract_objective(prob)

        assert sense == "min"
        assert len(c) == 2
        # Variables are sorted by name
        var_names = [v.name for v in variables]
        x_idx = var_names.index("x")
        y_idx = var_names.index("y")
        assert c[x_idx] == 3.0
        assert c[y_idx] == 2.0

    def test_extract_objective_maximize(self):
        """Test objective extraction for maximization."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem()
        prob.maximize(x + 2 * y)

        extractor = LinearProgramExtractor()
        c, sense, _ = extractor.extract_objective(prob)

        assert sense == "max"

    def test_extract_constraints_inequality(self):
        """Test inequality constraint extraction."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem()
        prob.minimize(x + y)
        prob.subject_to(x + y <= 10)

        extractor = LinearProgramExtractor()
        _, _, variables = extractor.extract_objective(prob)
        A_ub, b_ub, A_eq, b_eq = extractor.extract_constraints(prob, variables)

        assert A_eq is None
        assert b_eq is None
        assert A_ub is not None
        assert b_ub is not None
        assert A_ub.shape == (1, 2)
        assert b_ub.shape == (1,)
        assert b_ub[0] == 10.0

    def test_extract_constraints_equality(self):
        """Test equality constraint extraction."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem()
        prob.minimize(x + y)
        constraint_expr = x + y - 5
        prob.subject_to(Constraint(constraint_expr, "=="))

        extractor = LinearProgramExtractor()
        _, _, variables = extractor.extract_objective(prob)
        A_ub, b_ub, A_eq, b_eq = extractor.extract_constraints(prob, variables)

        assert A_ub is None
        assert b_ub is None
        assert A_eq is not None
        assert b_eq is not None
        assert A_eq.shape == (1, 2)
        assert b_eq[0] == 5.0

    def test_extract_constraints_ge(self):
        """Test >= constraint extraction (converted to <=)."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem()
        prob.minimize(x + y)
        prob.subject_to(x + y >= 5)

        extractor = LinearProgramExtractor()
        _, _, variables = extractor.extract_objective(prob)
        A_ub, b_ub, _, _ = extractor.extract_constraints(prob, variables)

        assert A_ub is not None
        assert b_ub is not None
        # x + y >= 5 becomes -x - y <= -5
        assert b_ub[0] == -5.0
        # Coefficients should be negated
        assert np.all(A_ub[0] == -1.0)

    def test_extract_bounds(self):
        """Test variable bounds extraction."""
        x = Variable("x", lb=0, ub=10)
        y = Variable("y", lb=-5)
        z = Variable("z", ub=100)

        prob = Problem()
        prob.minimize(x + y + z)

        extractor = LinearProgramExtractor()
        _, _, variables = extractor.extract_objective(prob)
        bounds = extractor.extract_bounds(variables)

        var_names = [v.name for v in variables]
        x_idx = var_names.index("x")
        y_idx = var_names.index("y")
        z_idx = var_names.index("z")

        assert bounds[x_idx] == (0, 10)
        assert bounds[y_idx] == (-5, None)
        assert bounds[z_idx] == (None, 100)

    def test_extract_full_lp(self):
        """Test complete LP extraction."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem()
        prob.maximize(3 * x + 2 * y)
        prob.subject_to(x + y <= 4)
        prob.subject_to(2 * x + y <= 5)

        extractor = LinearProgramExtractor()
        lp_data = extractor.extract(prob)

        assert lp_data.sense == "max"
        assert len(lp_data.c) == 2
        assert lp_data.A_ub is not None
        assert lp_data.A_ub.shape == (2, 2)
        assert lp_data.b_ub is not None
        assert len(lp_data.b_ub) == 2
        assert lp_data.A_eq is None
        assert lp_data.b_eq is None
        assert len(lp_data.bounds) == 2
        assert lp_data.variables == ["x", "y"]

    def test_no_objective_raises(self):
        """Problem without objective should raise NoObjectiveError."""
        prob = Problem()

        extractor = LinearProgramExtractor()
        with pytest.raises(NoObjectiveError, match="No objective"):
            extractor.extract_objective(prob)

    def test_nonlinear_objective_raises(self):
        """Non-linear objective should raise NonLinearError."""
        x = Variable("x")
        prob = Problem()
        prob.minimize(x**2)

        extractor = LinearProgramExtractor()
        with pytest.raises(NonLinearError, match="linear"):
            extractor.extract_objective(prob)


# =============================================================================
# Issue #32: Constraint Helpers and Classification Tests
# =============================================================================


class TestIsSimpleBound:
    """Tests for is_simple_bound function."""

    def test_simple_lower_bound(self):
        """x >= 0 is a simple bound."""
        x = Variable("x")
        y = Variable("y")
        constraint = x >= 0
        assert is_simple_bound(constraint, [x, y])

    def test_simple_upper_bound(self):
        """x <= 10 is a simple bound."""
        x = Variable("x")
        y = Variable("y")
        constraint = x <= 10
        assert is_simple_bound(constraint, [x, y])

    def test_simple_equality_bound(self):
        """x == 5 is a simple bound."""
        x = Variable("x")
        y = Variable("y")
        constraint_expr = x - 5
        constraint = Constraint(constraint_expr, "==")
        assert is_simple_bound(constraint, [x, y])

    def test_two_variable_not_simple(self):
        """x + y <= 10 is not a simple bound."""
        x = Variable("x")
        y = Variable("y")
        constraint = x + y <= 10
        assert not is_simple_bound(constraint, [x, y])

    def test_scaled_variable_is_simple(self):
        """2*x <= 10 is a simple bound."""
        x = Variable("x")
        y = Variable("y")
        constraint = 2 * x <= 10
        assert is_simple_bound(constraint, [x, y])


class TestClassifyConstraints:
    """Tests for classify_constraints function."""

    def test_mixed_constraints(self):
        """Test classification of mixed constraint types."""
        x = Variable("x")
        y = Variable("y")

        constraints = [
            x >= 0,  # simple bound, inequality
            y <= 10,  # simple bound, inequality
            x + y <= 15,  # general, inequality
            Constraint(x - y, "=="),  # general, equality
        ]

        result = classify_constraints(constraints, [x, y])

        assert result.n_equality == 1
        assert result.n_inequality == 3
        assert result.n_simple_bounds == 2
        assert result.n_general == 2
        assert result.equality_indices == [3]
        assert result.inequality_indices == [0, 1, 2]
        assert set(result.simple_bound_indices) == {0, 1}

    def test_all_equality(self):
        """Test with only equality constraints."""
        x = Variable("x")
        y = Variable("y")

        constraints = [
            Constraint(x - 5, "=="),
            Constraint(y - 3, "=="),
        ]

        result = classify_constraints(constraints, [x, y])

        assert result.n_equality == 2
        assert result.n_inequality == 0
        assert result.n_simple_bounds == 2

    def test_empty_constraints(self):
        """Test with no constraints."""
        x = Variable("x")

        result = classify_constraints([], [x])

        assert result.n_equality == 0
        assert result.n_inequality == 0
        assert result.n_simple_bounds == 0
        assert result.n_general == 0
