from optyx.core.expressions import Variable
from optyx.analysis import is_linear, is_quadratic, compute_degree, clear_degree_cache


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
