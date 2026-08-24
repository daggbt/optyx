"""Tests for NarySum, NaryProduct and flatten_expression."""

import numpy as np

from optyx.analysis import (
    _check_degree_bounded,
    compute_degree,
    extract_all_linear_coefficients,
    is_linear,
)
from optyx.core.expressions import (
    Variable,
    Constant,
    BinaryOp,
    NarySum,
    NaryProduct,
)
from optyx.core.functions import sin
from optyx.core.optimizer import flatten_expression, optimize_expression
from optyx.core.parameters import Parameter
from optyx.core.vectors import VectorVariable


def test_nary_sum_evaluation():
    x = Variable("x")
    y = Variable("y")
    z = Variable("z")

    expr = NarySum((x, y, z))
    # 1 + 2 + 3 = 6
    assert expr.evaluate({"x": 1, "y": 2, "z": 3}) == 6


def test_nary_product_evaluation():
    x = Variable("x")
    y = Variable("y")
    z = Variable("z")

    expr = NaryProduct((x, y, z))
    # 2 * 3 * 4 = 24
    assert expr.evaluate({"x": 2, "y": 3, "z": 4}) == 24


def test_flatten_binary_op_simple():
    # a + b -> a + b (BinaryOp)
    x = Variable("x")
    y = Variable("y")

    expr = x + y
    flat = flatten_expression(expr)

    assert isinstance(flat, BinaryOp)
    assert flat.left is x
    assert flat.right is y
    assert flat.op == "+"


def test_flatten_binary_op_nested_sum():
    # (x + y) + z -> NarySum(x, y, z)
    x = Variable("x")
    y = Variable("y")
    z = Variable("z")

    expr = (x + y) + z
    flat = flatten_expression(expr)

    assert isinstance(flat, NarySum)
    assert len(flat.terms) == 3
    assert flat.terms[0] is x
    assert flat.terms[1] is y
    assert flat.terms[2] is z


def test_flatten_binary_op_nested_product():
    # (x * y) * z -> NaryProduct(x, y, z)
    x = Variable("x")
    y = Variable("y")
    z = Variable("z")

    expr = (x * y) * z
    flat = flatten_expression(expr)

    assert isinstance(flat, NaryProduct)
    assert len(flat.factors) == 3
    assert flat.factors[0] is x
    assert flat.factors[1] is y
    assert flat.factors[2] is z


def test_flatten_deep_sum():
    # x0 + x1 + ... + x99 using standard loop
    vars = [Variable(f"x{i}") for i in range(100)]

    # Simulate sum() behavior: ((var0 + var1) + var2) ...
    expr = vars[0]
    for i in range(1, 100):
        expr = expr + vars[i]

    flat = flatten_expression(expr)

    assert isinstance(flat, NarySum)
    assert len(flat.terms) == 100
    for i in range(100):
        assert flat.terms[i] is vars[i]


def test_flatten_mixed_associativity():
    # (a + b) + (c + d) -> NarySum(a, b, c, d)
    a = Variable("a")
    b = Variable("b")
    c = Variable("c")
    d = Variable("d")

    expr = (a + b) + (c + d)
    flat = flatten_expression(expr)

    assert isinstance(flat, NarySum)
    assert len(flat.terms) == 4
    assert flat.terms == (a, b, c, d)


def test_flatten_preserves_evaluation():
    # ((x * 2) + y) * 3 + z
    # Should optimize outer sums/products but keep structure
    x = Variable("x")
    y = Variable("y")
    z = Variable("z")

    # 3 * ((2*x) + y) + z
    # This has structure:
    #       +
    #      / \
    #     *   z
    #    / \
    #   3   +
    #      / \
    #     *   y
    #    / \
    #   2   x

    # Top level + has 2 children: (*...) and z. So likely stays BinaryOp or NarySum(2).
    # Since our logic says len==2 -> BinaryOp, it stays BinaryOp.

    term1 = 3 * ((2 * x) + y)
    expr = term1 + z

    flat = flatten_expression(expr)

    # Evaluate check
    values = {"x": 2, "y": 5, "z": 10}
    # (2*2 + 5) * 3 + 10 = 9 * 3 + 10 = 37

    assert flat.evaluate(values) == 37.0


def test_flatten_nary_sum_input():
    # If input is already NarySum, flatten into it
    x = Variable("x")
    y = Variable("y")
    z = Variable("z")

    nary = NarySum((x, y))
    expr = nary + z

    flat = flatten_expression(expr)

    assert isinstance(flat, NarySum)
    assert len(flat.terms) == 3
    assert flat.terms == (x, y, z)


def test_optimize_expression_delegates_to_flatten_expression():
    x = Variable("x")
    y = Variable("y")
    z = Variable("z")

    expr = (x + y) + z
    optimized = optimize_expression(expr)

    assert isinstance(optimized, NarySum)
    assert optimized.terms == (x, y, z)


# ---- Gradient tests ----


def test_nary_sum_gradient():
    """d/dx(x + y + z) = 1 for any term, 0 for others."""
    from optyx.core.autodiff import gradient

    x = Variable("x")
    y = Variable("y")
    z = Variable("z")

    expr = NarySum((x, y, z))
    grad_x = gradient(expr, x)
    grad_y = gradient(expr, y)

    assert grad_x.evaluate({}) == 1.0
    assert grad_y.evaluate({}) == 1.0


def test_nary_product_gradient():
    """d/dx(x * y * z) = y * z."""
    from optyx.core.autodiff import gradient

    x = Variable("x")
    y = Variable("y")
    z = Variable("z")

    expr = NaryProduct((x, y, z))
    grad_x = gradient(expr, x)

    # d/dx(x * y * z) = y * z
    assert grad_x.evaluate({"x": 2, "y": 3, "z": 5}) == 15.0


def test_nary_sum_gradient_with_constants():
    """d/dx(2 + x + 3) = 1."""
    from optyx.core.autodiff import gradient
    from optyx.core.expressions import Constant

    x = Variable("x")
    expr = NarySum((Constant(2.0), x, Constant(3.0)))
    grad = gradient(expr, x)

    assert grad.evaluate({"x": 99}) == 1.0


def test_nary_get_variables():
    """NarySum and NaryProduct report all contained variables."""
    x = Variable("x")
    y = Variable("y")
    z = Variable("z")

    s = NarySum((x, y, z))
    p = NaryProduct((x, z))

    assert s.get_variables() == {x, y, z}
    assert p.get_variables() == {x, z}


def test_flatten_deep_product():
    """x0 * x1 * ... * x9 flattens to NaryProduct."""
    vars = [Variable(f"x{i}") for i in range(10)]

    expr = vars[0]
    for i in range(1, 10):
        expr = expr * vars[i]

    flat = flatten_expression(expr)

    assert isinstance(flat, NaryProduct)
    assert len(flat.factors) == 10


def test_flatten_does_not_mix_ops():
    """(a + b) * (c + d) should NOT flatten across different operators."""
    a = Variable("a")
    b = Variable("b")
    c = Variable("c")
    d = Variable("d")

    expr = (a + b) * (c + d)
    flat = flatten_expression(expr)

    # Top-level is *, children are sums — should not merge + and *
    assert isinstance(flat, BinaryOp)
    assert flat.op == "*"


# ===================================================================
# Degree analysis and LP extraction
# ===================================================================


def test_degree_is_preserved_by_flattening():
    x = Variable("x")
    y = Variable("y")
    z = Variable("z")
    expressions = (x + y + z, x * y * z)

    for expr in expressions:
        assert compute_degree(flatten_expression(expr)) == compute_degree(expr)


def test_nary_degree_aggregation():
    x = Variable("x")
    y = Variable("y")

    assert compute_degree(NarySum((Constant(1), x, y**2))) == 2
    assert compute_degree(NaryProduct((Constant(2), x, y))) == 2
    assert compute_degree(NarySum((Constant(1), Constant(2)))) == 0
    assert compute_degree(NaryProduct((Constant(2), Constant(3)))) == 0
    assert _check_degree_bounded(NaryProduct((Constant(2), x, y)), 2)
    assert not _check_degree_bounded(NaryProduct((Constant(2), x, y)), 1)


def test_nary_degree_rejects_nonpolynomial_child():
    x = Variable("x")

    assert compute_degree(NarySum((Constant(1), sin(x)))) is None
    assert compute_degree(NaryProduct((Constant(2), sin(x)))) is None


def test_nary_degree_handles_deep_child_iteratively():
    x = Variable("x")
    deep_expr = x
    for _ in range(450):
        deep_expr = deep_expr - x

    assert compute_degree(NarySum((deep_expr, Constant(1)))) == 1


def test_flattened_linear_sum_extracts_all_coefficients():
    variables = [Variable(f"x{i}") for i in range(100)]
    expr = flatten_expression(sum(variables[1:], start=variables[0]))
    var_index = {variable.name: i for i, variable in enumerate(variables)}

    coefficients = extract_all_linear_coefficients(expr, var_index, len(variables))

    assert is_linear(expr)
    np.testing.assert_array_equal(coefficients, np.ones(len(variables)))


def test_nary_product_extracts_parameterized_coefficient():
    x = Variable("x")
    scale = Parameter("scale", 3)
    expr = NaryProduct((Constant(2), scale, x))

    coefficients = extract_all_linear_coefficients(expr, {"x": 0}, 1)

    assert compute_degree(expr) == 1
    np.testing.assert_array_equal(coefficients, [6.0])


def test_nary_vector_sum_uses_single_vector_source():
    from optyx import Problem

    x = VectorVariable("x", 100, lb=0)
    expr = NarySum((x.sum(), Constant(2)))
    problem = Problem().minimize(expr)

    assert problem._single_vector_source() is x
    coefficients = extract_all_linear_coefficients(
        expr, {x._name_at(i): i for i in range(x.size)}, x.size
    )
    np.testing.assert_array_equal(coefficients, np.ones(x.size))


# ===================================================================
# NarySum gradient flatness tests
# ===================================================================


def test_nary_sum_gradient_produces_flat_output():
    """Gradient of NarySum with 4+ terms should produce NarySum, not nested BinaryOp."""
    from optyx.core.autodiff import gradient

    vars = [Variable(f"x{i}") for i in range(5)]
    expr = NarySum(tuple(v * v for v in vars))  # sum(x_i^2)

    g = gradient(expr, vars[0])

    # ∂(sum(x_i^2))/∂x_0 = 2*x_0 — only one Non-zero term
    # so it should just be a single expression, not NarySum
    assert not isinstance(g, NarySum)
    vals = {"x0": 3.0}
    assert g.evaluate(vals) == 6.0


def test_nary_sum_gradient_multiple_nonzero_terms():
    """Gradient of NarySum where multiple terms contribute should produce NarySum."""
    from optyx.core.autodiff import gradient

    x = Variable("x")
    # sum(x, 2*x, 3*x, 4*x, 5*x)
    expr = NarySum((x, x * 2, x * 3, x * 4, x * 5))

    g = gradient(expr, x)
    # ∂/∂x = 1 + 2 + 3 + 4 + 5 = 15
    assert g.evaluate({}) == 15.0
    # Should be flat — either NarySum or simplified
    if isinstance(g, NarySum):
        # Flat NarySum with 5 terms (not nested BinaryOp chain)
        assert len(g.terms) == 5
    # Regardless of structure, must not be a deep chain
    from optyx.core.expressions import _estimate_tree_depth

    assert _estimate_tree_depth(g) < 5


def test_nary_sum_gradient_all_zero_terms():
    """Gradient of NarySum where wrt is not present returns Constant(0)."""
    from optyx.core.autodiff import gradient
    from optyx.core.expressions import Constant

    x = Variable("x")
    y = Variable("y")
    expr = NarySum((y, y * 2, y * 3))

    g = gradient(expr, x)
    assert isinstance(g, Constant)
    assert g.evaluate({}) == 0.0


def test_nary_sum_gradient_single_nonzero_term():
    """If only one term has nonzero gradient, no wrapping in NarySum."""
    from optyx.core.autodiff import gradient

    x = Variable("x")
    y = Variable("y")
    z = Variable("z")
    expr = NarySum((x * 2, y, z))

    g = gradient(expr, x)
    # Only x*2 contributes: ∂(2*x)/∂x = 2
    assert not isinstance(g, NarySum)
    assert g.evaluate({}) == 2.0
