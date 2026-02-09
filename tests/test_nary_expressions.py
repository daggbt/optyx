"""Tests for NarySum, NaryProduct and flatten_expression."""

from optyx.core.expressions import Variable, BinaryOp, NarySum, NaryProduct
from optyx.core.optimizer import flatten_expression


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
