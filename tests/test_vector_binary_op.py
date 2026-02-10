"""Tests for VectorBinaryOp — single-node element-wise vector operations."""

import numpy as np
import pytest

from optyx import VectorVariable, Problem
from optyx.core.vectors import VectorBinaryOp, VectorExpression
from optyx.core.compiler import compile_expression
from optyx.core.autodiff import gradient


class TestVectorBinaryOpConstruction:
    """VectorBinaryOp should be produced by vector arithmetic."""

    def test_add_produces_vector_binary_op(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        result = x + y
        assert isinstance(result, VectorBinaryOp)
        assert result.op == "+"
        assert result.size == 3

    def test_sub_produces_vector_binary_op(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        result = x - y
        assert isinstance(result, VectorBinaryOp)
        assert result.op == "-"

    def test_mul_scalar_produces_vector_binary_op(self):
        x = VectorVariable("x", 3)
        result = x * 2
        assert isinstance(result, VectorBinaryOp)
        assert result.op == "*"

    def test_rmul_scalar_produces_vector_binary_op(self):
        x = VectorVariable("x", 3)
        result = 3 * x
        assert isinstance(result, VectorBinaryOp)
        assert result.op == "*"

    def test_div_scalar_produces_vector_binary_op(self):
        x = VectorVariable("x", 3)
        result = x / 2
        assert isinstance(result, VectorBinaryOp)
        assert result.op == "/"

    def test_rsub_produces_vector_binary_op(self):
        x = VectorVariable("x", 3)
        result = 5 - x
        assert isinstance(result, VectorBinaryOp)
        assert result.op == "-"

    def test_rtruediv_produces_vector_binary_op(self):
        x = VectorVariable("x", 3)
        result = 1 / x
        assert isinstance(result, VectorBinaryOp)
        assert result.op == "/"

    def test_neg_produces_vector_binary_op(self):
        x = VectorVariable("x", 3)
        result = -x
        assert isinstance(result, VectorBinaryOp)
        assert result.op == "*"

    def test_is_subclass_of_vector_expression(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        result = x + y
        assert isinstance(result, VectorExpression)

    def test_chained_ops_produce_nested_vector_binary_ops(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        result = x + y - x  # (x + y) - x
        assert isinstance(result, VectorBinaryOp)
        assert result.op == "-"
        assert isinstance(result.left, VectorBinaryOp)

    def test_dimension_mismatch_raises(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 4)
        with pytest.raises(Exception):
            _ = x + y

    def test_vector_expression_add_produces_vector_binary_op(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        z = VectorVariable("z", 3)
        result = (x + y) + z  # VectorBinaryOp + VectorVariable
        assert isinstance(result, VectorBinaryOp)


class TestVectorBinaryOpEvaluation:
    """VectorBinaryOp should evaluate correctly using numpy."""

    def test_add_evaluation(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        result = x + y
        vals = {"x[0]": 1, "x[1]": 2, "x[2]": 3, "y[0]": 4, "y[1]": 5, "y[2]": 6}
        assert result.evaluate(vals) == [5.0, 7.0, 9.0]

    def test_sub_evaluation(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        result = x - y
        vals = {"x[0]": 10, "x[1]": 20, "x[2]": 30, "y[0]": 1, "y[1]": 2, "y[2]": 3}
        assert result.evaluate(vals) == [9.0, 18.0, 27.0]

    def test_scalar_mul_evaluation(self):
        x = VectorVariable("x", 3)
        result = x * 3
        vals = {"x[0]": 1, "x[1]": 2, "x[2]": 3}
        assert result.evaluate(vals) == [3.0, 6.0, 9.0]

    def test_rsub_evaluation(self):
        x = VectorVariable("x", 3)
        result = 10 - x
        vals = {"x[0]": 1, "x[1]": 2, "x[2]": 3}
        assert result.evaluate(vals) == [9.0, 8.0, 7.0]

    def test_rtruediv_evaluation(self):
        x = VectorVariable("x", 3)
        result = 12 / x
        vals = {"x[0]": 1, "x[1]": 2, "x[2]": 3}
        assert result.evaluate(vals) == [12.0, 6.0, 4.0]

    def test_neg_evaluation(self):
        x = VectorVariable("x", 3)
        result = -x
        vals = {"x[0]": 1, "x[1]": 2, "x[2]": 3}
        assert result.evaluate(vals) == [-1.0, -2.0, -3.0]

    def test_chained_evaluation(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        result = 2 * x + y
        vals = {"x[0]": 1, "x[1]": 2, "x[2]": 3, "y[0]": 10, "y[1]": 20, "y[2]": 30}
        assert result.evaluate(vals) == [12.0, 24.0, 36.0]


class TestVectorBinaryOpMaterialization:
    """_expressions should be lazily materialized only when accessed."""

    def test_not_materialized_on_construction(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        result = x + y
        assert result._materialized is None

    def test_materialized_on_index_access(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        result = x + y
        assert result._materialized is None
        _ = result[0]  # triggers materialization
        assert result._materialized is not None

    def test_not_materialized_by_evaluate(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        result = x + y
        vals = {"x[0]": 1, "x[1]": 2, "x[2]": 3, "y[0]": 4, "y[1]": 5, "y[2]": 6}
        result.evaluate(vals)
        assert result._materialized is None

    def test_not_materialized_by_get_variables(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        result = x + y
        _ = result.get_variables()
        assert result._materialized is None

    def test_not_materialized_by_len(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        result = x + y
        assert len(result) == 3
        assert result._materialized is None

    def test_expressions_are_populated(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        result = x + y
        assert len(result._expressions) == 3

    def test_indexing_works(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        result = x + y
        elem = result[0]
        vals = {"x[0]": 1, "y[0]": 2}
        assert elem.evaluate(vals) == 3.0

    def test_iteration_works(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        result = x + y
        elems = list(result)
        assert len(elems) == 3

    def test_get_variables(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        result = x + y
        variables = result.get_variables()
        assert len(variables) == 6

    def test_chained_lazy(self):
        """Chained VectorBinaryOps should all be lazy."""
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        z = x + y  # first VectorBinaryOp
        w = z * 2  # second VectorBinaryOp wrapping first
        assert z._materialized is None
        assert w._materialized is None
        # evaluate doesn't trigger either
        vals = {"x[0]": 1, "x[1]": 2, "x[2]": 3, "y[0]": 4, "y[1]": 5, "y[2]": 6}
        assert w.evaluate(vals) == [10.0, 14.0, 18.0]
        assert z._materialized is None
        assert w._materialized is None


class TestVectorBinaryOpSum:
    """sum() on VectorBinaryOp should work."""

    def test_sum_evaluation(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        s = (x + y).sum()
        vals = {"x[0]": 1, "x[1]": 2, "x[2]": 3, "y[0]": 4, "y[1]": 5, "y[2]": 6}
        assert s.evaluate(vals) == 21.0

    def test_sum_type(self):
        from optyx.core.vectors import VectorExpressionSum

        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        s = (x + y).sum()
        assert isinstance(s, VectorExpressionSum)


class TestVectorBinaryOpCompiler:
    """Compiler should handle VectorBinaryOp efficiently."""

    def test_dot_product_with_vector_binary_op(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        expr = (x + y).dot(x)  # dot product involving VectorBinaryOp
        variables = list(x._variables) + list(y._variables)
        fn = compile_expression(expr, variables)
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        expected = np.dot([1 + 4, 2 + 5, 3 + 6], [1, 2, 3])
        assert abs(fn(vals) - expected) < 1e-10

    def test_sum_with_vector_binary_op_compile(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        expr = (x - y).sum()
        variables = list(x._variables) + list(y._variables)
        fn = compile_expression(expr, variables)
        vals = np.array([10.0, 20.0, 30.0, 1.0, 2.0, 3.0])
        assert abs(fn(vals) - 54.0) < 1e-10

    def test_linear_combination_with_vector_binary_op(self):
        from optyx.core.vectors import LinearCombination

        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        vec_expr = x + y
        lc = LinearCombination(np.array([1.0, 2.0, 3.0]), vec_expr)
        variables = list(x._variables) + list(y._variables)
        fn = compile_expression(lc, variables)
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        # 1*(1+4) + 2*(2+5) + 3*(3+6) = 5 + 14 + 27 = 46
        assert abs(fn(vals) - 46.0) < 1e-10


class TestVectorBinaryOpGradient:
    """Autodiff should work through VectorBinaryOp via materialized expressions."""

    def test_gradient_of_sum(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        expr = (x + y).sum()
        # ∂(sum(x + y))/∂x[0] = 1
        grad = gradient(expr, x._variables[0])
        assert grad.evaluate({}) == 1.0

    def test_gradient_of_sum_wrt_y(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        expr = (x + y).sum()
        # ∂(sum(x + y))/∂y[0] = 1
        grad = gradient(expr, y._variables[0])
        assert grad.evaluate({}) == 1.0

    def test_gradient_of_weighted_sum(self):
        x = VectorVariable("x", 3)
        expr = (x * 3).sum()
        # ∂(sum(3*x))/∂x[0] = 3
        grad = gradient(expr, x._variables[0])
        assert grad.evaluate({}) == 3.0

    def test_gradient_of_difference_sum(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        expr = (x - y).sum()
        # ∂(sum(x - y))/∂x[0] = 1
        # ∂(sum(x - y))/∂y[0] = -1
        grad_x = gradient(expr, x._variables[0])
        grad_y = gradient(expr, y._variables[0])
        assert grad_x.evaluate({}) == 1.0
        assert grad_y.evaluate({}) == -1.0


class TestVectorBinaryOpCompiledGradient:
    """compile_gradient should produce fast vectorized gradients for VectorBinaryOp."""

    def test_compiled_gradient_add_sum(self):
        """sum(x + y): gradient w.r.t. all vars should be 1."""
        from optyx.core.compiler import compile_gradient

        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        expr = (x + y).sum()
        variables = list(x._variables) + list(y._variables)
        grad_fn = compile_gradient(expr, variables)
        x_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        grad = grad_fn(x_val)
        np.testing.assert_array_almost_equal(grad, np.ones(6))

    def test_compiled_gradient_sub_sum(self):
        """sum(x - y): gradient is [1,1,1,-1,-1,-1]."""
        from optyx.core.compiler import compile_gradient

        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        expr = (x - y).sum()
        variables = list(x._variables) + list(y._variables)
        grad_fn = compile_gradient(expr, variables)
        x_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        grad = grad_fn(x_val)
        np.testing.assert_array_almost_equal(grad, [1, 1, 1, -1, -1, -1])

    def test_compiled_gradient_scalar_mul_sum(self):
        """sum(x * 3): gradient is [3,3,3]."""
        from optyx.core.compiler import compile_gradient

        x = VectorVariable("x", 3)
        expr = (x * 3).sum()
        variables = list(x._variables)
        grad_fn = compile_gradient(expr, variables)
        x_val = np.array([1.0, 2.0, 3.0])
        grad = grad_fn(x_val)
        np.testing.assert_array_almost_equal(grad, [3.0, 3.0, 3.0])

    def test_compiled_gradient_neg_sum(self):
        """sum(-x): gradient is [-1,-1,-1]."""
        from optyx.core.compiler import compile_gradient

        x = VectorVariable("x", 3)
        expr = (-x).sum()
        variables = list(x._variables)
        grad_fn = compile_gradient(expr, variables)
        x_val = np.array([1.0, 2.0, 3.0])
        grad = grad_fn(x_val)
        np.testing.assert_array_almost_equal(grad, [-1.0, -1.0, -1.0])

    def test_compiled_gradient_scalar_div_sum(self):
        """sum(x / 2): gradient is [0.5, 0.5, 0.5]."""
        from optyx.core.compiler import compile_gradient

        x = VectorVariable("x", 3)
        expr = (x / 2).sum()
        variables = list(x._variables)
        grad_fn = compile_gradient(expr, variables)
        x_val = np.array([1.0, 2.0, 3.0])
        grad = grad_fn(x_val)
        np.testing.assert_array_almost_equal(grad, [0.5, 0.5, 0.5])

    def test_compiled_gradient_vec_mul_sum(self):
        """sum(x * y): ∂/∂x_i = y_i, ∂/∂y_i = x_i."""
        from optyx.core.compiler import compile_gradient

        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        expr = (x * y).sum()
        variables = list(x._variables) + list(y._variables)
        grad_fn = compile_gradient(expr, variables)
        x_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        grad = grad_fn(x_val)
        # ∂/∂x = [y0, y1, y2] = [4, 5, 6], ∂/∂y = [x0, x1, x2] = [1, 2, 3]
        np.testing.assert_array_almost_equal(grad, [4, 5, 6, 1, 2, 3])

    def test_compiled_gradient_no_materialization(self):
        """Compiled gradient should not trigger _expressions materialization."""
        from optyx.core.compiler import compile_gradient

        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        z = x + y
        expr = z.sum()
        variables = list(x._variables) + list(y._variables)
        grad_fn = compile_gradient(expr, variables)
        # Check: the VectorBinaryOp was never materialized
        assert z._materialized is None
        # Gradient computation also shouldn't materialize
        x_val = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        _ = grad_fn(x_val)
        assert z._materialized is None


class TestVectorBinaryOpConstraints:
    """Constraints on VectorBinaryOp should work."""

    def test_le_constraints(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        constraints = (x + y) <= 10
        assert len(constraints) == 3

    def test_ge_constraints(self):
        x = VectorVariable("x", 3)
        constraints = (x * 2) >= 0
        assert len(constraints) == 3


class TestVectorBinaryOpSolve:
    """End-to-end solve with VectorBinaryOp in the problem."""

    def test_minimize_with_vector_binary_op(self):
        """min sum((x - target)^2) should recover target."""
        x = VectorVariable("x", 3)
        target = np.array([1.0, 2.0, 3.0])
        diff = x - target  # VectorBinaryOp
        prob = Problem().minimize(diff.dot(diff))
        sol = prob.solve()
        assert sol.is_optimal
        for i, t in enumerate(target):
            assert abs(sol[x._variables[i]] - t) < 1e-4

    def test_minimize_sum_of_vector_binary_op(self):
        """min sum(x + y) s.t. x >= 0, y >= 0, sum(x) >= 3."""
        x = VectorVariable("x", 3, lb=0)
        y = VectorVariable("y", 3, lb=0)
        expr = (x + y).sum()
        prob = Problem().minimize(expr)
        prob.subject_to(x._variables[0] >= 3)
        sol = prob.solve()
        assert sol.is_optimal
        # y should be 0, x[0] = 3, x[1] = x[2] = 0
        assert sol[x._variables[0]] >= 2.99

    def test_repr(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        result = x + y
        assert "VectorBinaryOp" in repr(result)
        assert "+" in repr(result)
        assert "3" in repr(result)


class TestVectorExpressionSumEvaluateNoMaterialization:
    """VectorExpressionSum.evaluate() should not trigger materialization."""

    def test_evaluate_no_materialization(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        z = x + y
        s = z.sum()
        vals = {"x[0]": 1, "x[1]": 2, "x[2]": 3, "y[0]": 4, "y[1]": 5, "y[2]": 6}
        result = s.evaluate(vals)
        assert result == 21.0
        assert z._materialized is None

    def test_evaluate_sub_no_materialization(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        z = x - y
        s = z.sum()
        vals = {"x[0]": 10, "x[1]": 20, "x[2]": 30, "y[0]": 1, "y[1]": 2, "y[2]": 3}
        result = s.evaluate(vals)
        assert result == 54.0
        assert z._materialized is None

    def test_evaluate_scalar_mul_no_materialization(self):
        x = VectorVariable("x", 3)
        z = x * 5
        s = z.sum()
        vals = {"x[0]": 1, "x[1]": 2, "x[2]": 3}
        result = s.evaluate(vals)
        assert result == 30.0
        assert z._materialized is None


class TestVectorBinaryOpGradientRule:
    """@register_gradient(VectorBinaryOp) should provide O(1) gradients."""

    def test_gradient_rule_registered(self):
        from optyx.core.autodiff import has_gradient_rule

        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        z = x + y
        assert has_gradient_rule(z)

    def test_gradient_add_no_materialization(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        z = x + y
        expr = z.sum()
        g = gradient(expr, x._variables[0])
        assert g.evaluate({}) == 1.0
        assert z._materialized is None

    def test_gradient_sub_no_materialization(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        z = x - y
        expr = z.sum()
        gx = gradient(expr, x._variables[0])
        gy = gradient(expr, y._variables[0])
        assert gx.evaluate({}) == 1.0
        assert gy.evaluate({}) == -1.0
        assert z._materialized is None

    def test_gradient_scalar_mul_no_materialization(self):
        x = VectorVariable("x", 3)
        z = x * 7
        expr = z.sum()
        g = gradient(expr, x._variables[0])
        assert g.evaluate({}) == 7.0
        assert z._materialized is None

    def test_gradient_scalar_div_no_materialization(self):
        x = VectorVariable("x", 3)
        z = x / 4
        expr = z.sum()
        g = gradient(expr, x._variables[0])
        assert g.evaluate({}) == 0.25
        assert z._materialized is None

    def test_gradient_unrelated_var_is_zero(self):
        x = VectorVariable("x", 3)
        y = VectorVariable("y", 3)
        z = x + x
        expr = z.sum()
        g = gradient(expr, y._variables[0])
        assert g.evaluate({}) == 0.0
        assert z._materialized is None
