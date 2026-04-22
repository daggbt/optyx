import numpy as np
from optyx import VectorVariable, Constant
from optyx.core.autodiff import compute_hessian
from optyx.core.matrices import QuadraticForm
from optyx.core.vectors import DotProduct, VectorSum, LinearCombination


class TestConstantHessianDetection:
    """Tests for constant Hessian detection optimizations."""

    def test_quadratic_form_hessian(self):
        """Test that QuadraticForm(x, Q) has Hessian Q + Q.T."""
        n = 3
        x = VectorVariable("x", n)
        # Random non-symmetric matrix
        Q = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        qf = QuadraticForm(x, Q)
        hessian = compute_hessian(qf, list(x))

        # Check values
        Q_plus_QT = Q + Q.T
        for i in range(n):
            for j in range(n):
                val_node = hessian[i][j]
                # Optimization should return Constant nodes directly
                assert isinstance(val_node, Constant), (
                    f"Hessian element [{i},{j}] should be Constant"
                )
                np.testing.assert_almost_equal(val_node.value, Q_plus_QT[i, j])

    def test_dot_product_self_hessian(self):
        """Test that DotProduct(x, x) has Hessian 2I."""
        n = 3
        x = VectorVariable("x", n)
        dot = DotProduct(x, x)

        hessian = compute_hessian(dot, list(x))

        for i in range(n):
            for j in range(n):
                val_node = hessian[i][j]
                assert isinstance(val_node, Constant)
                expected = 2.0 if i == j else 0.0
                np.testing.assert_almost_equal(val_node.value, expected)

    def test_vector_sum_hessian(self):
        """Test that VectorSum(x) has zero Hessian."""
        n = 5
        x = VectorVariable("x", n)
        vsum = VectorSum(x)

        hessian = compute_hessian(vsum, list(x))

        for i in range(n):
            for j in range(n):
                val_node = hessian[i][j]
                assert isinstance(val_node, Constant)
                # Should be exactly 0.0
                assert val_node.value == 0.0

    def test_linear_combination_hessian(self):
        """Test that LinearCombination has zero Hessian."""
        n = 3
        x = VectorVariable("x", n)
        coeffs = np.array([1.0, 2.0, 3.0])
        # Arguments are (coefficients, vector)
        lin_comb = LinearCombination(coeffs, x)

        hessian = compute_hessian(lin_comb, list(x))

        for i in range(n):
            for j in range(n):
                val_node = hessian[i][j]
                assert isinstance(val_node, Constant)
                assert val_node.value == 0.0

    def test_quadratic_form_subset_variables(self):
        """Test QuadraticForm Hessian with a subset/superset of variables."""
        # Create x (size 2) and y (size 2)
        x = VectorVariable("x", 2)
        y = VectorVariable("y", 2)

        Q = np.array([[1.0, 2.0], [3.0, 4.0]])

        # Form depends only on x: x'Qx
        qf = QuadraticForm(x, Q)

        # Calculate Hessian wrt [x0, x1, y0]
        # Top-left 2x2 should be Q+Q.T
        # Rest should be 0
        vars_to_diff = [x[0], x[1], y[0]]
        hessian = compute_hessian(qf, vars_to_diff)

        Q_plus_QT = Q + Q.T

        # Check x-x part
        np.testing.assert_almost_equal(hessian[0][0].value, Q_plus_QT[0, 0])
        np.testing.assert_almost_equal(hessian[0][1].value, Q_plus_QT[0, 1])
        np.testing.assert_almost_equal(hessian[1][0].value, Q_plus_QT[1, 0])
        np.testing.assert_almost_equal(hessian[1][1].value, Q_plus_QT[1, 1])

        # Check cross terms with y (should be 0)
        assert hessian[0][2].value == 0.0  # d2/dx0dy0
        assert hessian[1][2].value == 0.0  # d2/dx1dy0
        assert hessian[2][0].value == 0.0  # d2/dy0dx0
        assert hessian[2][1].value == 0.0  # d2/dy0dx1
        assert hessian[2][2].value == 0.0  # d2/dy0dy0

    def test_linear_combination_of_quadratic_forms(self):
        """Test Hessian of c1*QF1 + c2*QF2 avoids re-evaluation."""
        n = 2
        x = VectorVariable("x", n)
        Q1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        Q2 = np.array([[0.0, 1.0], [1.0, 0.0]])

        # Expr = 2 * x'Q1x + 3 * x'Q2x
        # Hessian should be 2*(Q1+Q1.T) + 3*(Q2+Q2.T)
        expr = Constant(2.0) * QuadraticForm(x, Q1) + Constant(3.0) * QuadraticForm(
            x, Q2
        )

        hessian = compute_hessian(expr, list(x))

        # Expected value
        # 2*[[2,0],[0,2]] + 3*[[0,2],[2,0]] = [[4,0],[0,4]] + [[0,6],[6,0]] = [[4,6],[6,4]]
        H_val = np.array([[4.0, 6.0], [6.0, 4.0]])

        for i in range(n):
            for j in range(n):
                val_node = hessian[i][j]
                # Optimization should return Constant nodes directly (pre-computed)
                # Not BinaryOp(BinaryOp(Constant...))
                assert isinstance(val_node, Constant), (
                    f"Hessian element [{i},{j}] should be Constant, got {type(val_node)}"
                )
                np.testing.assert_almost_equal(val_node.value, H_val[i, j])
