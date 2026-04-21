"""Accuracy benchmark: Gradient validation.

Validates Optyx autodiff gradients against finite difference approximations
AND hand-derived analytical formulas.
Target: < 1e-6 relative error.
"""

from __future__ import annotations

import numpy as np

from optyx import Variable, Problem, sin, cos, exp, log, sqrt
from optyx.core.autodiff import compile_jacobian
from optyx.core.optimizer import flatten_expression

import sys

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])


def finite_difference_gradient(
    func: callable,
    x: np.ndarray,
    eps: float = 1e-7,
) -> np.ndarray:
    """Compute gradient using central finite differences."""
    n = len(x)
    grad = np.zeros(n)
    for i in range(n):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (func(x_plus) - func(x_minus)) / (2 * eps)
    return grad


def _optyx_gradient(prob: Problem) -> callable:
    """Compile the Optyx gradient for a problem's objective.

    Returns a callable ``(x) -> 1-d gradient array``.
    """
    obj_expr = prob.objective
    if prob.sense == "maximize":
        obj_expr = -obj_expr  # type: ignore[operator]
    obj_expr = flatten_expression(obj_expr)
    variables = prob.variables
    jac_fn = compile_jacobian([obj_expr], variables)
    return lambda x: jac_fn(x).flatten()


class TestPolynomialGradients:
    """Test gradients of polynomial expressions."""

    def test_quadratic_gradient(self):
        """f(x,y) = x² + 2xy + 3y²"""
        x = Variable("x")
        y = Variable("y")

        prob = Problem()
        prob.minimize(x**2 + 2 * x * y + 3 * y**2)

        variables = prob.variables
        var_names = [v.name for v in variables]
        optyx_grad = _optyx_gradient(prob)

        def f(vals):
            v = dict(zip(var_names, vals))
            return v["x"] ** 2 + 2 * v["x"] * v["y"] + 3 * v["y"] ** 2

        test_points = [
            np.array([1.0, 2.0]),
            np.array([0.5, -0.3]),
            np.array([10.0, -5.0]),
        ]

        for point in test_points:
            fd_grad = finite_difference_gradient(f, point)
            x_val, y_val = point
            analytical_grad = np.array([2 * x_val + 2 * y_val, 2 * x_val + 6 * y_val])
            ox_grad = optyx_grad(point)

            error_fd = np.linalg.norm(fd_grad - analytical_grad)
            assert error_fd < 1e-5, f"FD vs analytical error at {point}: {error_fd}"

            error_ox = np.linalg.norm(ox_grad - analytical_grad)
            assert error_ox < 1e-5, f"Optyx vs analytical error at {point}: {error_ox}"

    def test_cubic_gradient(self):
        """f(x) = x³ - 3x + 1"""
        x = Variable("x")

        prob = Problem()
        prob.minimize(x**3 - 3 * x + 1)
        optyx_grad = _optyx_gradient(prob)

        def f(vals):
            return vals[0] ** 3 - 3 * vals[0] + 1

        test_points = [np.array([0.0]), np.array([1.0]), np.array([-2.0])]

        for point in test_points:
            fd_grad = finite_difference_gradient(f, point)
            analytical = np.array([3 * point[0] ** 2 - 3])
            ox_grad = optyx_grad(point)

            error_fd = np.abs(fd_grad[0] - analytical[0])
            assert error_fd < 1e-5, f"FD vs analytical error at {point}: {error_fd}"

            error_ox = np.abs(ox_grad[0] - analytical[0])
            assert error_ox < 1e-5, f"Optyx vs analytical error at {point}: {error_ox}"


class TestTranscendentalGradients:
    """Test gradients of transcendental functions."""

    def test_sin_cos_gradient(self):
        """f(x) = sin(x) + cos(x)"""
        x = Variable("x")

        prob = Problem()
        prob.minimize(sin(x) + cos(x))
        optyx_grad = _optyx_gradient(prob)

        def f(vals):
            return np.sin(vals[0]) + np.cos(vals[0])

        test_points = [np.array([0.0]), np.array([np.pi / 4]), np.array([1.5])]

        for point in test_points:
            fd_grad = finite_difference_gradient(f, point)
            analytical = np.array([np.cos(point[0]) - np.sin(point[0])])
            ox_grad = optyx_grad(point)

            error_fd = np.abs(fd_grad[0] - analytical[0])
            assert error_fd < 1e-5, f"FD vs analytical error at {point}: {error_fd}"

            error_ox = np.abs(ox_grad[0] - analytical[0])
            assert error_ox < 1e-5, f"Optyx vs analytical error at {point}: {error_ox}"

    def test_exp_gradient(self):
        """f(x,y) = exp(x) + exp(-y)"""
        x = Variable("x")
        y = Variable("y")

        prob = Problem()
        prob.minimize(exp(x) + exp(-y))
        optyx_grad = _optyx_gradient(prob)

        def f(vals):
            return np.exp(vals[0]) + np.exp(-vals[1])

        test_points = [
            np.array([0.0, 0.0]),
            np.array([1.0, -1.0]),
            np.array([-0.5, 0.5]),
        ]

        for point in test_points:
            fd_grad = finite_difference_gradient(f, point)
            analytical = np.array([np.exp(point[0]), -np.exp(-point[1])])
            ox_grad = optyx_grad(point)

            error_fd = np.linalg.norm(fd_grad - analytical)
            assert error_fd < 1e-5, f"FD vs analytical error at {point}: {error_fd}"

            error_ox = np.linalg.norm(ox_grad - analytical)
            assert error_ox < 1e-5, f"Optyx vs analytical error at {point}: {error_ox}"

    def test_log_gradient(self):
        """f(x) = log(x) + x*log(x)"""
        x = Variable("x", lb=0.1)  # Avoid log(0)

        prob = Problem()
        prob.minimize(log(x) + x * log(x))
        optyx_grad = _optyx_gradient(prob)

        def f(vals):
            return np.log(vals[0]) + vals[0] * np.log(vals[0])

        test_points = [np.array([0.5]), np.array([1.0]), np.array([2.0])]

        for point in test_points:
            fd_grad = finite_difference_gradient(f, point)
            analytical = np.array([1 / point[0] + np.log(point[0]) + 1])
            ox_grad = optyx_grad(point)

            error_fd = np.abs(fd_grad[0] - analytical[0])
            assert error_fd < 1e-5, f"FD vs analytical error at {point}: {error_fd}"

            error_ox = np.abs(ox_grad[0] - analytical[0])
            assert error_ox < 1e-5, f"Optyx vs analytical error at {point}: {error_ox}"

    def test_sqrt_gradient(self):
        """f(x,y) = sqrt(x² + y²)"""
        x = Variable("x")
        y = Variable("y")

        prob = Problem()
        prob.minimize(sqrt(x**2 + y**2))
        optyx_grad = _optyx_gradient(prob)

        def f(vals):
            return np.sqrt(vals[0] ** 2 + vals[1] ** 2)

        test_points = [
            np.array([3.0, 4.0]),
            np.array([1.0, 1.0]),
            np.array([2.0, -3.0]),
        ]

        for point in test_points:
            fd_grad = finite_difference_gradient(f, point, eps=1e-6)
            r = np.sqrt(point[0] ** 2 + point[1] ** 2)
            analytical = np.array([point[0] / r, point[1] / r])
            ox_grad = optyx_grad(point)

            error_fd = np.linalg.norm(fd_grad - analytical)
            assert error_fd < 1e-4, f"FD vs analytical error at {point}: {error_fd}"

            error_ox = np.linalg.norm(ox_grad - analytical)
            assert error_ox < 1e-4, f"Optyx vs analytical error at {point}: {error_ox}"


class TestCompositeGradients:
    """Test gradients of composite expressions."""

    def test_rosenbrock_gradient(self):
        """f(x,y) = (1-x)² + 100(y-x²)²"""
        x = Variable("x")
        y = Variable("y")

        prob = Problem()
        prob.minimize((1 - x) ** 2 + 100 * (y - x**2) ** 2)
        optyx_grad = _optyx_gradient(prob)

        def f(vals):
            x, y = vals
            return (1 - x) ** 2 + 100 * (y - x**2) ** 2

        test_points = [
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
            np.array([0.5, 0.25]),
        ]

        for point in test_points:
            fd_grad = finite_difference_gradient(f, point)
            x, y = point
            analytical = np.array(
                [
                    -2 * (1 - x) - 400 * x * (y - x**2),
                    200 * (y - x**2),
                ]
            )
            ox_grad = optyx_grad(point)

            error_fd = np.linalg.norm(fd_grad - analytical)
            assert error_fd < 1e-4, f"FD vs analytical error at {point}: {error_fd}"

            error_ox = np.linalg.norm(ox_grad - analytical)
            assert error_ox < 1e-4, f"Optyx vs analytical error at {point}: {error_ox}"

    def test_mixed_expression_gradient(self):
        """f(x,y) = x*sin(y) + exp(x*y)"""
        x = Variable("x")
        y = Variable("y")

        prob = Problem()
        prob.minimize(x * sin(y) + exp(x * y))
        optyx_grad = _optyx_gradient(prob)

        def f(vals):
            x, y = vals
            return x * np.sin(y) + np.exp(x * y)

        test_points = [
            np.array([0.5, 0.5]),
            np.array([1.0, 0.0]),
            np.array([0.3, 1.0]),
        ]

        for point in test_points:
            fd_grad = finite_difference_gradient(f, point)
            x, y = point
            analytical = np.array(
                [
                    np.sin(y) + y * np.exp(x * y),
                    x * np.cos(y) + x * np.exp(x * y),
                ]
            )
            ox_grad = optyx_grad(point)

            error_fd = np.linalg.norm(fd_grad - analytical)
            assert error_fd < 1e-4, f"FD vs analytical error at {point}: {error_fd}"

            error_ox = np.linalg.norm(ox_grad - analytical)
            assert error_ox < 1e-4, f"Optyx vs analytical error at {point}: {error_ox}"


class TestConstraintGradients:
    """Test gradients of constraint functions."""

    def test_linear_constraint_gradient(self):
        """g(x,y) = 2x + 3y - 5"""
        x = Variable("x")
        y = Variable("y")

        prob = Problem()
        prob.minimize(x**2 + y**2)
        prob.subject_to(2 * x + 3 * y >= 5)

        # Compile Optyx constraint Jacobian
        c_expr = prob.constraints[0].expr
        c_expr = flatten_expression(c_expr)
        variables = prob.variables
        c_jac_fn = compile_jacobian([c_expr], variables)

        def g(vals):
            return 2 * vals[0] + 3 * vals[1] - 5

        point = np.array([1.0, 1.0])
        fd_grad = finite_difference_gradient(g, point)
        analytical = np.array([2.0, 3.0])
        ox_grad = c_jac_fn(point).flatten()

        error_fd = np.linalg.norm(fd_grad - analytical)
        assert error_fd < 1e-6, f"FD vs analytical error: {error_fd}"

        error_ox = np.linalg.norm(ox_grad - analytical)
        assert error_ox < 1e-6, f"Optyx vs analytical error: {error_ox}"

    def test_nonlinear_constraint_gradient(self):
        """g(x,y) = x² + y² - 1"""
        x = Variable("x")
        y = Variable("y")

        prob = Problem()
        prob.minimize(x + y)
        prob.subject_to(x**2 + y**2 <= 1)

        # Compile Optyx constraint Jacobian
        c_expr = prob.constraints[0].expr
        c_expr = flatten_expression(c_expr)
        variables = prob.variables
        c_jac_fn = compile_jacobian([c_expr], variables)

        def g(vals):
            return vals[0] ** 2 + vals[1] ** 2 - 1

        test_points = [
            np.array([0.5, 0.5]),
            np.array([1.0, 0.0]),
            np.array([0.6, 0.8]),
        ]

        for point in test_points:
            fd_grad = finite_difference_gradient(g, point)
            analytical = 2 * point
            ox_grad = c_jac_fn(point).flatten()

            error_fd = np.linalg.norm(fd_grad - analytical)
            assert error_fd < 1e-5, f"FD vs analytical error at {point}: {error_fd}"

            error_ox = np.linalg.norm(ox_grad - analytical)
            assert error_ox < 1e-5, f"Optyx vs analytical error at {point}: {error_ox}"


if __name__ == "__main__":
    print("=" * 60)
    print("GRADIENT VALIDATION")
    print("=" * 60)

    test_poly = TestPolynomialGradients()
    test_poly.test_quadratic_gradient()
    test_poly.test_cubic_gradient()
    print("✓ Polynomial gradients OK")

    test_trans = TestTranscendentalGradients()
    test_trans.test_sin_cos_gradient()
    test_trans.test_exp_gradient()
    test_trans.test_log_gradient()
    test_trans.test_sqrt_gradient()
    print("✓ Transcendental gradients OK")

    test_comp = TestCompositeGradients()
    test_comp.test_rosenbrock_gradient()
    test_comp.test_mixed_expression_gradient()
    print("✓ Composite gradients OK")

    test_const = TestConstraintGradients()
    test_const.test_linear_constraint_gradient()
    test_const.test_nonlinear_constraint_gradient()
    print("✓ Constraint gradients OK")
