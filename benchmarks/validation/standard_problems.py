"""Validation benchmarks for standard unconstrained optimization problems.

Tests Optyx against well-known test functions with known global minima.
"""

from __future__ import annotations

import numpy as np

from optyx import Variable, Problem


class TestRosenbrock:
    """Rosenbrock function tests.

    f(x,y) = (a-x)² + b(y-x²)²
    Standard: a=1, b=100
    Optimal: (a, a²) = (1, 1) with f* = 0
    """

    def test_rosenbrock_2d(self):
        """Classic 2D Rosenbrock."""
        x = Variable("x")
        y = Variable("y")

        prob = Problem(name="rosenbrock_2d")
        prob.minimize((1 - x) ** 2 + 100 * (y - x**2) ** 2)

        sol = prob.solve(x0=np.array([-1.0, -1.0]))

        assert sol.is_optimal
        assert abs(sol["x"] - 1.0) < 1e-4
        assert abs(sol["y"] - 1.0) < 1e-4
        assert abs(sol.objective_value) < 1e-6

    def test_rosenbrock_4d(self):
        """4D Rosenbrock (sum of 2D slices)."""
        n = 4
        variables = [Variable(f"x{i}") for i in range(n)]

        prob = Problem(name="rosenbrock_4d")
        objective = sum(
            (1 - variables[i]) ** 2 + 100 * (variables[i + 1] - variables[i] ** 2) ** 2
            for i in range(n - 1)
        )
        prob.minimize(objective)

        sol = prob.solve(x0=np.zeros(n))

        assert sol.is_optimal
        # All variables should be close to 1
        for i in range(n):
            assert abs(sol[f"x{i}"] - 1.0) < 1e-3, f"x{i} = {sol[f'x{i}']}"
        assert sol.objective_value < 1e-5


class TestSphere:
    """Sphere function tests.

    f(x) = sum(x_i²)
    Optimal: x = 0 with f* = 0
    """

    def test_sphere_2d(self):
        """2D sphere function."""
        x = Variable("x")
        y = Variable("y")

        prob = Problem(name="sphere_2d")
        prob.minimize(x**2 + y**2)

        sol = prob.solve(x0=np.array([5.0, -3.0]))

        assert sol.is_optimal
        assert abs(sol["x"]) < 1e-4
        assert abs(sol["y"]) < 1e-4
        assert abs(sol.objective_value) < 1e-8

    def test_sphere_10d(self):
        """10D sphere function."""
        n = 10
        variables = [Variable(f"x{i}") for i in range(n)]

        prob = Problem(name="sphere_10d")
        prob.minimize(sum(v**2 for v in variables))

        sol = prob.solve(x0=np.arange(n, dtype=float))

        assert sol.is_optimal
        for i in range(n):
            assert abs(sol[f"x{i}"]) < 1e-5
        assert abs(sol.objective_value) < 1e-8


class TestBeale:
    """Beale function test.

    f(x,y) = (1.5 - x + xy)² + (2.25 - x + xy²)² + (2.625 - x + xy³)²
    Optimal: (3, 0.5) with f* = 0
    Domain: |x|, |y| <= 4.5
    """

    def test_beale(self):
        """Beale function."""
        x = Variable("x", lb=-4.5, ub=4.5)
        y = Variable("y", lb=-4.5, ub=4.5)

        prob = Problem(name="beale")
        prob.minimize(
            (1.5 - x + x * y) ** 2
            + (2.25 - x + x * y**2) ** 2
            + (2.625 - x + x * y**3) ** 2
        )

        sol = prob.solve(x0=np.array([0.0, 0.0]))

        assert sol.is_optimal
        assert abs(sol["x"] - 3.0) < 1e-3
        assert abs(sol["y"] - 0.5) < 1e-3
        assert sol.objective_value < 1e-6


class TestBooth:
    """Booth function test.

    f(x,y) = (x + 2y - 7)² + (2x + y - 5)²
    Optimal: (1, 3) with f* = 0
    """

    def test_booth(self):
        """Booth function."""
        x = Variable("x")
        y = Variable("y")

        prob = Problem(name="booth")
        prob.minimize((x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2)

        sol = prob.solve(x0=np.array([0.0, 0.0]))

        assert sol.is_optimal
        assert abs(sol["x"] - 1.0) < 1e-4
        assert abs(sol["y"] - 3.0) < 1e-4
        assert abs(sol.objective_value) < 1e-8


class TestMatyas:
    """Matyas function test.

    f(x,y) = 0.26(x² + y²) - 0.48xy
    Optimal: (0, 0) with f* = 0
    """

    def test_matyas(self):
        """Matyas function."""
        x = Variable("x")
        y = Variable("y")

        prob = Problem(name="matyas")
        prob.minimize(0.26 * (x**2 + y**2) - 0.48 * x * y)

        sol = prob.solve(x0=np.array([5.0, -5.0]))

        assert sol.is_optimal
        assert abs(sol["x"]) < 1e-4
        assert abs(sol["y"]) < 1e-4
        assert abs(sol.objective_value) < 1e-6


class TestQuadratic:
    """Simple quadratic function tests."""

    def test_simple_quadratic(self):
        """f(x) = (x-3)² with f* = 0 at x = 3."""
        x = Variable("x")

        prob = Problem(name="simple_quadratic")
        prob.minimize((x - 3) ** 2)

        sol = prob.solve(x0=np.array([0.0]))

        assert sol.is_optimal
        assert abs(sol["x"] - 3.0) < 1e-4
        assert abs(sol.objective_value) < 1e-8

    def test_shifted_quadratic(self):
        """f(x,y) = (x-2)² + (y+1)² with f* = 0 at (2, -1)."""
        x = Variable("x")
        y = Variable("y")

        prob = Problem(name="shifted_quadratic")
        prob.minimize((x - 2) ** 2 + (y + 1) ** 2)

        sol = prob.solve(x0=np.array([0.0, 0.0]))

        assert sol.is_optimal
        assert abs(sol["x"] - 2.0) < 1e-4
        assert abs(sol["y"] + 1.0) < 1e-4
        assert abs(sol.objective_value) < 1e-8
