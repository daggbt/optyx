"""Shared fixtures for benchmark tests."""

from __future__ import annotations

import pytest
import numpy as np

from optyx import Variable, Problem


# =============================================================================
# Standard Test Problems
# =============================================================================


@pytest.fixture
def rosenbrock_2d() -> tuple[Problem, dict[str, float], float]:
    """Rosenbrock function in 2D.

    f(x,y) = (1-x)² + 100(y-x²)²
    Optimal: (1, 1) with f* = 0
    """
    x = Variable("x", initial=0.0)
    y = Variable("y", initial=0.0)

    prob = Problem(name="rosenbrock_2d")
    prob.minimize((1 - x) ** 2 + 100 * (y - x**2) ** 2)

    optimal_values = {"x": 1.0, "y": 1.0}
    optimal_objective = 0.0

    return prob, optimal_values, optimal_objective


@pytest.fixture
def sphere_10d() -> tuple[Problem, dict[str, float], float]:
    """Sphere function in 10D.

    f(x) = sum(x_i²)
    Optimal: (0, 0, ..., 0) with f* = 0
    """
    n = 10
    variables = [Variable(f"x{i}", initial=1.0) for i in range(n)]

    prob = Problem(name="sphere_10d")
    objective = sum(v**2 for v in variables)
    prob.minimize(objective)

    optimal_values = {f"x{i}": 0.0 for i in range(n)}
    optimal_objective = 0.0

    return prob, optimal_values, optimal_objective


@pytest.fixture
def simple_lp() -> tuple[Problem, dict[str, float], float]:
    """Simple linear program.

    max 3x + 2y
    s.t. x + y <= 4
         x <= 2
         y <= 3
         x, y >= 0

    Optimal: (2, 2) with f* = 10
    """
    x = Variable("x", lb=0)
    y = Variable("y", lb=0)

    prob = Problem(name="simple_lp")
    prob.maximize(3 * x + 2 * y)
    prob.subject_to(x + y <= 4)
    prob.subject_to(x <= 2)
    prob.subject_to(y <= 3)

    optimal_values = {"x": 2.0, "y": 2.0}
    optimal_objective = 10.0

    return prob, optimal_values, optimal_objective


@pytest.fixture
def constrained_quadratic() -> tuple[Problem, dict[str, float], float]:
    """Constrained quadratic problem.

    min x² + y²
    s.t. x + y >= 1

    Optimal: (0.5, 0.5) with f* = 0.5
    """
    x = Variable("x", initial=0.0)
    y = Variable("y", initial=0.0)

    prob = Problem(name="constrained_quadratic")
    prob.minimize(x**2 + y**2)
    prob.subject_to(x + y >= 1)

    optimal_values = {"x": 0.5, "y": 0.5}
    optimal_objective = 0.5

    return prob, optimal_values, optimal_objective


# =============================================================================
# Parameterized Fixtures
# =============================================================================


@pytest.fixture(params=[10, 50, 100, 500, 1000])
def scaling_size(request) -> int:
    """Problem sizes for scaling analysis."""
    return request.param


@pytest.fixture
def lp_scaling(scaling_size: int) -> tuple[Problem, float]:
    """Generate LP of given size.

    max sum(c_i * x_i)
    s.t. sum(a_ij * x_j) <= b_i  for all i
         0 <= x_i <= 1  for all i

    Returns problem and expected objective (computed via linprog).
    """
    n = scaling_size
    m = n // 2  # constraints

    np.random.seed(42)
    c = np.random.rand(n)
    A = np.random.rand(m, n)
    b = np.sum(A, axis=1) * 0.5  # Feasible constraints

    variables = [Variable(f"x{i}", lb=0, ub=1) for i in range(n)]

    prob = Problem(name=f"lp_{n}vars")
    prob.maximize(sum(c[i] * variables[i] for i in range(n)))

    for i in range(m):
        constraint_expr = sum(A[i, j] * variables[j] for j in range(n))
        prob.subject_to(constraint_expr <= b[i])

    # Compute expected via scipy linprog
    from scipy.optimize import linprog

    result = linprog(-c, A_ub=A, b_ub=b, bounds=[(0, 1)] * n, method="highs")
    expected = -result.fun if result.success else float("nan")

    return prob, expected


@pytest.fixture
def quadratic_scaling(scaling_size: int) -> tuple[Problem, float]:
    """Generate quadratic problem of given size.

    min sum(x_i²) - sum(x_i)
    Optimal: x_i = 0.5 for all i, f* = -n/4
    """
    n = scaling_size
    variables = [Variable(f"x{i}", initial=0.0) for i in range(n)]

    prob = Problem(name=f"quadratic_{n}vars")
    objective = sum(v**2 for v in variables) - sum(variables)
    prob.minimize(objective)

    expected = -n / 4  # At x_i = 0.5

    return prob, expected
