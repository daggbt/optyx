"""Comparison benchmark: Optyx vs Pyomo.

Compares Optyx against Pyomo for NLP problems.
Pyomo is an optional dependency - tests gracefully skip if not installed.

Install with: uv sync --extra benchmarks
"""

from __future__ import annotations

import numpy as np
import pytest

import sys

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from utils import compare_timing

# Check if pyomo is available
try:
    from pyomo.environ import (
        ConcreteModel,
        Var,
        Objective,
        Constraint,
        minimize as pyomo_minimize,
        maximize as pyomo_maximize,
        SolverFactory,
        value,
        NonNegativeReals,
    )

    # Check for a solver
    try:
        solver = SolverFactory("ipopt")
        if not solver.available():
            solver = SolverFactory("scipy")
        PYOMO_AVAILABLE = solver.available()
    except Exception:
        PYOMO_AVAILABLE = False
except ImportError:
    PYOMO_AVAILABLE = False

from optyx import Variable, Problem


pytestmark = pytest.mark.skipif(
    not PYOMO_AVAILABLE, reason="pyomo or solver not installed"
)


def get_pyomo_solver():
    """Get an available Pyomo solver."""
    for solver_name in ["ipopt", "scipy"]:
        solver = SolverFactory(solver_name)
        if solver.available():
            return solver
    raise RuntimeError("No Pyomo solver available")


class TestNLPComparison:
    """Compare Optyx vs Pyomo for nonlinear programs."""

    def test_rosenbrock(self):
        """Rosenbrock function comparison."""
        # Optyx
        x = Variable("x")
        y = Variable("y")
        prob = Problem(name="rosenbrock")
        prob.minimize((1 - x) ** 2 + 100 * (y - x**2) ** 2)

        x0 = np.array([-1.0, -1.0])
        optyx_sol = prob.solve(x0=x0)

        # Pyomo
        model = ConcreteModel()
        model.x = Var(initialize=-1.0)
        model.y = Var(initialize=-1.0)
        model.obj = Objective(
            expr=(1 - model.x) ** 2 + 100 * (model.y - model.x**2) ** 2,
            sense=pyomo_minimize,
        )

        solver = get_pyomo_solver()
        solver.solve(model, tee=False)

        # Compare solutions
        assert optyx_sol.is_optimal
        assert abs(optyx_sol["x"] - value(model.x)) < 0.1
        assert abs(optyx_sol["y"] - value(model.y)) < 0.1

        # Compare timing
        def optyx_run():
            return prob.solve(x0=x0)

        def pyomo_run():
            solver.solve(model, tee=False)
            return value(model.obj)

        result = compare_timing(optyx_run, pyomo_run, n_warmup=2, n_runs=10)
        print(f"\nRosenbrock - Optyx vs Pyomo:\n{result}")

    def test_constrained_nlp(self):
        """Constrained NLP comparison (vectorized)."""
        # Optyx (vectorized with np.array and np.sum)
        x = np.array([Variable("x", lb=0), Variable("y", lb=0)])
        prob = Problem(name="constrained_nlp")
        prob.minimize(np.sum(x**2))
        prob.subject_to(np.sum(x) >= 1)

        x0 = np.array([0.5, 0.5])
        optyx_sol = prob.solve(x0=x0)

        # Pyomo
        model = ConcreteModel()
        model.x = Var(domain=NonNegativeReals, initialize=0.5)
        model.y = Var(domain=NonNegativeReals, initialize=0.5)
        model.obj = Objective(expr=model.x**2 + model.y**2, sense=pyomo_minimize)
        model.con = Constraint(expr=model.x + model.y >= 1)

        solver = get_pyomo_solver()
        solver.solve(model, tee=False)

        # Compare solutions
        assert optyx_sol.is_optimal
        assert abs(optyx_sol["x"] - 0.5) < 0.1
        assert abs(optyx_sol["y"] - 0.5) < 0.1

        # Compare timing
        def optyx_run():
            return prob.solve(x0=x0)

        def pyomo_run():
            solver.solve(model, tee=False)
            return value(model.obj)

        result = compare_timing(optyx_run, pyomo_run, n_warmup=2, n_runs=10)
        print(f"\nConstrained NLP - Optyx vs Pyomo:\n{result}")


class TestLPComparison:
    """Compare Optyx vs Pyomo for linear programs."""

    def test_simple_lp(self):
        """Simple LP comparison (vectorized)."""
        # Coefficients
        c = np.array([3.0, 2.0])

        # Optyx (vectorized with np.array and @ operator)
        x = np.array([Variable("x", lb=0, ub=6), Variable("y", lb=0, ub=6)])
        prob = Problem(name="simple_lp")
        prob.maximize(c @ x)
        prob.subject_to(np.sum(x) <= 10)

        optyx_sol = prob.solve()

        # Pyomo
        model = ConcreteModel()
        model.x = Var(domain=NonNegativeReals, bounds=(0, 6))
        model.y = Var(domain=NonNegativeReals, bounds=(0, 6))
        model.obj = Objective(expr=3 * model.x + 2 * model.y, sense=pyomo_maximize)
        model.con = Constraint(expr=model.x + model.y <= 10)

        solver = get_pyomo_solver()
        solver.solve(model, tee=False)

        # Compare solutions
        assert optyx_sol.is_optimal
        assert abs(optyx_sol.objective_value - value(model.obj)) < 0.1

        # Compare timing
        def optyx_run():
            return prob.solve()

        def pyomo_run():
            solver.solve(model, tee=False)
            return value(model.obj)

        result = compare_timing(optyx_run, pyomo_run, n_warmup=2, n_runs=10)
        print(f"\nSimple LP - Optyx vs Pyomo:\n{result}")


class TestSyntaxComparison:
    """Compare API syntax and ergonomics."""

    def test_syntax_simplicity(self):
        """Demonstrate syntax differences.

        Optyx uses pure Python operators.
        Pyomo uses a modeling language with explicit components.
        """
        print("\n" + "=" * 60)
        print("SYNTAX COMPARISON: Optyx vs Pyomo")
        print("=" * 60)

        print("\n--- Simple NLP ---")
        print("Optyx:")
        print("  x = Variable('x', initial=0.0)")
        print("  prob = Problem()")
        print("  prob.minimize(x**2 + y**2)")
        print("  prob.subject_to(x + y >= 1)")
        print("  sol = prob.solve()")
        print("  print(sol['x'])")

        print("\nPyomo:")
        print("  model = ConcreteModel()")
        print("  model.x = Var(initialize=0.0)")
        print("  model.y = Var(initialize=0.0)")
        print("  model.obj = Objective(expr=model.x**2 + model.y**2)")
        print("  model.con = Constraint(expr=model.x + model.y >= 1)")
        print("  solver = SolverFactory('ipopt')")
        print("  solver.solve(model)")
        print("  print(value(model.x))")

        print("\n--- Key Differences ---")
        print("• Optyx: 5 lines, fluent API")
        print("• Pyomo: 7 lines, component-based")
        print("• Optyx: Automatic solver selection")
        print("• Pyomo: Manual solver factory")
        print("• Optyx: dict-like solution access")
        print("• Pyomo: value() function required")
        print("• Pyomo: Better for MILP (via CBC, Gurobi)")
        print("• Optyx: Easier for prototyping NLP")


if __name__ == "__main__":
    if not PYOMO_AVAILABLE:
        print("Pyomo not installed or no solver available.")
        print("Install with: uv sync --extra benchmarks")
        print("Also need: conda install -c conda-forge ipopt")
        exit(1)

    print("=" * 60)
    print("OPTYX VS PYOMO COMPARISON")
    print("=" * 60)

    test_nlp = TestNLPComparison()
    test_nlp.test_rosenbrock()
    test_nlp.test_constrained_nlp()

    test_lp = TestLPComparison()
    test_lp.test_simple_lp()

    test_syntax = TestSyntaxComparison()
    test_syntax.test_syntax_simplicity()
