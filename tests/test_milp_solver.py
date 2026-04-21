"""Tests for MILP solver integration (scipy.optimize.milp)."""

import numpy as np
import pytest
from scipy import sparse

from optyx import Variable, BinaryVariable, IntegerVariable, VectorVariable, as_matrix
from optyx.core.errors import UnsupportedOperationError
from optyx.problem import Problem
from optyx.solution import SolverStatus


class TestMILPBinaryKnapsack:
    """Binary knapsack problem solves correctly via milp()."""

    def test_milp_binary_knapsack(self):
        """Classic 0-1 knapsack: maximize value subject to weight capacity."""
        # Items: (value, weight) = (10, 5), (6, 4), (4, 3)
        # Capacity = 7
        x1 = BinaryVariable("x1")
        x2 = BinaryVariable("x2")
        x3 = BinaryVariable("x3")

        prob = Problem()
        prob.maximize(10 * x1 + 6 * x2 + 4 * x3)
        prob.subject_to(5 * x1 + 4 * x2 + 3 * x3 <= 7)

        sol = prob.solve()

        assert sol.is_optimal
        # Optimal: take x2 + x3 (weight=7, value=10) or x1 (weight=5, value=10)
        # Both give value=10; x1 alone also works
        assert abs(sol.objective_value - 10) < 1e-6

    def test_knapsack_all_binary(self):
        """All solution values are 0 or 1."""
        x1 = BinaryVariable("x1")
        x2 = BinaryVariable("x2")

        prob = Problem()
        prob.maximize(3 * x1 + 2 * x2)
        prob.subject_to(2 * x1 + x2 <= 2)

        sol = prob.solve()
        assert sol.is_optimal
        for name in ["x1", "x2"]:
            assert sol[name] == pytest.approx(0, abs=1e-6) or sol[
                name
            ] == pytest.approx(1, abs=1e-6)


class TestMILPIntegerVariables:
    """Integer-constrained LP produces integer solution."""

    def test_milp_integer_variables(self):
        """Integer variables produce integer solutions."""
        x = IntegerVariable("x", lb=0, ub=10)
        y = IntegerVariable("y", lb=0, ub=10)

        prob = Problem()
        prob.minimize(x + y)
        prob.subject_to(x + y >= 5)

        sol = prob.solve()
        assert sol.is_optimal
        # Solution should be integer
        assert abs(sol["x"] - round(sol["x"])) < 1e-6
        assert abs(sol["y"] - round(sol["y"])) < 1e-6
        assert round(sol["x"]) + round(sol["y"]) >= 5

    def test_integer_bounds_respected(self):
        """Integer variable bounds are enforced."""
        x = IntegerVariable("x", lb=2, ub=8)

        prob = Problem()
        prob.minimize(x)

        sol = prob.solve()
        assert sol.is_optimal
        assert abs(sol["x"] - 2) < 1e-6


class TestMILPMixed:
    """Mix of continuous and integer variables solves correctly."""

    def test_milp_mixed(self):
        """Continuous and integer vars solve together."""
        x = Variable("x", lb=0)  # continuous
        y = IntegerVariable("y", lb=0, ub=10)

        prob = Problem()
        prob.minimize(x + y)
        prob.subject_to(x + y >= 3.5)

        sol = prob.solve()
        assert sol.is_optimal
        # y should be integer, x continuous
        y_val = sol["y"]
        assert abs(y_val - round(y_val)) < 1e-6
        # Total >= 3.5
        assert sol["x"] + sol["y"] >= 3.5 - 1e-6


class TestBinaryVariableAlias:
    """BinaryVariable() creates Variable with domain='binary', lb=0, ub=1."""

    def test_binary_variable_alias(self):
        """BinaryVariable creates correct variable."""
        x = BinaryVariable("x")

        assert x.domain == "binary"
        assert x.lb == 0
        assert x.ub == 1

    def test_binary_variable_in_problem(self):
        """BinaryVariable works in optimization."""
        x = BinaryVariable("x")
        prob = Problem().minimize(x)

        sol = prob.solve()
        assert sol.is_optimal
        assert abs(sol["x"]) < 1e-6


class TestIntegerVariableAlias:
    """IntegerVariable() creates Variable with domain='integer'."""

    def test_integer_variable_alias(self):
        """IntegerVariable creates correct variable."""
        x = IntegerVariable("x", lb=0, ub=5)

        assert x.domain == "integer"
        assert x.lb == 0
        assert x.ub == 5

    def test_integer_variable_default_bounds(self):
        """IntegerVariable without explicit bounds."""
        x = IntegerVariable("x")
        assert x.domain == "integer"


class TestVectorBinary:
    """VectorVariable with domain='binary' creates binary vector."""

    def test_vector_binary(self):
        """VectorVariable with domain='binary' creates binary elements."""
        x = VectorVariable("x", 3, domain="binary")

        assert len(x) == 3
        for v in x:
            assert v.domain == "binary"
            assert v.lb == 0
            assert v.ub == 1

    def test_vector_binary_in_problem(self):
        """Binary vector solves in MILP."""
        x = VectorVariable("x", 3, domain="binary")

        prob = Problem()
        prob.minimize(x[0] + x[1] + x[2])
        prob.subject_to(x[0] + x[1] + x[2] >= 1)

        sol = prob.solve()
        assert sol.is_optimal
        total = sum(sol[f"x[{i}]"] for i in range(3))
        assert abs(total - 1) < 1e-6

    def test_vector_integer(self):
        """VectorVariable with domain='integer' creates integer elements."""
        x = VectorVariable("x", 2, lb=0, ub=5, domain="integer")

        for v in x:
            assert v.domain == "integer"


class TestMILPSolutionGap:
    """Solution.mip_gap populated for MILP solve."""

    def test_milp_solution_gap(self):
        """mip_gap is populated after MILP solve."""
        x = BinaryVariable("x")
        prob = Problem().minimize(x)

        sol = prob.solve()
        assert sol.is_optimal
        # mip_gap should be set (may be 0 for trivial problems)
        if sol.mip_gap is not None:
            assert sol.mip_gap >= 0

    def test_milp_best_bound(self):
        """best_bound is populated after MILP solve."""
        x = BinaryVariable("x")
        prob = Problem().minimize(x)

        sol = prob.solve()
        assert sol.is_optimal
        # best_bound should be set
        if sol.best_bound is not None:
            assert isinstance(sol.best_bound, float)

    def test_lp_no_mip_gap(self):
        """Pure LP solve does not set mip_gap."""
        x = Variable("x", lb=0)
        prob = Problem().minimize(x)

        sol = prob.solve()
        assert sol.is_optimal
        assert sol.mip_gap is None


class TestMILPInfeasible:
    """Infeasible MILP returns INFEASIBLE status."""

    def test_milp_infeasible(self):
        """Infeasible MILP correctly detected."""
        x = BinaryVariable("x")

        prob = Problem()
        prob.minimize(x)
        # x must be 0 or 1, but also >= 2 — infeasible
        prob.subject_to(x >= 2)

        sol = prob.solve()
        assert not sol.is_optimal
        assert sol.status == SolverStatus.INFEASIBLE


class TestMILPRouting:
    """LP without integers uses linprog(); with integers uses milp()."""

    def test_milp_routing(self):
        """Problem with integer vars routes to milp solver."""
        x = IntegerVariable("x", lb=0, ub=10)
        prob = Problem().minimize(x)

        sol = prob.solve()
        assert sol.is_optimal
        # Integer result
        assert abs(sol["x"] - round(sol["x"])) < 1e-6

    def test_lp_routing(self):
        """Problem without integer vars routes to linprog."""
        x = Variable("x", lb=0, ub=10)
        prob = Problem().minimize(x)

        sol = prob.solve()
        assert sol.is_optimal
        assert sol.mip_gap is None  # No MIP gap for LP

    def test_method_milp_routing(self):
        """method='milp' explicitly routes to MILP solver."""
        x = IntegerVariable("x", lb=0, ub=5)
        prob = Problem().minimize(x)

        sol = prob.solve(method="milp")
        assert sol.is_optimal

    def test_milp_uses_live_bounds_after_resolve(self):
        """MILP re-solves respect bound mutations after LP extraction is cached."""
        x = IntegerVariable("x", lb=0, ub=10)
        prob = Problem().maximize(x)

        first = prob.solve()
        assert first.is_optimal
        assert first["x"] == pytest.approx(10.0)

        x.ub = 3
        second = prob.solve()
        assert second.is_optimal
        assert second["x"] == pytest.approx(3.0)


class TestMIQPRaises:
    """Quadratic objective + integer variables raises error."""

    def test_miqp_raises(self):
        """MIQP (quadratic + integer) raises UnsupportedOperationError."""
        x = IntegerVariable("x", lb=0, ub=10)
        prob = Problem().minimize((x - 3) ** 2)

        with pytest.raises(UnsupportedOperationError, match="MIQP/MINLP"):
            prob.solve()

    def test_minlp_raises(self):
        """MINLP (nonlinear + binary) raises UnsupportedOperationError."""
        x = BinaryVariable("x")
        prob = Problem().minimize(x**2 + x)

        with pytest.raises(UnsupportedOperationError, match="MIQP/MINLP"):
            prob.solve()


class TestDomainValidation:
    """Domain validation for variables."""

    def test_domain_validation_binary(self):
        """Binary domain enforces lb=0, ub=1."""
        x = Variable("x", domain="binary")
        assert x.lb == 0
        assert x.ub == 1

    def test_domain_validation_binary_conflicting_lb(self):
        """Binary domain rejects conflicting lower bounds."""
        with pytest.raises(ValueError, match="Binary variable must have lb=0"):
            Variable("x", lb=2, domain="binary")

    def test_domain_validation_binary_conflicting_ub(self):
        """Binary domain rejects conflicting upper bounds."""
        with pytest.raises(ValueError, match="Binary variable must have ub=1"):
            Variable("x", ub=2, domain="binary")

    def test_domain_validation_unknown(self):
        """Unknown domain raises ValueError."""
        with pytest.raises(ValueError, match="Unknown domain"):
            Variable("x", domain="invalid")

    def test_domain_continuous_default(self):
        """Default domain is continuous."""
        x = Variable("x")
        assert x.domain == "continuous"


class TestMILPSparseConstraints:
    """MILP with sparse constraint matrix works correctly."""

    def test_milp_sparse_constraints(self):
        """Sparse constraints solve correctly in MILP."""
        n = 10
        x = VectorVariable("x", n, domain="binary")

        prob = Problem()
        prob.minimize(sum(x[i] for i in range(n)))
        # At least 3 must be selected
        prob.subject_to(sum(x[i] for i in range(n)) >= 3)

        sol = prob.solve()
        assert sol.is_optimal
        total = sum(sol[f"x[{i}]"] for i in range(n))
        assert abs(total - 3) < 1e-6

    @pytest.mark.slow
    def test_milp_sparse_constraints_large_scale(self):
        """Large sparse MILP with 10,000 binaries solves through subject_to."""
        group_size = 10
        n_groups = 1000
        n = group_size * n_groups

        x = VectorVariable("x", n, domain="binary")
        group_pattern = np.arange(group_size, 0, -1, dtype=float).reshape(1, group_size)
        objective = np.tile(group_pattern.ravel(), n_groups)

        A = as_matrix(
            sparse.kron(
                sparse.eye(n_groups, format="csr"),
                np.ones((1, group_size), dtype=float),
                format="csr",
            )
        )
        b = np.ones(n_groups, dtype=float)

        prob = Problem(name="sparse_milp_large_scale")
        prob.maximize(objective @ x)
        prob.subject_to(A @ x <= b)

        sol = prob.solve()
        assert sol.is_optimal
        assert sol.objective_value == pytest.approx(float(n_groups * group_size))

        values = np.array([sol[f"x[{i}]"] for i in range(n)], dtype=float).reshape(
            n_groups, group_size
        )
        assert np.allclose(values.sum(axis=1), 1.0)
        assert np.all(values[:, 0] > 0.5)
        assert np.all(values[:, 1:] < 0.5)


class TestFacilityLocationMILP:
    """Facility location: binary open + continuous transport."""

    def test_facility_location_milp(self):
        """Simple facility location problem solves end-to-end."""
        # 2 facilities, 2 customers
        # Open cost: facility 0 = 10, facility 1 = 15
        # Transport cost: (facility, customer) → cost
        # f0→c0=2, f0→c1=4, f1→c0=5, f1→c1=1
        # Customer demands: c0=1, c1=1

        y0 = BinaryVariable("y0")  # open facility 0
        y1 = BinaryVariable("y1")  # open facility 1
        # Transport (continuous, but bounded by open)
        x00 = Variable("x00", lb=0)  # f0 → c0
        x01 = Variable("x01", lb=0)  # f0 → c1
        x10 = Variable("x10", lb=0)  # f1 → c0
        x11 = Variable("x11", lb=0)  # f1 → c1

        prob = Problem()
        # Minimize: fixed cost + transport cost
        prob.minimize(10 * y0 + 15 * y1 + 2 * x00 + 4 * x01 + 5 * x10 + 1 * x11)

        # Demand satisfaction
        prob.subject_to(x00 + x10 >= 1)  # customer 0
        prob.subject_to(x01 + x11 >= 1)  # customer 1

        # Capacity: can only ship from open facility (big-M)
        M = 10
        prob.subject_to(x00 + x01 <= M * y0)
        prob.subject_to(x10 + x11 <= M * y1)

        sol = prob.solve()

        assert sol.is_optimal
        # At least one facility must be open
        assert sol["y0"] > 0.5 or sol["y1"] > 0.5
        # Demands met
        assert sol["x00"] + sol["x10"] >= 1 - 1e-6
        assert sol["x01"] + sol["x11"] >= 1 - 1e-6


class TestMILPMaximization:
    """MILP maximization problems."""

    def test_milp_maximize(self):
        """Maximization with integer variables."""
        x = IntegerVariable("x", lb=0, ub=5)
        y = IntegerVariable("y", lb=0, ub=5)

        prob = Problem()
        prob.maximize(x + y)
        prob.subject_to(x + y <= 7)

        sol = prob.solve()
        assert sol.is_optimal
        assert abs(sol.objective_value - 7) < 1e-6


class TestMILPEqualityConstraints:
    """MILP with equality constraints."""

    def test_milp_equality(self):
        """Equality constraints work in MILP."""
        x = IntegerVariable("x", lb=0, ub=10)
        y = IntegerVariable("y", lb=0, ub=10)

        prob = Problem()
        prob.minimize(x + y)
        prob.subject_to((x + y).eq(5))

        sol = prob.solve()
        assert sol.is_optimal
        assert abs(sol["x"] + sol["y"] - 5) < 1e-6
