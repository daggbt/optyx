"""Tests for incremental model modification and warm start (Issue #104).

Tests cover:
- remove_constraint() by index and name
- Selective cache invalidation
- Warm start (previous solution as x0)
- Bounds freshness (LP bounds never stale)
- reset() clears warm start state
- No cache staleness after modifications
"""

import numpy as np
import pytest

from optyx import Variable
from optyx.core.vectors import VectorVariable
from optyx.problem import Problem
from optyx.solution import SolverStatus


# ---------------------------------------------------------------------------
# remove_constraint() tests
# ---------------------------------------------------------------------------

class TestRemoveConstraintByIndex:
    """Test removing constraints by integer index."""

    def test_remove_first_constraint(self):
        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.minimize(x)
        prob.subject_to(x >= 3)
        prob.subject_to(x >= 5)
        assert prob.n_constraints == 2

        prob.remove_constraint(0)
        assert prob.n_constraints == 1
        # Only x >= 5 remains
        sol = prob.solve()
        assert sol.status == SolverStatus.OPTIMAL
        assert sol.values["x"] == pytest.approx(5.0, abs=1e-4)

    def test_remove_last_constraint(self):
        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.minimize(x)
        prob.subject_to(x >= 5)
        prob.subject_to(x >= 3)

        prob.remove_constraint(1)
        assert prob.n_constraints == 1
        # Only x >= 5 remains
        sol = prob.solve()
        assert sol.status == SolverStatus.OPTIMAL
        assert sol.values["x"] == pytest.approx(5.0, abs=1e-4)

    def test_remove_only_constraint(self):
        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.minimize(x)
        prob.subject_to(x >= 5)

        prob.remove_constraint(0)
        assert prob.n_constraints == 0
        # With no constraints, minimum is at lb=0
        sol = prob.solve()
        assert sol.status == SolverStatus.OPTIMAL
        assert sol.values["x"] == pytest.approx(0.0, abs=1e-4)

    def test_remove_invalid_index_raises(self):
        x = Variable("x")
        prob = Problem()
        prob.minimize(x)
        prob.subject_to(x >= 0)

        with pytest.raises(IndexError, match="out of range"):
            prob.remove_constraint(5)

    def test_remove_negative_index_raises(self):
        x = Variable("x")
        prob = Problem()
        prob.minimize(x)
        prob.subject_to(x >= 0)

        with pytest.raises(IndexError, match="out of range"):
            prob.remove_constraint(-1)

    def test_remove_from_empty_raises(self):
        prob = Problem()
        prob.minimize(Variable("x"))

        with pytest.raises(IndexError, match="out of range"):
            prob.remove_constraint(0)

    def test_remove_returns_self(self):
        x = Variable("x")
        prob = Problem()
        prob.minimize(x)
        prob.subject_to(x >= 0)
        result = prob.remove_constraint(0)
        assert result is prob

    def test_remove_invalid_type_raises(self):
        prob = Problem()
        prob.minimize(Variable("x"))
        prob.subject_to(Variable("x") >= 0)

        with pytest.raises(TypeError, match="Expected int or str"):
            prob.remove_constraint(3.14)  # type: ignore[arg-type]


class TestRemoveConstraintByName:
    """Test removing constraints by name."""

    def test_remove_named_constraint(self):
        from optyx.constraints import Constraint

        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.minimize(x)

        c1 = Constraint(expr=(x - 3).expr if hasattr(x - 3, "expr") else (x - 3), sense=">=", name="lower")
        c2 = Constraint(expr=(x - 7).expr if hasattr(x - 7, "expr") else (x - 7), sense="<=", name="upper")
        prob.subject_to(c1)
        prob.subject_to(c2)

        prob.remove_constraint("lower")
        assert prob.n_constraints == 1
        assert prob.constraints[0].name == "upper"

    def test_remove_nonexistent_name_raises(self):
        x = Variable("x")
        prob = Problem()
        prob.minimize(x)
        prob.subject_to(x >= 0)

        with pytest.raises(KeyError, match="No constraint named"):
            prob.remove_constraint("nonexistent")


class TestRemoveConstraintCaching:
    """Test that caches are properly invalidated after remove_constraint."""

    def test_solve_after_remove(self):
        """Problem solves correctly after removing a constraint."""
        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.minimize(x)
        prob.subject_to(x >= 5)
        prob.subject_to(x >= 8)

        sol1 = prob.solve()
        assert sol1.values["x"] == pytest.approx(8.0, abs=1e-4)

        prob.remove_constraint(1)  # Remove x >= 8
        sol2 = prob.solve()
        assert sol2.values["x"] == pytest.approx(5.0, abs=1e-4)

    def test_add_then_remove_then_solve(self):
        """Incremental add + remove + solve cycle works."""
        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.minimize(x)

        # Solve unconstrained
        sol = prob.solve()
        assert sol.values["x"] == pytest.approx(0.0, abs=1e-4)

        # Add constraint and solve
        prob.subject_to(x >= 5)
        sol = prob.solve()
        assert sol.values["x"] == pytest.approx(5.0, abs=1e-4)

        # Remove constraint and solve
        prob.remove_constraint(0)
        sol = prob.solve()
        assert sol.values["x"] == pytest.approx(0.0, abs=1e-4)

    def test_nlp_selective_cache_invalidation(self):
        """Objective cache is preserved when only constraints change (NLP)."""
        x = Variable("x", lb=-10, ub=10)
        prob = Problem()
        prob.minimize(x**2)
        prob.subject_to(x >= 3)

        sol = prob.solve(method="SLSQP")
        assert sol.values["x"] == pytest.approx(3.0, abs=1e-4)

        # Solver cache should exist with obj_fn
        assert prob._solver_cache is not None
        obj_fn = prob._solver_cache.get("obj_fn")
        assert obj_fn is not None

        # Remove the constraint
        prob.remove_constraint(0)

        # Objective cache should be preserved
        assert prob._solver_cache is not None
        assert prob._solver_cache.get("obj_fn") is obj_fn
        # Constraint cache should be cleared
        assert "scipy_constraints" not in prob._solver_cache

        # Solve again — should work correctly using rebuilt constraints
        sol = prob.solve(method="SLSQP")
        assert sol.values["x"] == pytest.approx(0.0, abs=1e-4)

    def test_add_constraint_preserves_objective_cache(self):
        """Adding a constraint preserves objective cache too."""
        x = Variable("x", lb=-10, ub=10)
        prob = Problem()
        prob.minimize(x**2)

        sol = prob.solve(method="SLSQP")
        assert sol.values["x"] == pytest.approx(0.0, abs=1e-4)

        obj_fn = prob._solver_cache["obj_fn"]

        # Add constraint — objective cache should be preserved
        prob.subject_to(x >= 3)
        assert prob._solver_cache is not None
        assert prob._solver_cache.get("obj_fn") is obj_fn

        sol = prob.solve(method="SLSQP")
        assert sol.values["x"] == pytest.approx(3.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Warm start tests
# ---------------------------------------------------------------------------

class TestWarmStart:
    """Test warm start (using previous solution as initial point)."""

    def test_warm_start_stores_solution(self):
        """After solve, _last_solution is populated."""
        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.minimize((x - 5) ** 2)

        sol = prob.solve()
        assert sol.status == SolverStatus.OPTIMAL
        assert prob._last_solution is not None
        assert prob._last_solution[0] == pytest.approx(5.0, abs=1e-4)

    def test_warm_start_used_as_x0(self):
        """Warm start uses previous solution for next solve."""
        x = Variable("x", lb=0, ub=10)
        y = Variable("y", lb=0, ub=10)
        prob = Problem()
        prob.minimize((x - 3) ** 2 + (y - 4) ** 2)
        prob.subject_to(x + y >= 5)

        sol1 = prob.solve(method="SLSQP")
        assert sol1.status == SolverStatus.OPTIMAL
        assert prob._last_solution is not None

        # Solve again with warm start — should converge faster or same result
        sol2 = prob.solve(method="SLSQP")
        assert sol2.status == SolverStatus.OPTIMAL
        assert sol2.values["x"] == pytest.approx(sol1.values["x"], abs=1e-4)
        assert sol2.values["y"] == pytest.approx(sol1.values["y"], abs=1e-4)

    def test_warm_start_disabled(self):
        """warm_start=False ignores previous solution."""
        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.minimize((x - 5) ** 2)

        sol1 = prob.solve(method="SLSQP")
        assert prob._last_solution is not None

        # With warm_start=False, x0 should be computed fresh (not from _last_solution)
        sol2 = prob.solve(method="SLSQP", warm_start=False)
        assert sol2.status == SolverStatus.OPTIMAL
        assert sol2.values["x"] == pytest.approx(5.0, abs=1e-4)

    def test_warm_start_after_constraint_change(self):
        """Warm start works after adding/removing constraints."""
        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.minimize((x - 5) ** 2)
        prob.subject_to(x >= 7)

        sol1 = prob.solve(method="SLSQP")
        assert sol1.values["x"] == pytest.approx(7.0, abs=1e-4)

        # Remove the constraint
        prob.remove_constraint(0)

        # Warm start should use x=7 as x0, but converge to x=5
        sol2 = prob.solve(method="SLSQP")
        assert sol2.values["x"] == pytest.approx(5.0, abs=1e-4)

    def test_warm_start_lp(self):
        """LP solver stores solution for warm start state."""
        x = Variable("x", lb=0, ub=10)
        y = Variable("y", lb=0, ub=10)
        prob = Problem()
        prob.minimize(x + y)
        prob.subject_to(x + y >= 5)

        sol = prob.solve()
        assert sol.status == SolverStatus.OPTIMAL
        # Solution should be stored
        assert prob._last_solution is not None

    def test_explicit_x0_overrides_warm_start(self):
        """An explicit x0 takes precedence over warm start."""
        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.minimize((x - 5) ** 2)

        sol1 = prob.solve(method="SLSQP")
        assert prob._last_solution is not None

        # Pass explicit x0 — should use it instead of warm start
        sol2 = prob.solve(method="SLSQP", x0=np.array([1.0]))
        assert sol2.status == SolverStatus.OPTIMAL
        assert sol2.values["x"] == pytest.approx(5.0, abs=1e-4)


# ---------------------------------------------------------------------------
# reset() tests
# ---------------------------------------------------------------------------

class TestReset:
    """Test that reset() clears warm start state."""

    def test_reset_clears_warm_start(self):
        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.minimize((x - 5) ** 2)

        prob.solve(method="SLSQP")
        assert prob._last_solution is not None

        prob.reset()
        assert prob._last_solution is None
        assert prob._solver_cache is None
        assert prob._lp_cache is None

    def test_solve_after_reset(self):
        """Problem solves correctly after reset."""
        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.minimize((x - 5) ** 2)

        sol1 = prob.solve(method="SLSQP")
        prob.reset()
        sol2 = prob.solve(method="SLSQP")

        assert sol2.status == SolverStatus.OPTIMAL
        assert sol2.values["x"] == pytest.approx(5.0, abs=1e-4)


# ---------------------------------------------------------------------------
# LP bounds freshness tests
# ---------------------------------------------------------------------------

class TestBoundsFreshness:
    """Test that LP solver always uses fresh bounds from variables."""

    def test_lp_bounds_update_respected(self):
        """Changing variable bounds is reflected in subsequent LP solves."""
        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.minimize(x)

        sol1 = prob.solve()
        assert sol1.status == SolverStatus.OPTIMAL
        assert sol1.values["x"] == pytest.approx(0.0, abs=1e-4)

        # Change lower bound — should be respected even with cached LP data
        x.lb = 5.0
        sol2 = prob.solve()
        assert sol2.status == SolverStatus.OPTIMAL
        assert sol2.values["x"] == pytest.approx(5.0, abs=1e-4)

    def test_lp_bounds_tighten_and_relax(self):
        """Bounds can be tightened and relaxed between solves."""
        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.minimize(x)

        # Baseline
        sol = prob.solve()
        assert sol.values["x"] == pytest.approx(0.0, abs=1e-4)

        # Tighten
        x.lb = 3.0
        sol = prob.solve()
        assert sol.values["x"] == pytest.approx(3.0, abs=1e-4)

        # Relax back
        x.lb = 0.0
        sol = prob.solve()
        assert sol.values["x"] == pytest.approx(0.0, abs=1e-4)

    def test_upper_bound_update(self):
        """Changing upper bound is reflected in LP solve."""
        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.maximize(x)

        sol1 = prob.solve()
        assert sol1.values["x"] == pytest.approx(10.0, abs=1e-4)

        x.ub = 5.0
        sol2 = prob.solve()
        assert sol2.values["x"] == pytest.approx(5.0, abs=1e-4)

    def test_fix_and_unfix_variable(self):
        """Fix a variable (lb == ub) then unfix it."""
        x = Variable("x", lb=0, ub=10)
        y = Variable("y", lb=0, ub=10)
        prob = Problem()
        prob.minimize(x + y)

        sol = prob.solve()
        assert sol.values["x"] == pytest.approx(0.0, abs=1e-4)

        # Fix x = 5
        x.lb = 5.0
        x.ub = 5.0
        sol = prob.solve()
        assert sol.values["x"] == pytest.approx(5.0, abs=1e-4)
        assert sol.values["y"] == pytest.approx(0.0, abs=1e-4)

        # Unfix x
        x.lb = 0.0
        x.ub = 10.0
        sol = prob.solve()
        assert sol.values["x"] == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# NLP bounds freshness (already correct, but verify)
# ---------------------------------------------------------------------------

class TestNLPBoundsFreshness:
    """Verify NLP solver reads fresh bounds (was already correct)."""

    def test_nlp_bounds_update(self):
        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.minimize(x**2)

        sol = prob.solve(method="SLSQP")
        assert sol.values["x"] == pytest.approx(0.0, abs=1e-4)

        x.lb = 3.0
        sol = prob.solve(method="SLSQP")
        assert sol.values["x"] == pytest.approx(3.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Incremental add/remove cycle tests
# ---------------------------------------------------------------------------

class TestIncrementalCycles:
    """Test repeated add/remove/solve cycles."""

    def test_multiple_incremental_modifications(self):
        """Multiple rounds of add/remove with solves in between."""
        x = Variable("x", lb=0, ub=20)
        prob = Problem()
        prob.minimize(x)

        # Solve unconstrained
        sol = prob.solve()
        assert sol.values["x"] == pytest.approx(0.0, abs=1e-4)

        # Add x >= 5
        prob.subject_to(x >= 5)
        sol = prob.solve()
        assert sol.values["x"] == pytest.approx(5.0, abs=1e-4)

        # Add x >= 10
        prob.subject_to(x >= 10)
        sol = prob.solve()
        assert sol.values["x"] == pytest.approx(10.0, abs=1e-4)

        # Remove x >= 5 (index 0)
        prob.remove_constraint(0)
        sol = prob.solve()
        # x >= 10 is now index 0
        assert sol.values["x"] == pytest.approx(10.0, abs=1e-4)

        # Remove x >= 10
        prob.remove_constraint(0)
        sol = prob.solve()
        assert sol.values["x"] == pytest.approx(0.0, abs=1e-4)

    def test_nlp_incremental_cycle(self):
        """NLP incremental add/remove cycle."""
        x = Variable("x", lb=-10, ub=10)
        prob = Problem()
        prob.minimize(x**2)

        sol = prob.solve(method="SLSQP")
        assert sol.values["x"] == pytest.approx(0.0, abs=1e-4)

        prob.subject_to(x >= 3)
        sol = prob.solve(method="SLSQP")
        assert sol.values["x"] == pytest.approx(3.0, abs=1e-4)

        prob.subject_to(x <= 2)
        # Infeasible: x >= 3 and x <= 2
        sol = prob.solve(method="SLSQP")
        # Solver may return infeasible or a compromised solution
        # Just verify it completes without error

        prob.remove_constraint(0)  # Remove x >= 3
        sol = prob.solve(method="SLSQP")
        # Now only x <= 2 remains
        assert sol.values["x"] == pytest.approx(0.0, abs=1e-4)

    def test_vector_variable_incremental(self):
        """Incremental modification with VectorVariable."""
        x = VectorVariable("x", 3, lb=0, ub=10)
        prob = Problem()
        prob.minimize(x[0] + x[1] + x[2])

        sol = prob.solve()
        assert sol.status == SolverStatus.OPTIMAL

        # Add sum constraint
        prob.subject_to(x[0] + x[1] + x[2] >= 10)
        sol = prob.solve()
        total = sol.values["x[0]"] + sol.values["x[1]"] + sol.values["x[2]"]
        assert total == pytest.approx(10.0, abs=1e-4)

        # Remove and verify
        prob.remove_constraint(0)
        sol = prob.solve()
        total = sol.values["x[0]"] + sol.values["x[1]"] + sol.values["x[2]"]
        assert total == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Cache staleness tests
# ---------------------------------------------------------------------------

class TestNoCacheStaleness:
    """Ensure no stale cached data is ever used."""

    def test_lp_cache_not_stale_after_remove(self):
        """LP cache is invalidated after remove_constraint."""
        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.minimize(x)
        prob.subject_to(x >= 5)

        prob.solve()
        assert prob._lp_cache is not None

        prob.remove_constraint(0)
        assert prob._lp_cache is None  # Should be invalidated

    def test_linearity_cache_not_stale(self):
        """Linearity cache invalidated when constraints change."""
        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.minimize(x)
        prob.subject_to(x >= 5)

        _ = prob._is_linear_problem()
        assert prob._is_linear_cache is not None

        prob.remove_constraint(0)
        assert prob._is_linear_cache is None

    def test_variables_cache_not_stale(self):
        """Variable list recalculated after remove_constraint."""
        x = Variable("x", lb=0, ub=10)
        y = Variable("y", lb=0, ub=10)
        prob = Problem()
        prob.minimize(x)
        prob.subject_to(y >= 5)  # y only appears in constraint

        vars_before = prob.variables
        assert any(v.name == "y" for v in vars_before)

        prob.remove_constraint(0)
        vars_after = prob.variables
        # y should no longer be in variables (only appears in removed constraint)
        assert not any(v.name == "y" for v in vars_after)
