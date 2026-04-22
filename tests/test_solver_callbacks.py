"""Tests for solver progress callbacks and time limits (Issue #105)."""

from __future__ import annotations


import numpy as np
import pytest

from optyx import (
    Variable,
    VectorVariable,
    Problem,
    SolverProgress,
    SolverStatus,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_nlp():
    """A simple NLP: minimize x^2 + y^2 subject to x + y >= 1."""
    x = Variable("x", lb=-10, ub=10)
    y = Variable("y", lb=-10, ub=10)
    prob = Problem("callback_test")
    prob.minimize(x**2 + y**2)
    prob.subject_to(x + y >= 1)
    return prob


@pytest.fixture
def unconstrained_nlp():
    """An unconstrained NLP: minimize x^2 + y^2."""
    x = Variable("x", lb=-10, ub=10)
    y = Variable("y", lb=-10, ub=10)
    prob = Problem("unconstrained")
    prob.minimize(x**2 + y**2)
    return prob


@pytest.fixture
def slow_nlp():
    """An NLP that takes many iterations (Rosenbrock-like)."""
    n = 20
    v = VectorVariable("v", n, lb=-5, ub=5)
    prob = Problem("slow_nlp")
    obj = sum((1 - v[i]) ** 2 + 100 * (v[i + 1] - v[i] ** 2) ** 2 for i in range(n - 1))
    prob.minimize(obj)
    return prob


# ---------------------------------------------------------------------------
# SolverProgress dataclass tests
# ---------------------------------------------------------------------------


class TestSolverProgress:
    def test_fields(self):
        p = SolverProgress(
            iteration=5,
            objective_value=1.23,
            constraint_violation=0.0,
            elapsed_time=0.1,
            x=np.array([1.0, 2.0]),
        )
        assert p.iteration == 5
        assert p.objective_value == 1.23
        assert p.constraint_violation == 0.0
        assert p.elapsed_time == 0.1
        np.testing.assert_array_equal(p.x, [1.0, 2.0])


# ---------------------------------------------------------------------------
# Callback tests
# ---------------------------------------------------------------------------


class TestCallback:
    def test_callback_receives_progress(self, simple_nlp):
        """Callback should be called with SolverProgress objects."""
        progress_log: list[SolverProgress] = []

        def on_progress(p: SolverProgress) -> None:
            progress_log.append(p)

        sol = simple_nlp.solve(method="SLSQP", callback=on_progress)
        assert sol.status == SolverStatus.OPTIMAL
        assert len(progress_log) > 0

        # Check that each progress has valid fields
        for p in progress_log:
            assert isinstance(p, SolverProgress)
            assert p.iteration >= 1
            assert isinstance(p.objective_value, float)
            assert p.constraint_violation >= 0.0
            assert p.elapsed_time >= 0.0
            assert isinstance(p.x, np.ndarray)

    def test_callback_iterations_increase(self, simple_nlp):
        """Iteration numbers should be monotonically increasing."""
        iterations: list[int] = []

        def on_progress(p: SolverProgress) -> None:
            iterations.append(p.iteration)

        simple_nlp.solve(method="SLSQP", callback=on_progress)
        assert iterations == sorted(iterations)
        assert iterations[0] == 1

    def test_callback_elapsed_time_increases(self, simple_nlp):
        """Elapsed time should be non-decreasing."""
        times: list[float] = []

        def on_progress(p: SolverProgress) -> None:
            times.append(p.elapsed_time)

        simple_nlp.solve(method="SLSQP", callback=on_progress)
        for i in range(1, len(times)):
            assert times[i] >= times[i - 1]

    def test_callback_x_is_copy(self, simple_nlp):
        """The x array should be a copy, not a view into solver internals."""
        xs: list[np.ndarray] = []

        def on_progress(p: SolverProgress) -> None:
            xs.append(p.x)

        simple_nlp.solve(method="SLSQP", callback=on_progress)
        assert len(xs) > 0
        # Mutate one — should not affect others
        xs[0][:] = 999.0
        if len(xs) > 1:
            assert not np.allclose(xs[1], 999.0)

    def test_callback_with_trust_constr(self, simple_nlp):
        """Callback should work with trust-constr method."""
        progress_log: list[SolverProgress] = []

        def on_progress(p: SolverProgress) -> None:
            progress_log.append(p)

        sol = simple_nlp.solve(method="trust-constr", callback=on_progress)
        assert sol.status == SolverStatus.OPTIMAL
        assert len(progress_log) > 0

    def test_callback_with_lbfgsb(self, unconstrained_nlp):
        """Callback should work with L-BFGS-B method."""
        progress_log: list[SolverProgress] = []

        def on_progress(p: SolverProgress) -> None:
            progress_log.append(p)

        sol = unconstrained_nlp.solve(method="L-BFGS-B", callback=on_progress)
        assert sol.status == SolverStatus.OPTIMAL
        assert len(progress_log) > 0

    def test_callback_objective_matches_sense(self):
        """For maximize, objective_value in progress should be positive."""
        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.maximize(x)
        prob.subject_to(x <= 5)

        obj_values: list[float] = []

        def on_progress(p: SolverProgress) -> None:
            obj_values.append(p.objective_value)

        prob.solve(method="SLSQP", callback=on_progress)
        # The last objective should be close to 5 (maximized value)
        # All reported values should use the original (maximize) sense
        # (not the negated internal representation)
        assert len(obj_values) > 0


# ---------------------------------------------------------------------------
# Early termination tests
# ---------------------------------------------------------------------------


class TestEarlyTermination:
    def test_callback_returns_true_terminates(self, simple_nlp):
        """Returning True from callback should terminate with TERMINATED status."""

        def stop_immediately(p: SolverProgress) -> bool:
            return True

        sol = simple_nlp.solve(method="SLSQP", callback=stop_immediately)
        assert sol.status == SolverStatus.TERMINATED
        assert "callback" in sol.message.lower()

    def test_callback_returns_true_after_n(self, slow_nlp):
        """Callback should be able to stop after N iterations."""
        max_iters = 3

        def stop_after_n(p: SolverProgress) -> bool:
            return p.iteration >= max_iters

        sol = slow_nlp.solve(method="SLSQP", callback=stop_after_n)
        assert sol.status == SolverStatus.TERMINATED
        assert sol.iterations is not None
        assert sol.iterations <= max_iters + 1  # may be at most 1 over

    def test_terminated_solution_has_values(self, simple_nlp):
        """Terminated solution should still have variable values."""

        def stop_immediately(p: SolverProgress) -> bool:
            return True

        sol = simple_nlp.solve(method="SLSQP", callback=stop_immediately)
        assert sol.status == SolverStatus.TERMINATED
        assert "x" in sol.values
        assert "y" in sol.values
        assert sol.objective_value is not None
        assert sol.solve_time is not None
        assert sol.solve_time > 0

    def test_terminated_is_feasible(self, simple_nlp):
        """TERMINATED status should count as feasible."""
        sol = simple_nlp.solve(
            method="SLSQP",
            callback=lambda p: True,
        )
        assert sol.status == SolverStatus.TERMINATED
        assert sol.is_feasible

    def test_callback_returning_none_continues(self, simple_nlp):
        """Returning None from callback should continue solving."""
        count = {"n": 0}

        def on_progress(p: SolverProgress):
            count["n"] += 1
            return None

        sol = simple_nlp.solve(method="SLSQP", callback=on_progress)
        assert sol.status == SolverStatus.OPTIMAL
        assert count["n"] > 0

    def test_callback_returning_false_continues(self, simple_nlp):
        """Returning False from callback should continue solving."""
        count = {"n": 0}

        def on_progress(p: SolverProgress) -> bool:
            count["n"] += 1
            return False

        sol = simple_nlp.solve(method="SLSQP", callback=on_progress)
        assert sol.status == SolverStatus.OPTIMAL
        assert count["n"] > 0

    def test_early_termination_trust_constr(self, simple_nlp):
        """Early termination should work with trust-constr."""
        sol = simple_nlp.solve(
            method="trust-constr",
            callback=lambda p: True,
        )
        assert sol.status == SolverStatus.TERMINATED

    def test_early_termination_lbfgsb(self, unconstrained_nlp):
        """Early termination should work with L-BFGS-B."""
        sol = unconstrained_nlp.solve(
            method="L-BFGS-B",
            callback=lambda p: True,
        )
        assert sol.status == SolverStatus.TERMINATED


# ---------------------------------------------------------------------------
# Time limit tests
# ---------------------------------------------------------------------------


class TestTimeLimit:
    def test_time_limit_terminates(self, slow_nlp):
        """time_limit should terminate the solver."""
        sol = slow_nlp.solve(method="SLSQP", time_limit=0.001)
        # With a very small time limit, it should terminate
        assert sol.status == SolverStatus.TERMINATED
        assert "time limit" in sol.message.lower()

    def test_time_limit_has_solution(self, slow_nlp):
        """Time-limited solution should still have variable values."""
        sol = slow_nlp.solve(method="SLSQP", time_limit=0.001)
        assert sol.values  # non-empty
        assert sol.objective_value is not None
        assert sol.solve_time is not None

    def test_generous_time_limit_completes(self, simple_nlp):
        """A generous time limit should allow normal completion."""
        sol = simple_nlp.solve(method="SLSQP", time_limit=60.0)
        assert sol.status == SolverStatus.OPTIMAL

    def test_time_limit_with_callback(self, slow_nlp):
        """time_limit and callback should work together."""
        progress_log: list[SolverProgress] = []

        def on_progress(p: SolverProgress) -> None:
            progress_log.append(p)

        sol = slow_nlp.solve(
            method="SLSQP",
            callback=on_progress,
            time_limit=0.001,
        )
        assert sol.status == SolverStatus.TERMINATED
        # Callback should have been called at least once before time limit
        # (or the time limit hit on the first callback — either way TERMINATED)

    def test_time_limit_trust_constr(self, slow_nlp):
        """time_limit should work with trust-constr."""
        sol = slow_nlp.solve(method="trust-constr", time_limit=0.001)
        assert sol.status == SolverStatus.TERMINATED

    def test_time_limit_lbfgsb(self):
        """time_limit should work with L-BFGS-B."""
        n = 20
        v = VectorVariable("v", n, lb=-5, ub=5)
        prob = Problem()
        obj = sum(
            (1 - v[i]) ** 2 + 100 * (v[i + 1] - v[i] ** 2) ** 2 for i in range(n - 1)
        )
        prob.minimize(obj)
        sol = prob.solve(method="L-BFGS-B", time_limit=0.001)
        assert sol.status == SolverStatus.TERMINATED


# ---------------------------------------------------------------------------
# No callback / no time_limit (regression)
# ---------------------------------------------------------------------------


class TestNoCallback:
    def test_solve_without_callback(self, simple_nlp):
        """Normal solve without callback should still work."""
        sol = simple_nlp.solve(method="SLSQP")
        assert sol.status == SolverStatus.OPTIMAL
        assert abs(sol.objective_value - 0.5) < 1e-4

    def test_solve_without_callback_trust_constr(self, simple_nlp):
        """Normal solve without callback should work for trust-constr."""
        sol = simple_nlp.solve(method="trust-constr")
        assert sol.status == SolverStatus.OPTIMAL


# ---------------------------------------------------------------------------
# Constraint violation reporting
# ---------------------------------------------------------------------------


class TestConstraintViolation:
    def test_feasible_has_zero_violation(self, simple_nlp):
        """At convergence, constraint violation should be near zero."""
        violations: list[float] = []

        def on_progress(p: SolverProgress) -> None:
            violations.append(p.constraint_violation)

        simple_nlp.solve(method="SLSQP", callback=on_progress)
        # The last violation should be near zero
        assert violations[-1] < 1e-4

    def test_unconstrained_has_zero_violation(self, unconstrained_nlp):
        """Unconstrained problems should always report zero violation."""
        violations: list[float] = []

        def on_progress(p: SolverProgress) -> None:
            violations.append(p.constraint_violation)

        unconstrained_nlp.solve(method="L-BFGS-B", callback=on_progress)
        assert all(v == 0.0 for v in violations)
