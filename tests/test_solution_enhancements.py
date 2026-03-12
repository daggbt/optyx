"""Tests for Solution/Problem enhancements (Issue #98).

Tests cover:
1. Solution.to_dict(), to_json(), from_json()
2. Problem.reset()
3. SolverStatus.TERMINATED
Also covers print_vars() (roadmap 3.7).
"""

import json
import os
import tempfile

import numpy as np
import pytest

from optyx import Constant, Problem, Variable, VectorVariable
from optyx.solution import Solution, SolverStatus


# ============================================================
# 1. Solution Serialization
# ============================================================


class TestSolutionToDict:
    """Tests for Solution.to_dict()."""

    def test_to_dict_basic(self):
        """to_dict returns all fields."""
        sol = Solution(
            status=SolverStatus.OPTIMAL,
            objective_value=42.0,
            values={"x": 1.0, "y": 2.0},
            iterations=10,
            message="Optimal",
            solve_time=0.5,
        )
        d = sol.to_dict()
        assert d["status"] == "optimal"
        assert d["objective_value"] == 42.0
        assert d["values"] == {"x": 1.0, "y": 2.0}
        assert d["iterations"] == 10
        assert d["message"] == "Optimal"
        assert d["solve_time"] == 0.5

    def test_to_dict_none_fields(self):
        """to_dict handles None fields."""
        sol = Solution(status=SolverStatus.FAILED)
        d = sol.to_dict()
        assert d["status"] == "failed"
        assert d["objective_value"] is None
        assert d["values"] == {}
        assert d["multipliers"] is None

    def test_to_dict_with_multipliers(self):
        """to_dict includes multipliers."""
        sol = Solution(
            status=SolverStatus.OPTIMAL,
            values={"x": 1.0},
            multipliers={"c1": 0.5},
        )
        d = sol.to_dict()
        assert d["multipliers"] == {"c1": 0.5}

    def test_to_dict_all_statuses(self):
        """to_dict works with all SolverStatus values."""
        for status in SolverStatus:
            sol = Solution(status=status)
            d = sol.to_dict()
            assert d["status"] == status.value


class TestSolutionToJson:
    """Tests for Solution.to_json()."""

    def test_to_json_string(self):
        """to_json returns valid JSON string."""
        sol = Solution(
            status=SolverStatus.OPTIMAL,
            objective_value=3.14,
            values={"x": 1.0},
        )
        json_str = sol.to_json()
        data = json.loads(json_str)
        assert data["status"] == "optimal"
        assert data["objective_value"] == 3.14
        assert data["values"]["x"] == 1.0

    def test_to_json_file(self):
        """to_json saves to file when path is given."""
        sol = Solution(
            status=SolverStatus.OPTIMAL,
            objective_value=99.0,
            values={"a": 1.0, "b": 2.0},
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            sol.to_json(path=path)
            with open(path) as f:
                data = json.load(f)
            assert data["status"] == "optimal"
            assert data["objective_value"] == 99.0
            assert data["values"] == {"a": 1.0, "b": 2.0}
        finally:
            os.unlink(path)

    def test_to_json_roundtrip(self):
        """to_json -> from_json roundtrip preserves data."""
        sol = Solution(
            status=SolverStatus.OPTIMAL,
            objective_value=42.0,
            values={"x": 1.5, "y": -3.7},
            iterations=25,
            message="converged",
            solve_time=1.23,
        )
        json_str = sol.to_json()
        restored = Solution.from_json(json_str)
        assert restored.status == sol.status
        assert restored.objective_value == sol.objective_value
        assert restored.values == sol.values
        assert restored.iterations == sol.iterations
        assert restored.message == sol.message
        assert restored.solve_time == sol.solve_time


class TestSolutionFromJson:
    """Tests for Solution.from_json()."""

    def test_from_json_string(self):
        """from_json parses JSON string."""
        json_str = '{"status": "optimal", "objective_value": 5.0, "values": {"x": 2.0}}'
        sol = Solution.from_json(json_str)
        assert sol.status == SolverStatus.OPTIMAL
        assert sol.objective_value == 5.0
        assert sol.values == {"x": 2.0}

    def test_from_json_file(self):
        """from_json reads from file path."""
        data = {
            "status": "infeasible",
            "objective_value": None,
            "values": {},
            "message": "no feasible solution",
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name

        try:
            sol = Solution.from_json(path)
            assert sol.status == SolverStatus.INFEASIBLE
            assert sol.objective_value is None
            assert sol.message == "no feasible solution"
        finally:
            os.unlink(path)

    def test_from_json_missing_optional_fields(self):
        """from_json handles missing optional fields with defaults."""
        json_str = '{"status": "failed"}'
        sol = Solution.from_json(json_str)
        assert sol.status == SolverStatus.FAILED
        assert sol.objective_value is None
        assert sol.values == {}
        assert sol.multipliers is None
        assert sol.iterations is None
        assert sol.message == ""
        assert sol.solve_time is None

    def test_from_json_all_statuses(self):
        """from_json handles all SolverStatus values."""
        for status in SolverStatus:
            json_str = json.dumps({"status": status.value})
            sol = Solution.from_json(json_str)
            assert sol.status == status

    def test_from_json_file_roundtrip(self):
        """to_json(path) -> from_json(path) roundtrip via file."""
        sol = Solution(
            status=SolverStatus.OPTIMAL,
            objective_value=100.0,
            values={"x": 10.0},
            multipliers={"c0": 1.5},
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        try:
            sol.to_json(path=path)
            restored = Solution.from_json(path)
            assert restored.status == sol.status
            assert restored.objective_value == sol.objective_value
            assert restored.values == sol.values
            assert restored.multipliers == sol.multipliers
        finally:
            os.unlink(path)

    def test_from_json_with_solve(self):
        """Roundtrip from actual solve result."""
        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.minimize(x)
        prob.subject_to(x >= 3)
        sol = prob.solve()

        json_str = sol.to_json()
        restored = Solution.from_json(json_str)
        assert restored.status == SolverStatus.OPTIMAL
        assert abs(restored.objective_value - 3.0) < 1e-6
        assert abs(restored.values["x"] - 3.0) < 1e-6


# ============================================================
# 2. Problem.reset()
# ============================================================


class TestProblemReset:
    """Tests for Problem.reset()."""

    def test_reset_clears_solver_cache(self):
        """reset() clears the solver cache."""
        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.minimize(x**2)
        prob.subject_to(x >= 1)

        # Solve to populate cache
        prob.solve()
        assert prob._solver_cache is not None

        # Reset clears it
        prob.reset()
        assert prob._solver_cache is None

    def test_reset_clears_lp_cache(self):
        """reset() clears the LP cache."""
        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.minimize(x)
        prob.subject_to(x >= 1)

        prob.solve()
        assert prob._lp_cache is not None

        prob.reset()
        assert prob._lp_cache is None

    def test_reset_clears_linearity_cache(self):
        """reset() clears the linearity check cache."""
        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.minimize(x)
        prob.subject_to(x >= 1)

        prob.solve()
        # Linearity cache should be set after solve
        assert prob._is_linear_cache is not None

        prob.reset()
        assert prob._is_linear_cache is None

    def test_reset_preserves_problem_definition(self):
        """reset() doesn't clear objective or constraints."""
        x = Variable("x", lb=0, ub=10)
        prob = Problem(name="test")
        prob.minimize(x)
        prob.subject_to(x >= 3)

        prob.solve()
        prob.reset()

        # Problem definition is preserved
        assert prob.name == "test"
        assert prob.objective is not None
        assert prob.n_constraints == 1

        # Can solve again after reset
        sol = prob.solve()
        assert sol.is_optimal
        assert abs(sol[x] - 3.0) < 1e-6

    def test_reset_forces_cold_solve(self):
        """reset() forces recompilation on next solve."""
        x = Variable("x", lb=0, ub=10)
        prob = Problem()
        prob.minimize(x)
        prob.subject_to(x >= 5)

        sol1 = prob.solve()
        prob.reset()
        sol2 = prob.solve()

        # Same result
        assert abs(sol1[x] - sol2[x]) < 1e-8

    def test_reset_nlp(self):
        """reset() works for NLP problems."""
        x = Variable("x", lb=-5, ub=5)
        prob = Problem()
        prob.minimize(x**2)

        sol1 = prob.solve()
        prob.reset()
        assert prob._solver_cache is None

        sol2 = prob.solve()
        assert abs(sol1[x] - sol2[x]) < 1e-6


# ============================================================
# 3. SolverStatus.TERMINATED
# ============================================================


class TestSolverStatusTerminated:
    """Tests for SolverStatus.TERMINATED."""

    def test_terminated_exists(self):
        """TERMINATED is a valid SolverStatus member."""
        assert hasattr(SolverStatus, "TERMINATED")
        assert SolverStatus.TERMINATED.value == "terminated"

    def test_terminated_is_feasible(self):
        """TERMINATED counts as feasible in is_feasible."""
        sol = Solution(
            status=SolverStatus.TERMINATED,
            objective_value=10.0,
            values={"x": 5.0},
        )
        assert sol.is_feasible
        assert not sol.is_optimal

    def test_terminated_serialization(self):
        """TERMINATED roundtrips through JSON."""
        sol = Solution(
            status=SolverStatus.TERMINATED,
            objective_value=10.0,
            message="stopped by callback",
        )
        json_str = sol.to_json()
        restored = Solution.from_json(json_str)
        assert restored.status == SolverStatus.TERMINATED
        assert restored.message == "stopped by callback"

    def test_all_statuses_in_enum(self):
        """All expected statuses exist in SolverStatus."""
        expected = {
            "OPTIMAL", "INFEASIBLE", "UNBOUNDED",
            "MAX_ITERATIONS", "TERMINATED", "FAILED", "NOT_SOLVED",
        }
        actual = {s.name for s in SolverStatus}
        assert expected == actual


# ============================================================
# 4. print_vars() (Roadmap 3.7 bonus)
# ============================================================


class TestPrintVars:
    """Tests for Solution.print_vars()."""

    def test_print_vars_output(self, capsys):
        """print_vars outputs variable values."""
        sol = Solution(
            status=SolverStatus.OPTIMAL,
            objective_value=42.0,
            values={"x": 1.0, "y": 2.5},
        )
        sol.print_vars()
        captured = capsys.readouterr()
        assert "Status: optimal" in captured.out
        assert "Objective: 42" in captured.out
        assert "x: 1" in captured.out
        assert "y: 2.5" in captured.out

    def test_print_vars_no_objective(self, capsys):
        """print_vars handles None objective."""
        sol = Solution(
            status=SolverStatus.FAILED,
            values={},
        )
        sol.print_vars()
        captured = capsys.readouterr()
        assert "Status: failed" in captured.out
        assert "Objective" not in captured.out

    def test_print_vars_sorted(self, capsys):
        """print_vars outputs variables in sorted order."""
        sol = Solution(
            status=SolverStatus.OPTIMAL,
            objective_value=0.0,
            values={"z": 3.0, "a": 1.0, "m": 2.0},
        )
        sol.print_vars()
        captured = capsys.readouterr()
        lines = captured.out.strip().split("\n")
        # Find variable lines (indented with "  ")
        var_lines = [l.strip() for l in lines if l.startswith("  ")]
        names = [l.split(":")[0] for l in var_lines]
        assert names == ["a", "m", "z"]
