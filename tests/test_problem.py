"""Tests for the Problem class."""

import pytest

from optyx import Variable
from optyx.problem import Problem


class TestProblemCreation:
    """Tests for creating optimization problems."""
    
    def test_empty_problem(self):
        prob = Problem()
        assert prob.objective is None
        assert prob.n_variables == 0
        assert prob.n_constraints == 0
    
    def test_problem_with_name(self):
        prob = Problem(name="test_problem")
        assert prob.name == "test_problem"
    
    def test_minimize_objective(self):
        x = Variable("x")
        prob = Problem().minimize(x**2)
        assert prob.objective is not None
        assert prob.sense == "minimize"
    
    def test_maximize_objective(self):
        x = Variable("x")
        prob = Problem().maximize(x)
        assert prob.objective is not None
        assert prob.sense == "maximize"
    
    def test_subject_to(self):
        x = Variable("x")
        prob = Problem().minimize(x**2).subject_to(x >= 0)
        assert prob.n_constraints == 1
    
    def test_multiple_constraints(self):
        x = Variable("x")
        prob = (
            Problem()
            .minimize(x**2)
            .subject_to(x >= 0)
            .subject_to(x <= 10)
        )
        assert prob.n_constraints == 2


class TestProblemVariables:
    """Tests for variable extraction from problems."""
    
    def test_variables_from_objective(self):
        x = Variable("x")
        y = Variable("y")
        prob = Problem().minimize(x**2 + y**2)
        assert set(prob.variables) == {x, y}
    
    def test_variables_from_constraints(self):
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")
        prob = Problem().minimize(x**2).subject_to(y + z <= 10)
        assert set(prob.variables) == {x, y, z}
    
    def test_variables_sorted_by_name(self):
        z = Variable("z")
        a = Variable("a")
        m = Variable("m")
        prob = Problem().minimize(z + a + m)
        assert [v.name for v in prob.variables] == ["a", "m", "z"]
    
    def test_no_duplicate_variables(self):
        x = Variable("x")
        prob = (
            Problem()
            .minimize(x**2)
            .subject_to(x >= 0)
            .subject_to(x <= 10)
        )
        assert prob.variables == [x]


class TestProblemBounds:
    """Tests for variable bounds extraction."""
    
    def test_get_bounds(self):
        x = Variable("x", lb=0, ub=10)
        y = Variable("y", lb=-5)
        prob = Problem().minimize(x + y)
        bounds = prob.get_bounds()
        assert bounds[0] == (0, 10)  # x
        assert bounds[1] == (-5, None)  # y
    
    def test_unbounded_variables(self):
        x = Variable("x")
        prob = Problem().minimize(x**2)
        bounds = prob.get_bounds()
        assert bounds[0] == (None, None)


class TestProblemChaining:
    """Tests for fluent API method chaining."""
    
    def test_full_chain(self):
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)
        
        prob = (
            Problem(name="quadratic")
            .minimize(x**2 + y**2)
            .subject_to(x + y >= 1)
            .subject_to(x <= 5)
            .subject_to(y <= 5)
        )
        
        assert prob.name == "quadratic"
        assert prob.sense == "minimize"
        assert prob.n_variables == 2
        assert prob.n_constraints == 3


class TestProblemSolve:
    """Tests for solve method prerequisites."""
    
    def test_solve_without_objective_raises(self):
        prob = Problem()
        with pytest.raises(ValueError, match="No objective"):
            prob.solve()


class TestProblemRepr:
    """Tests for problem representation."""
    
    def test_repr_empty(self):
        prob = Problem()
        repr_str = repr(prob)
        assert "Problem" in repr_str
        assert "not set" in repr_str
    
    def test_repr_with_objective(self):
        x = Variable("x")
        prob = Problem().minimize(x**2)
        repr_str = repr(prob)
        assert "minimize" in repr_str
