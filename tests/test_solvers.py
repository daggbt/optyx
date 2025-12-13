"""Tests for the SciPy solver integration."""

import warnings
import pytest
import numpy as np

from optyx import Variable
from optyx.problem import Problem
from optyx.solution import SolverStatus


class TestIntegerBinaryWarning:
    """Tests for warnings when using integer/binary variables with SciPy."""
    
    def test_binary_variable_emits_warning(self):
        """Binary variables should emit a warning about relaxation."""
        x = Variable("x", domain="binary")
        prob = Problem().minimize((x - 0.5)**2)
        
        with pytest.warns(UserWarning, match="integer/binary domains"):
            sol = prob.solve()
        
        assert sol.is_optimal
        # Solution is relaxed to continuous [0, 1]
        assert 0 <= sol["x"] <= 1
    
    def test_integer_variable_emits_warning(self):
        """Integer variables should emit a warning about relaxation."""
        x = Variable("x", lb=0, ub=10, domain="integer")
        prob = Problem().minimize((x - 3.7)**2)
        
        with pytest.warns(UserWarning, match="integer/binary domains"):
            sol = prob.solve()
        
        assert sol.is_optimal
        # Solution is relaxed, not rounded to integer
        assert abs(sol["x"] - 3.7) < 1e-4
    
    def test_warning_lists_variable_names(self):
        """Warning should list all affected variable names."""
        a = Variable("a", domain="binary")
        b = Variable("b", domain="integer", lb=0, ub=5)
        c = Variable("c")  # continuous, no warning
        prob = Problem().minimize(a + b + c**2)
        
        with pytest.warns(UserWarning, match=r"\[a, b\]"):
            prob.solve()
    
    def test_continuous_no_warning(self):
        """Continuous variables should not emit a warning."""
        x = Variable("x")
        prob = Problem().minimize(x**2)
        
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # Fail if any warning is raised
            sol = prob.solve()
        
        assert sol.is_optimal


class TestUnconstrainedOptimization:
    """Tests for unconstrained optimization problems."""
    
    def test_simple_quadratic(self):
        """min x^2 → x* = 0"""
        x = Variable("x")
        prob = Problem().minimize(x**2)
        sol = prob.solve()
        
        assert sol.is_optimal
        assert abs(sol["x"]) < 1e-5
        assert sol.objective_value < 1e-10
    
    def test_two_variable_quadratic(self):
        """min x^2 + y^2 → (x*, y*) = (0, 0)"""
        x = Variable("x")
        y = Variable("y")
        prob = Problem().minimize(x**2 + y**2)
        sol = prob.solve()
        
        assert sol.is_optimal
        assert abs(sol["x"]) < 1e-5
        assert abs(sol["y"]) < 1e-5
    
    def test_rosenbrock(self):
        """min (1-x)^2 + 100(y-x^2)^2 → (x*, y*) = (1, 1)"""
        x = Variable("x")
        y = Variable("y")
        rosenbrock = (1 - x)**2 + 100*(y - x**2)**2
        prob = Problem().minimize(rosenbrock)
        sol = prob.solve()
        
        assert sol.is_optimal
        assert abs(sol["x"] - 1.0) < 1e-3
        assert abs(sol["y"] - 1.0) < 1e-3


class TestBoundedOptimization:
    """Tests for optimization with variable bounds."""
    
    def test_lower_bound_active(self):
        """min x s.t. x >= 5 → x* = 5"""
        x = Variable("x", lb=5)
        prob = Problem().minimize(x)
        sol = prob.solve()
        
        assert sol.is_optimal
        assert abs(sol["x"] - 5.0) < 1e-5
    
    def test_upper_bound_active(self):
        """max x s.t. x <= 10 → x* = 10"""
        x = Variable("x", ub=10)
        prob = Problem().maximize(x)
        sol = prob.solve()
        
        assert sol.is_optimal
        assert abs(sol["x"] - 10.0) < 1e-5
    
    def test_box_constrained(self):
        """min (x-3)^2 s.t. 0 <= x <= 2 → x* = 2"""
        x = Variable("x", lb=0, ub=2)
        prob = Problem().minimize((x - 3)**2)
        sol = prob.solve()
        
        assert sol.is_optimal
        assert abs(sol["x"] - 2.0) < 1e-5


class TestConstrainedOptimization:
    """Tests for optimization with general constraints."""
    
    def test_inequality_constraint(self):
        """min x^2 + y^2 s.t. x + y >= 1 → (x*, y*) = (0.5, 0.5)"""
        x = Variable("x")
        y = Variable("y")
        prob = (
            Problem()
            .minimize(x**2 + y**2)
            .subject_to(x + y >= 1)
        )
        sol = prob.solve()
        
        assert sol.is_optimal
        assert abs(sol["x"] - 0.5) < 1e-4
        assert abs(sol["y"] - 0.5) < 1e-4
        assert abs(sol.objective_value - 0.5) < 1e-4
    
    def test_equality_constraint(self):
        """min x^2 + y^2 s.t. x + y == 2 → (x*, y*) = (1, 1)"""
        x = Variable("x")
        y = Variable("y")
        prob = (
            Problem()
            .minimize(x**2 + y**2)
            .subject_to((x + y).constraint_eq(2))
        )
        sol = prob.solve()
        
        assert sol.is_optimal
        assert abs(sol["x"] - 1.0) < 1e-4
        assert abs(sol["y"] - 1.0) < 1e-4
    
    def test_le_constraint(self):
        """min -x s.t. x <= 5 → x* = 5"""
        x = Variable("x")
        prob = (
            Problem()
            .minimize(-x)
            .subject_to(x <= 5)
        )
        sol = prob.solve()
        
        assert sol.is_optimal
        assert abs(sol["x"] - 5.0) < 1e-4
    
    def test_multiple_constraints(self):
        """min -x - y s.t. x + y <= 4, x <= 2, y <= 3 → (x*, y*) = (2, 2)"""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)
        prob = (
            Problem()
            .minimize(-x - y)
            .subject_to(x + y <= 4)
            .subject_to(x <= 2)
            .subject_to(y <= 3)
        )
        sol = prob.solve()
        
        assert sol.is_optimal
        assert abs(sol["x"] - 2.0) < 1e-4
        assert abs(sol["y"] - 2.0) < 1e-4


class TestMaximization:
    """Tests for maximization problems."""
    
    def test_simple_maximize(self):
        """max -x^2 → x* = 0"""
        x = Variable("x")
        prob = Problem().maximize(-x**2)
        sol = prob.solve()
        
        assert sol.is_optimal
        assert abs(sol["x"]) < 1e-5
        assert abs(sol.objective_value) < 1e-10
    
    def test_maximize_with_bounds(self):
        """max x s.t. 0 <= x <= 5 → x* = 5"""
        x = Variable("x", lb=0, ub=5)
        prob = Problem().maximize(x)
        sol = prob.solve()
        
        assert sol.is_optimal
        assert abs(sol["x"] - 5.0) < 1e-5
        assert abs(sol.objective_value - 5.0) < 1e-5


class TestSolutionObject:
    """Tests for Solution object properties."""
    
    def test_solution_values_dict(self):
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)
        prob = Problem().minimize(x**2 + y**2).subject_to(x + y >= 1)
        sol = prob.solve()
        
        assert "x" in sol.values
        assert "y" in sol.values
    
    def test_solution_getitem(self):
        x = Variable("x")
        prob = Problem().minimize(x**2)
        sol = prob.solve()
        
        # Access by variable object
        assert abs(sol[x]) < 1e-5
        # Access by name
        assert abs(sol["x"]) < 1e-5
    
    def test_solution_iterations(self):
        x = Variable("x")
        prob = Problem().minimize(x**2)
        sol = prob.solve()
        
        assert sol.iterations is not None
        assert sol.iterations >= 0
    
    def test_solution_solve_time(self):
        x = Variable("x")
        prob = Problem().minimize(x**2)
        sol = prob.solve()
        
        assert sol.solve_time is not None
        assert sol.solve_time >= 0


class TestSolverMethods:
    """Tests for different solver methods."""
    
    def test_slsqp(self):
        x = Variable("x", lb=0)
        prob = Problem().minimize(x**2).subject_to(x >= 1)
        sol = prob.solve(method="SLSQP")
        
        assert sol.is_optimal
        assert abs(sol["x"] - 1.0) < 1e-4
    
    @pytest.mark.filterwarnings("ignore:delta_grad == 0.0:UserWarning")
    def test_trust_constr(self):
        x = Variable("x", lb=0)
        prob = Problem().minimize(x**2).subject_to(x >= 1)
        sol = prob.solve(method="trust-constr")
        
        assert sol.is_optimal
        assert abs(sol["x"] - 1.0) < 1e-3  # trust-constr has looser tolerance
    
    def test_lbfgsb_bounds_only(self):
        """L-BFGS-B only supports bounds, not general constraints."""
        x = Variable("x", lb=1, ub=10)
        prob = Problem().minimize(x**2)
        sol = prob.solve(method="L-BFGS-B")
        
        assert sol.is_optimal
        assert abs(sol["x"] - 1.0) < 1e-4


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_single_variable(self):
        x = Variable("x")
        prob = Problem().minimize((x - 3)**2)
        sol = prob.solve()
        
        assert sol.is_optimal
        assert abs(sol["x"] - 3.0) < 1e-4
    
    def test_custom_initial_point(self):
        x = Variable("x")
        prob = Problem().minimize(x**2)
        sol = prob.solve(x0=np.array([5.0]))
        
        assert sol.is_optimal
        assert abs(sol["x"]) < 1e-5
    
    def test_repr(self):
        x = Variable("x")
        prob = Problem().minimize(x**2)
        sol = prob.solve()
        
        repr_str = repr(sol)
        assert "Solution" in repr_str
        assert "optimal" in repr_str
