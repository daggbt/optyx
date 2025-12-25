"""Tests for the SciPy solver integration."""

import warnings
import pytest
import numpy as np

from optyx import Variable
from optyx.problem import Problem


class TestIntegerBinaryWarning:
    """Tests for warnings when using integer/binary variables with SciPy."""

    def test_binary_variable_emits_warning(self):
        """Binary variables should emit a warning about relaxation."""
        x = Variable("x", domain="binary")
        prob = Problem().minimize((x - 0.5) ** 2)

        with pytest.warns(UserWarning, match="integer/binary domains"):
            sol = prob.solve()

        assert sol.is_optimal
        # Solution is relaxed to continuous [0, 1]
        assert 0 <= sol["x"] <= 1

    def test_integer_variable_emits_warning(self):
        """Integer variables should emit a warning about relaxation."""
        x = Variable("x", lb=0, ub=10, domain="integer")
        prob = Problem().minimize((x - 3.7) ** 2)

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


class TestStrictMode:
    """Tests for strict mode enforcement of integer/binary variables."""

    def test_strict_mode_raises_for_binary(self):
        """strict=True should raise ValueError for binary variables."""
        x = Variable("x", domain="binary")
        prob = Problem().minimize((x - 0.5) ** 2)

        with pytest.raises(ValueError, match="integer/binary domains"):
            prob.solve(strict=True)

    def test_strict_mode_raises_for_integer(self):
        """strict=True should raise ValueError for integer variables."""
        x = Variable("x", lb=0, ub=10, domain="integer")
        prob = Problem().minimize((x - 3.7) ** 2)

        with pytest.raises(ValueError, match="integer/binary domains"):
            prob.solve(strict=True)

    def test_strict_mode_ok_for_continuous(self):
        """strict=True should not raise for continuous variables."""
        x = Variable("x")
        prob = Problem().minimize(x**2)

        # Should not raise
        sol = prob.solve(strict=True)
        assert sol.is_optimal

    def test_strict_false_still_warns(self):
        """strict=False (default) should still emit warning."""
        x = Variable("x", domain="binary")
        prob = Problem().minimize((x - 0.5) ** 2)

        with pytest.warns(UserWarning, match="integer/binary domains"):
            sol = prob.solve(strict=False)

        assert sol.is_optimal

    def test_error_message_includes_variable_names(self):
        """Error message should list affected variable names."""
        a = Variable("a", domain="binary")
        b = Variable("b", domain="integer", lb=0, ub=5)
        prob = Problem().minimize(a + b)

        with pytest.raises(ValueError, match=r"\[a, b\]"):
            prob.solve(strict=True)


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
        rosenbrock = (1 - x) ** 2 + 100 * (y - x**2) ** 2
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
        prob = Problem().minimize((x - 3) ** 2)
        sol = prob.solve()

        assert sol.is_optimal
        assert abs(sol["x"] - 2.0) < 1e-5


class TestConstrainedOptimization:
    """Tests for optimization with general constraints."""

    def test_inequality_constraint(self):
        """min x^2 + y^2 s.t. x + y >= 1 → (x*, y*) = (0.5, 0.5)"""
        x = Variable("x")
        y = Variable("y")
        prob = Problem().minimize(x**2 + y**2).subject_to(x + y >= 1)
        sol = prob.solve()

        assert sol.is_optimal
        assert abs(sol["x"] - 0.5) < 1e-4
        assert abs(sol["y"] - 0.5) < 1e-4
        assert abs(sol.objective_value - 0.5) < 1e-4

    def test_equality_constraint(self):
        """min x^2 + y^2 s.t. x + y == 2 → (x*, y*) = (1, 1)"""
        x = Variable("x")
        y = Variable("y")
        prob = Problem().minimize(x**2 + y**2).subject_to((x + y).eq(2))
        sol = prob.solve()

        assert sol.is_optimal
        assert abs(sol["x"] - 1.0) < 1e-4
        assert abs(sol["y"] - 1.0) < 1e-4

    def test_le_constraint(self):
        """min -x s.t. x <= 5 → x* = 5"""
        x = Variable("x")
        prob = Problem().minimize(-x).subject_to(x <= 5)
        sol = prob.solve()

        assert sol.is_optimal
        assert abs(sol["x"] - 5.0) < 1e-4

    def test_multiple_constraints(self):
        """min -x - y s.t. x + y <= 4, x <= 2, y <= 3 → objective = -4

        Multiple optimal solutions exist: (2, 2) and (1, 3) are both optimal.
        We test the objective value instead of specific variable values.
        """
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
        assert abs(sol.objective_value - (-4.0)) < 1e-4
        # Verify feasibility of solution
        assert sol["x"] + sol["y"] <= 4.0 + 1e-6
        assert sol["x"] <= 2.0 + 1e-6
        assert sol["y"] <= 3.0 + 1e-6


class TestMaximization:
    """Tests for maximization problems."""

    def test_simple_maximize(self):
        """max -x^2 → x* = 0"""
        x = Variable("x")
        prob = Problem().maximize(-(x**2))
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
        prob = Problem().minimize((x - 3) ** 2)
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


class TestHessianIntegration:
    """Tests for Hessian support with trust-region methods."""

    def test_trust_constr_with_hessian(self):
        """trust-constr method should use symbolic Hessian by default."""
        x = Variable("x")
        y = Variable("y")
        # Rosenbrock function - benefits from Hessian
        rosenbrock = (1 - x) ** 2 + 100 * (y - x**2) ** 2
        prob = Problem().minimize(rosenbrock)

        sol = prob.solve(method="trust-constr")

        assert sol.is_optimal
        assert abs(sol["x"] - 1.0) < 1e-3
        assert abs(sol["y"] - 1.0) < 1e-3

    def test_trust_constr_without_hessian(self):
        """trust-constr with use_hessian=False should still work."""
        x = Variable("x")
        y = Variable("y")
        rosenbrock = (1 - x) ** 2 + 100 * (y - x**2) ** 2
        prob = Problem().minimize(rosenbrock)

        sol = prob.solve(method="trust-constr", use_hessian=False)

        assert sol.is_optimal
        assert abs(sol["x"] - 1.0) < 1e-3
        assert abs(sol["y"] - 1.0) < 1e-3

    def test_newton_cg_with_hessian(self):
        """Newton-CG method should use symbolic Hessian."""
        x = Variable("x")
        y = Variable("y")
        # Simple quadratic - easy for Newton-CG
        quadratic = x**2 + y**2
        prob = Problem().minimize(quadratic)

        sol = prob.solve(method="Newton-CG")

        assert sol.is_optimal
        assert abs(sol["x"]) < 1e-5
        assert abs(sol["y"]) < 1e-5

    def test_slsqp_ignores_hessian(self):
        """SLSQP doesn't use Hessian, should still work."""
        x = Variable("x")
        prob = Problem().minimize((x - 3) ** 2)

        sol = prob.solve(method="SLSQP")

        assert sol.is_optimal
        assert abs(sol["x"] - 3.0) < 1e-5

    def test_hessian_with_constraints(self):
        """trust-constr with Hessian and constraints."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem().minimize(x**2 + y**2).subject_to(x + y >= 1)

        sol = prob.solve(method="trust-constr")

        assert sol.is_optimal
        assert abs(sol["x"] - 0.5) < 1e-3
        assert abs(sol["y"] - 0.5) < 1e-3


class TestSolverCaching:
    """Tests for solver cache behavior."""

    def test_cache_reused_on_repeated_solve(self):
        """Multiple solve() calls should reuse cached callables."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem().minimize(x**2 + y**2).subject_to(x + y >= 1)

        # First solve - builds cache
        sol1 = prob.solve()
        assert prob._solver_cache is not None
        cache1 = prob._solver_cache

        # Second solve - reuses cache
        sol2 = prob.solve()
        assert prob._solver_cache is cache1  # Same cache object

        # Results should be the same
        assert abs(sol1["x"] - sol2["x"]) < 1e-10
        assert abs(sol1["y"] - sol2["y"]) < 1e-10

    def test_cache_invalidated_on_constraint_add(self):
        """Adding a constraint should invalidate the cache."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem().minimize(x**2 + y**2)
        prob.solve()

        cache_after_first_solve = prob._solver_cache
        assert cache_after_first_solve is not None

        # Add a constraint
        prob.subject_to(x + y >= 1)

        # Cache should be invalidated
        assert prob._solver_cache is None

    def test_cache_invalidated_on_objective_change(self):
        """Changing objective should invalidate the cache."""
        x = Variable("x")

        prob = Problem().minimize(x**2)
        prob.solve()

        assert prob._solver_cache is not None

        # Change objective
        prob.maximize(x)

        # Cache should be invalidated
        assert prob._solver_cache is None

    def test_solve_with_different_x0_uses_cache(self):
        """Different initial points should still use cached callables."""
        import numpy as np

        x = Variable("x", lb=-10, ub=10)

        prob = Problem().minimize((x - 5) ** 2)

        # Solve with different initial points
        sol1 = prob.solve(x0=np.array([0.0]))
        cache1 = prob._solver_cache

        sol2 = prob.solve(x0=np.array([10.0]))
        cache2 = prob._solver_cache

        # Cache should be reused
        assert cache1 is cache2

        # Both should find optimal
        assert abs(sol1["x"] - 5.0) < 1e-4
        assert abs(sol2["x"] - 5.0) < 1e-4
