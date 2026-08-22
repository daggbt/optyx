"""Tests for the SciPy solver integration."""

import warnings
import pytest
import numpy as np
from scipy.optimize import Bounds, OptimizeResult

from optyx import Variable, VectorVariable
from optyx.core.errors import UnsupportedOperationError
from optyx.problem import Problem
from optyx.solution import SolverStatus
from optyx.solvers import scipy_solver


class TestIntegerBinaryWarning:
    """Tests for integer/binary variable handling.

    With MIP support, linear problems with integer vars are routed to milp().
    Nonlinear problems with integer vars raise UnsupportedOperationError.
    """

    def test_binary_variable_solves_via_milp(self):
        """Binary variables in LP are solved via milp()."""
        x = Variable("x", domain="binary")
        prob = Problem().minimize(x)
        sol = prob.solve()

        assert sol.is_optimal
        assert abs(sol["x"]) < 1e-6

    def test_integer_variable_solves_via_milp(self):
        """Integer variables in LP are solved via milp()."""
        x = Variable("x", lb=0, ub=10, domain="integer")
        prob = Problem().minimize(x)
        sol = prob.solve()

        assert sol.is_optimal
        assert abs(sol["x"]) < 1e-6

    def test_nonlinear_integer_raises(self):
        """Nonlinear + integer variables raises UnsupportedOperationError."""
        x = Variable("x", domain="binary")
        prob = Problem().minimize((x - 0.5) ** 2)

        with pytest.raises(UnsupportedOperationError, match="nonlinear"):
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

    def test_strict_mode_ok_for_milp(self):
        """strict=True still works for linear MIP (no error)."""
        x = Variable("x", domain="binary")
        prob = Problem().minimize(x)

        sol = prob.solve(strict=True)
        assert sol.is_optimal

    def test_nonlinear_integer_raises_regardless_of_strict(self):
        """Nonlinear + integer raises UnsupportedOperationError regardless of strict flag."""
        x = Variable("x", lb=0, ub=10, domain="integer")
        prob = Problem().minimize((x - 3.7) ** 2)

        with pytest.raises(UnsupportedOperationError, match="nonlinear"):
            prob.solve(strict=True)

    @pytest.mark.parametrize("method", ["SLSQP", "trust-constr", "CG", "trust-krylov"])
    def test_nonlinear_integer_named_nlp_methods_raise(self, method):
        """Named NLP methods should reject MINLP models consistently."""
        x = Variable("x", lb=0, ub=10, domain="integer")
        prob = Problem().minimize((x - 3.7) ** 2)

        with pytest.raises(UnsupportedOperationError, match="nonlinear"):
            prob.solve(method=method)

    def test_strict_mode_ok_for_continuous(self):
        """strict=True should not raise for continuous variables."""
        x = Variable("x")
        prob = Problem().minimize(x**2)

        # Should not raise
        sol = prob.solve(strict=True)
        assert sol.is_optimal

    def test_milp_default_mode_solves(self):
        """Default mode (strict=False) solves MILP correctly."""
        x = Variable("x", domain="binary")
        prob = Problem().minimize(x)

        sol = prob.solve(strict=False)
        assert sol.is_optimal

    def test_milp_linear_with_multiple_integer_vars(self):
        """Multiple integer/binary vars in LP all route to milp()."""
        a = Variable("a", domain="binary")
        b = Variable("b", domain="integer", lb=0, ub=5)
        prob = Problem().minimize(a + b)

        sol = prob.solve()
        assert sol.is_optimal
        assert abs(sol["a"]) < 1e-6
        assert abs(sol["b"]) < 1e-6


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

    def test_lbfgsb_skips_all_infinite_bounds(self, monkeypatch):
        """Unbounded L-BFGS-B problems should not pass no-op bounds to SciPy."""
        captured: dict[str, object] = {}

        def fake_minimize(*args, **kwargs):
            captured["bounds"] = kwargs.get("bounds")
            x0 = np.asarray(kwargs["x0"], dtype=float)
            return OptimizeResult(
                x=x0,
                fun=float(kwargs["fun"](x0)),
                success=True,
                message="ok",
                nit=0,
            )

        monkeypatch.setattr(scipy_solver, "minimize", fake_minimize)

        x = Variable("x")
        prob = Problem().minimize(x**2)
        sol = prob.solve(method="L-BFGS-B", x0=np.array([0.0]))

        assert sol.is_optimal
        assert captured["bounds"] is None

    def test_lbfgsb_preserves_finite_bounds(self, monkeypatch):
        """Finite bounds should still be forwarded to SciPy."""
        captured: dict[str, object] = {}

        def fake_minimize(*args, **kwargs):
            captured["bounds"] = kwargs.get("bounds")
            x0 = np.asarray(kwargs["x0"], dtype=float)
            return OptimizeResult(
                x=x0,
                fun=float(kwargs["fun"](x0)),
                success=True,
                message="ok",
                nit=0,
            )

        monkeypatch.setattr(scipy_solver, "minimize", fake_minimize)

        x = Variable("x", lb=1.0, ub=2.0)
        prob = Problem().minimize(x**2)
        sol = prob.solve(method="L-BFGS-B", x0=np.array([1.5]))

        assert sol.is_optimal
        assert isinstance(captured["bounds"], Bounds)

    def test_lbfgsb_skips_unbounded_single_vector_bounds(self, monkeypatch):
        """Unbounded single-vector problems should bypass bounds scanning."""
        captured: dict[str, object] = {}

        def fake_minimize(*args, **kwargs):
            captured["bounds"] = kwargs.get("bounds")
            x0 = np.asarray(kwargs["x0"], dtype=float)
            return OptimizeResult(
                x=x0,
                fun=float(kwargs["fun"](x0)),
                success=True,
                message="ok",
                nit=0,
            )

        def fail_build_bounds(_variables):
            raise AssertionError(
                "_build_bounds should not run for unbounded vector problems"
            )

        monkeypatch.setattr(scipy_solver, "minimize", fake_minimize)
        monkeypatch.setattr(scipy_solver, "_build_bounds", fail_build_bounds)

        x = VectorVariable("x", 3)
        prob = Problem().minimize(x.dot(x) - x.sum())
        sol = prob.solve(method="L-BFGS-B", x0=np.zeros(3))

        assert sol.is_optimal
        assert captured["bounds"] is None

    def test_lbfgsb_vector_bound_override_preserves_bounds(self, monkeypatch):
        """Per-element bound overrides must still flow through to SciPy."""
        captured: dict[str, object] = {}

        def fake_minimize(*args, **kwargs):
            captured["bounds"] = kwargs.get("bounds")
            x0 = np.asarray(kwargs["x0"], dtype=float)
            bounds = kwargs["bounds"]
            result_x = np.maximum(x0, bounds.lb)
            return OptimizeResult(
                x=result_x,
                fun=float(kwargs["fun"](result_x)),
                success=True,
                message="ok",
                nit=0,
            )

        monkeypatch.setattr(scipy_solver, "minimize", fake_minimize)

        x = VectorVariable("x", 3)
        x[1].lb = 2.0
        prob = Problem().minimize(x.dot(x) - x.sum())
        sol = prob.solve(method="L-BFGS-B", x0=np.zeros(3))

        assert sol.is_optimal
        assert isinstance(captured["bounds"], Bounds)
        bounds = captured["bounds"]
        assert isinstance(bounds, Bounds)
        assert bounds.lb[1] == pytest.approx(2.0)


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


class TestPostSolveFeasibilityValidation:
    """Candidate feasibility must be independent of SciPy's success flag."""

    def test_infeasible_positive_directional_result_is_not_optimal(self):
        x = Variable("x")
        prob = Problem().minimize(x**2).subject_to([x >= 1, x <= 0])

        with pytest.warns(UserWarning, match="SLSQP returned a solution"):
            sol = prob.solve(method="SLSQP")

        assert not sol.is_optimal
        assert sol.status in (SolverStatus.FAILED, SolverStatus.INFEASIBLE)
        assert "Maximum constraint or bound violation" in sol.message

    def test_feasible_positive_directional_result_remains_compatible(self, monkeypatch):
        def fake_minimize(**kwargs):
            x = np.array([1.0])
            return OptimizeResult(
                x=x,
                fun=float(kwargs["fun"](x)),
                success=False,
                message="Positive directional derivative for linesearch",
                nit=1,
            )

        monkeypatch.setattr(scipy_solver, "minimize", fake_minimize)
        x = Variable("x")
        prob = Problem().minimize((x - 1) ** 2).subject_to(x >= 0)

        sol = prob.solve(method="SLSQP")

        assert sol.status == SolverStatus.OPTIMAL

    def test_successful_result_that_violates_variable_bound_is_not_optimal(
        self, monkeypatch
    ):
        def fake_minimize(**kwargs):
            x = np.array([-1.0])
            return OptimizeResult(
                x=x,
                fun=float(kwargs["fun"](x)),
                success=True,
                message="Optimization terminated successfully",
                nit=1,
            )

        monkeypatch.setattr(scipy_solver, "minimize", fake_minimize)
        x = Variable("x", lb=0)
        prob = Problem().minimize(x**2)

        sol = prob.solve(method="L-BFGS-B")

        assert sol.status == SolverStatus.FAILED
        assert "Maximum constraint or bound violation: 1.00e+00" in sol.message

    def test_matrix_constraint_violation_is_not_optimal(self, monkeypatch):
        def fake_minimize(**kwargs):
            x = np.zeros(2)
            return OptimizeResult(
                x=x,
                fun=float(kwargs["fun"](x)),
                success=True,
                message="Optimization terminated successfully",
                nit=1,
            )

        monkeypatch.setattr(scipy_solver, "minimize", fake_minimize)
        x = VectorVariable("x", 2)
        prob = Problem().minimize(x.dot(x))
        prob.subject_to(np.eye(2) @ x >= np.ones(2))

        sol = prob.solve(method="trust-constr")

        assert sol.status == SolverStatus.FAILED
        assert "Maximum constraint or bound violation: 1.00e+00" in sol.message

    def test_non_finite_candidate_is_not_optimal(self, monkeypatch):
        def fake_minimize(**kwargs):
            return OptimizeResult(
                x=np.array([np.nan]),
                fun=np.nan,
                success=True,
                message="Optimization terminated successfully",
                nit=1,
            )

        monkeypatch.setattr(scipy_solver, "minimize", fake_minimize)
        x = Variable("x")
        prob = Problem().minimize(x**2)

        sol = prob.solve(method="BFGS")

        assert sol.status == SolverStatus.FAILED
        assert "Candidate contains non-finite" in sol.message

    @pytest.mark.parametrize(
        ("candidate", "expected_feasible"),
        [(1.0, True), (-1.0, False)],
    )
    def test_max_iterations_uses_candidate_feasibility(
        self, monkeypatch, candidate, expected_feasible
    ):
        def fake_minimize(**kwargs):
            x = np.array([candidate])
            return OptimizeResult(
                x=x,
                fun=float(kwargs["fun"](x)),
                success=False,
                message="Maximum number of iterations reached",
                nit=1,
            )

        monkeypatch.setattr(scipy_solver, "minimize", fake_minimize)
        x = Variable("x")
        prob = Problem().minimize(x**2).subject_to(x >= 0)

        sol = prob.solve(method="trust-constr")

        assert sol.status == SolverStatus.MAX_ITERATIONS
        assert sol.feasibility_checked
        assert sol.is_feasible is expected_feasible


class TestSLSQPStationarityValidation:
    """SLSQP candidates must satisfy a first-order necessary condition."""

    def test_feasible_nonstationary_candidate_retries_with_trust_constr(self):
        x = Variable("x")
        y = Variable("y")
        prob = Problem().minimize((x - 1) ** 2).subject_to(y >= 2)

        with pytest.warns(UserWarning, match="feasible but non-stationary"):
            sol = prob.solve(method="SLSQP", warm_start=False)

        assert sol.status == SolverStatus.OPTIMAL
        assert sol.objective_value == pytest.approx(0.0, abs=1e-8)
        assert sol["x"] == pytest.approx(1.0, abs=1e-5)
        assert sol["y"] >= 2.0 - 1e-5

    @pytest.mark.parametrize("active_kind", ["constraint", "bound"])
    def test_valid_active_optimum_does_not_trigger_fallback(
        self, monkeypatch, active_kind
    ):
        original_minimize = scipy_solver.minimize
        methods: list[str] = []

        def tracking_minimize(*args, **kwargs):
            methods.append(kwargs["method"])
            return original_minimize(*args, **kwargs)

        monkeypatch.setattr(scipy_solver, "minimize", tracking_minimize)
        if active_kind == "constraint":
            x = Variable("x")
            prob = Problem().minimize(x**2).subject_to(x >= 1)
        else:
            x = Variable("x", lb=1)
            prob = Problem().minimize(x)

        sol = prob.solve(method="SLSQP", warm_start=False)

        assert sol.status == SolverStatus.OPTIMAL
        assert sol["x"] == pytest.approx(1.0, abs=1e-5)
        assert methods == ["SLSQP"]

    def test_valid_matrix_active_optimum_does_not_trigger_fallback(self, monkeypatch):
        original_minimize = scipy_solver.minimize
        methods: list[str] = []

        def tracking_minimize(*args, **kwargs):
            methods.append(kwargs["method"])
            return original_minimize(*args, **kwargs)

        monkeypatch.setattr(scipy_solver, "minimize", tracking_minimize)
        x = VectorVariable("x", 2)
        prob = Problem().minimize(x.dot(x))
        prob.subject_to(np.array([[1.0, 1.0]]) @ x >= np.array([2.0]))

        sol = prob.solve(method="SLSQP", warm_start=False)

        assert sol.status == SolverStatus.OPTIMAL
        assert sol["x[0]"] == pytest.approx(1.0, abs=1e-5)
        assert sol["x[1]"] == pytest.approx(1.0, abs=1e-5)
        assert methods == ["SLSQP"]


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
        """Adding a constraint should selectively invalidate constraint cache."""
        x = Variable("x", lb=0)
        y = Variable("y", lb=0)

        prob = Problem().minimize(x**2 + y**2)
        prob.solve()

        cache_after_first_solve = prob._solver_cache
        assert cache_after_first_solve is not None

        # Add a constraint
        prob.subject_to(x + y >= 1)

        # Objective cache should be preserved (selective invalidation)
        assert prob._solver_cache is not None
        assert "obj_fn" in prob._solver_cache
        # Constraint cache should be cleared
        assert "scipy_constraints" not in prob._solver_cache

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
