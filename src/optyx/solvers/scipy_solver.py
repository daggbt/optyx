"""SciPy-based optimization solver.

Maps Optyx problems to scipy.optimize.minimize for solving.
"""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
from scipy.optimize import minimize

if TYPE_CHECKING:
    from optyx.problem import Problem
    from optyx.solution import Solution


def solve_scipy(
    problem: Problem,
    method: str = "SLSQP",
    x0: np.ndarray | None = None,
    tol: float | None = None,
    maxiter: int | None = None,
    use_hessian: bool = True,
    strict: bool = False,
    **kwargs: Any,
) -> Solution:
    """Solve an optimization problem using SciPy.

    Args:
        problem: The optimization problem to solve.
        method: SciPy optimization method. Options:
            - "SLSQP": Sequential Least Squares Programming (default)
            - "trust-constr": Trust-region constrained optimization
            - "L-BFGS-B": Limited-memory BFGS with bounds (no constraints)
        x0: Initial point. If None, uses midpoint of bounds or zeros.
        tol: Solver tolerance.
        maxiter: Maximum number of iterations.
        use_hessian: Whether to compute and pass the symbolic Hessian to methods
            that support it (trust-constr, Newton-CG, etc.). Default True.
            Set to False if Hessian computation is too expensive.
        strict: If True, raise ValueError when the problem contains integer/binary
            variables that cannot be enforced by the solver. If False (default),
            emit a warning and relax to continuous.
        **kwargs: Additional arguments passed to scipy.optimize.minimize.

    Returns:
        Solution object with optimization results.

    Raises:
        ValueError: If strict=True and problem contains integer/binary variables.
    """
    from optyx.core.autodiff import compile_hessian
    from optyx.solution import Solution, SolverStatus

    # Methods that support Hessian
    HESSIAN_METHODS = {
        "trust-constr",
        "Newton-CG",
        "dogleg",
        "trust-ncg",
        "trust-exact",
    }

    # Derivative-free methods that don't use gradient information
    DERIVATIVE_FREE_METHODS = {
        "Nelder-Mead",
        "Powell",
        "COBYLA",
    }

    # Methods that support bounds
    BOUNDS_METHODS = {
        "L-BFGS-B",
        "TNC",
        "SLSQP",
        "Powell",
        "trust-constr",
        "Nelder-Mead",
    }

    variables = problem.variables
    n = len(variables)

    if n == 0:
        return Solution(
            status=SolverStatus.FAILED,
            message="Problem has no variables",
        )

    # Check for non-continuous domains
    non_continuous = [v for v in variables if v.domain != "continuous"]
    if non_continuous:
        names = ", ".join(v.name for v in non_continuous)
        if strict:
            raise ValueError(
                f"Variables [{names}] have integer/binary domains but the SciPy "
                f"solver does not support integer programming. Use strict=False "
                f"to relax to continuous, or use a MIP solver like PuLP or Pyomo."
            )
        else:
            warnings.warn(
                f"Variables [{names}] have integer/binary domains but will be relaxed "
                f"to continuous. SciPy solver does not support integer programming. "
                f"For true MIP, consider PuLP or Pyomo.",
                UserWarning,
                stacklevel=3,
            )

    # Check for cached compiled callables
    cache = problem._solver_cache
    if cache is None:
        cache = _build_solver_cache(problem, variables)
        problem._solver_cache = cache

    # Extract cached callables
    obj_fn = cache["obj_fn"]
    grad_fn = cache["grad_fn"]
    scipy_constraints = cache["scipy_constraints"]
    bounds = cache["bounds"]

    # Track if we've warned about inf/nan to avoid spamming
    _inf_nan_warned = [False]  # Use list to allow mutation in nested function

    def objective(x: np.ndarray) -> float:
        val = float(obj_fn(x))
        if not np.isfinite(val) and not _inf_nan_warned[0]:
            warnings.warn(
                f"Objective function returned {val} at point {x}. "
                "This may indicate evaluation at a singularity (e.g., log(0), 1/0). "
                "Consider adjusting variable bounds or initial point.",
                RuntimeWarning,
                stacklevel=2,
            )
            _inf_nan_warned[0] = True
        return val

    def gradient(x: np.ndarray) -> np.ndarray:
        grad = grad_fn(x).flatten()
        if not np.all(np.isfinite(grad)) and not _inf_nan_warned[0]:
            warnings.warn(
                f"Gradient contains inf/nan values at point {x}. "
                "This may indicate evaluation near a singularity. "
                "Consider adjusting variable bounds or initial point.",
                RuntimeWarning,
                stacklevel=2,
            )
            _inf_nan_warned[0] = True
        return grad

    # Build Hessian for methods that support it (not cached - method-dependent)
    hess_fn: Callable[[np.ndarray], np.ndarray] | None = None
    if use_hessian and method in HESSIAN_METHODS:
        # Check if Hessian is cached for this method
        if "hess_fn" not in cache:
            obj_expr = problem.objective
            assert obj_expr is not None, "Objective must be set before solving"
            if problem.sense == "maximize":
                obj_expr = -obj_expr  # type: ignore[operator]
            compiled_hess = compile_hessian(obj_expr, variables)
            cache["hess_fn"] = compiled_hess

        compiled_hess = cache["hess_fn"]

        def _hess_fn(x: np.ndarray) -> np.ndarray:
            return compiled_hess(x)

        hess_fn = _hess_fn

    # Initial point
    if x0 is None:
        x0 = _compute_initial_point(variables)

    # Solver options
    options: dict[str, Any] = {}
    if maxiter is not None:
        options["maxiter"] = maxiter

    # Solve
    start_time = time.perf_counter()

    # Track if we see the linear problem warning
    linear_problem_detected = False

    def warning_handler(message, category, filename, lineno, file=None, line=None):
        nonlocal linear_problem_detected
        if "delta_grad == 0.0" in str(message):
            linear_problem_detected = True
            return  # Suppress this specific warning
        # Let other warnings through using the original handler
        old_showwarning(message, category, filename, lineno, file, line)

    old_showwarning = warnings.showwarning

    # Determine if gradient should be passed (not for derivative-free methods)
    use_gradient = method not in DERIVATIVE_FREE_METHODS

    try:
        # Temporarily override warning handling during solve
        warnings.showwarning = warning_handler

        result = minimize(
            fun=objective,
            x0=x0,
            method=method,
            jac=gradient if use_gradient else None,
            hess=hess_fn if hess_fn is not None else None,
            bounds=bounds if bounds and method in BOUNDS_METHODS else None,
            constraints=scipy_constraints if scipy_constraints else (),
            tol=tol,
            options=options if options else None,
            **kwargs,
        )
    except Exception as e:
        warnings.showwarning = old_showwarning
        return Solution(
            status=SolverStatus.FAILED,
            message=str(e),
            solve_time=time.perf_counter() - start_time,
        )
    finally:
        warnings.showwarning = old_showwarning

    solve_time = time.perf_counter() - start_time

    # Check if constraints are satisfied (SLSQP can return "optimal" with violated constraints)
    # Use scaled tolerance: atol + rtol * max(1, |constraint_value|)
    atol = tol if tol is not None else 1e-6
    rtol = 1e-6
    constraints_violated = False
    max_violation = 0.0

    if result.success and scipy_constraints:
        for c in scipy_constraints:
            c_val = c["fun"](result.x)
            # Scaled tolerance based on constraint magnitude
            scaled_tol = atol + rtol * max(1.0, abs(c_val))

            if c["type"] == "ineq" and c_val < -scaled_tol:
                # Inequality constraint violated (should be >= 0)
                violation = -c_val
                max_violation = max(max_violation, violation)
                constraints_violated = True
            elif c["type"] == "eq" and abs(c_val) > scaled_tol:
                # Equality constraint violated (should be == 0)
                violation = abs(c_val)
                max_violation = max(max_violation, violation)
                constraints_violated = True

    # If SLSQP returned "optimal" but constraints are violated, retry with trust-constr
    if constraints_violated and method == "SLSQP":
        warnings.warn(
            f"SLSQP returned a solution that violates constraints (max violation: {max_violation:.2e}). "
            "Retrying with trust-constr method for more robust optimization.",
            UserWarning,
            stacklevel=3,
        )
        # Recursive call with trust-constr
        return solve_scipy(
            problem=problem,
            method="trust-constr",
            x0=x0,
            tol=tol,
            maxiter=maxiter,
            use_hessian=use_hessian,
            strict=strict,
            **kwargs,
        )

    # Map SciPy result to Solution
    if result.success and not constraints_violated:
        status = SolverStatus.OPTIMAL
    elif "maximum" in result.message.lower() and "iteration" in result.message.lower():
        status = SolverStatus.MAX_ITERATIONS
    elif "infeasible" in result.message.lower() or constraints_violated:
        status = SolverStatus.INFEASIBLE
    elif "positive directional derivative" in result.message.lower():
        # SLSQP reports this when it converged but hit numerical precision limits
        # The solution is typically still good - treat as optimal
        status = SolverStatus.OPTIMAL
    else:
        status = SolverStatus.FAILED

    # Compute actual objective value (undo negation for maximize)
    obj_value = float(result.fun)
    if problem.sense == "maximize":
        obj_value = -obj_value

    # Build message, noting if problem appears linear
    message = result.message if hasattr(result, "message") else ""
    if linear_problem_detected:
        message = f"{message} (Note: problem appears linear)"

    return Solution(
        status=status,
        objective_value=obj_value,
        values={v.name: float(result.x[i]) for i, v in enumerate(variables)},
        iterations=result.nit if hasattr(result, "nit") else None,
        message=message,
        solve_time=solve_time,
    )


def _compute_initial_point(
    variables: list, problem: "Problem | None" = None
) -> np.ndarray:
    """Compute a reasonable initial point from variable bounds.

    Strategy:
    - If both bounds exist: use interior point lb + epsilon*(ub-lb) to avoid
      singularities at boundaries (e.g., log(0), 1/0)
    - If only lower bound: use lb + epsilon to stay interior
    - If only upper bound: use ub - 1
    - If unbounded: use 0

    Note: Using strictly interior points avoids singularities for functions
    like log(x), 1/x, sqrt(x) when lb=0. The epsilon offset (1% of range or
    1e-4 minimum) provides a safe starting point.
    """
    x0 = np.zeros(len(variables))

    # Small epsilon for interior point calculation
    _INTERIOR_EPSILON = 1e-4
    _INTERIOR_FRACTION = 0.01  # 1% of range

    for i, v in enumerate(variables):
        lb = v.lb if v.lb is not None else -np.inf
        ub = v.ub if v.ub is not None else np.inf

        if np.isfinite(lb) and np.isfinite(ub):
            # Use interior point: lb + small fraction of range
            range_size = ub - lb
            epsilon = max(_INTERIOR_EPSILON, _INTERIOR_FRACTION * range_size)
            # Ensure we don't exceed upper bound
            x0[i] = min(lb + epsilon, (lb + ub) / 2)
        elif np.isfinite(lb):
            # Interior to lower bound
            x0[i] = lb + _INTERIOR_EPSILON
        elif np.isfinite(ub):
            x0[i] = ub - 1.0
        else:
            x0[i] = 0.0

    return x0


def _build_solver_cache(problem: Problem, variables: list) -> dict[str, Any]:
    """Build and cache compiled callables for the solver.

    This function compiles the objective, gradient, constraints, and bounds
    once and stores them in a cache dict. Subsequent solve() calls reuse
    these compiled callables, avoiding recompilation overhead.

    Args:
        problem: The optimization problem.
        variables: Ordered list of decision variables.

    Returns:
        Dict containing compiled callables and constraint data.
    """
    from optyx.core.autodiff import compile_jacobian
    from optyx.core.compiler import compile_expression

    cache: dict[str, Any] = {}

    # Build objective function
    obj_expr = problem.objective
    if obj_expr is None:
        raise ValueError("Problem has no objective to optimize")
    if problem.sense == "maximize":
        obj_expr = -obj_expr  # Negate for maximization

    cache["obj_fn"] = compile_expression(obj_expr, variables)
    cache["grad_fn"] = compile_jacobian([obj_expr], variables)

    # Build bounds
    bounds = []
    for v in variables:
        lb = v.lb if v.lb is not None else -np.inf
        ub = v.ub if v.ub is not None else np.inf
        bounds.append((lb, ub))
    cache["bounds"] = bounds

    # Build constraints for SciPy
    scipy_constraints = []

    for c in problem.constraints:
        c_expr = c.expr
        if c_expr is None:
            continue
        c_fn = compile_expression(c_expr, variables)
        c_jac_fn = compile_jacobian([c_expr], variables)

        if c.sense == ">=":
            # f(x) >= 0 → SciPy ineq: f(x) >= 0 (return f(x))
            scipy_constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, fn=c_fn: float(fn(x)),
                    "jac": lambda x, jfn=c_jac_fn: jfn(x).flatten(),
                }
            )
        elif c.sense == "<=":
            # f(x) <= 0 → SciPy ineq: -f(x) >= 0 (return -f(x))
            scipy_constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, fn=c_fn: -float(fn(x)),
                    "jac": lambda x, jfn=c_jac_fn: -jfn(x).flatten(),
                }
            )
        else:  # ==
            scipy_constraints.append(
                {
                    "type": "eq",
                    "fun": lambda x, fn=c_fn: float(fn(x)),
                    "jac": lambda x, jfn=c_jac_fn: jfn(x).flatten(),
                }
            )

    cache["scipy_constraints"] = scipy_constraints

    return cache
