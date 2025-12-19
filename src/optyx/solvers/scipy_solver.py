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
    from optyx.core.autodiff import compile_hessian, compile_jacobian
    from optyx.core.compiler import compile_expression
    from optyx.solution import Solution, SolverStatus

    # Methods that support Hessian
    HESSIAN_METHODS = {
        "trust-constr",
        "Newton-CG",
        "dogleg",
        "trust-ncg",
        "trust-exact",
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

    # Build objective function
    obj_expr = problem.objective
    if obj_expr is None:
        raise ValueError("Problem has no objective to optimize")
    if problem.sense == "maximize":
        obj_expr = -obj_expr  # Negate for maximization

    obj_fn = compile_expression(obj_expr, variables)
    grad_fn = compile_jacobian([obj_expr], variables)

    def objective(x: np.ndarray) -> float:
        return float(obj_fn(x))

    def gradient(x: np.ndarray) -> np.ndarray:
        return grad_fn(x).flatten()

    # Build Hessian for methods that support it
    hess_fn: Callable[[np.ndarray], np.ndarray] | None = None
    if use_hessian and method in HESSIAN_METHODS:
        compiled_hess = compile_hessian(obj_expr, variables)

        def _hess_fn(x: np.ndarray) -> np.ndarray:
            return compiled_hess(x)

        hess_fn = _hess_fn

    # Build bounds
    bounds = []
    for v in variables:
        lb = v.lb if v.lb is not None else -np.inf
        ub = v.ub if v.ub is not None else np.inf
        bounds.append((lb, ub))

    # Build constraints for SciPy
    scipy_constraints = []

    for i, c in enumerate(problem.constraints):
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
        # Let other warnings through
        warnings.showwarning(message, category, filename, lineno, file, line)

    old_showwarning = warnings.showwarning
    try:
        # Temporarily override warning handling during solve
        warnings.showwarning = warning_handler

        result = minimize(
            fun=objective,
            x0=x0,
            method=method,
            jac=gradient,
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

    # Map SciPy result to Solution
    if result.success:
        status = SolverStatus.OPTIMAL
    elif "maximum" in result.message.lower() and "iteration" in result.message.lower():
        status = SolverStatus.MAX_ITERATIONS
    elif "infeasible" in result.message.lower():
        status = SolverStatus.INFEASIBLE
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


def _compute_initial_point(variables: list) -> np.ndarray:
    """Compute a reasonable initial point from variable bounds.

    Strategy:
    - If both bounds exist: use midpoint
    - If only lower bound: use lb + 1
    - If only upper bound: use ub - 1
    - If unbounded: use 0
    """
    x0 = np.zeros(len(variables))

    for i, v in enumerate(variables):
        lb = v.lb if v.lb is not None else -np.inf
        ub = v.ub if v.ub is not None else np.inf

        if np.isfinite(lb) and np.isfinite(ub):
            x0[i] = (lb + ub) / 2
        elif np.isfinite(lb):
            x0[i] = lb + 1.0
        elif np.isfinite(ub):
            x0[i] = ub - 1.0
        else:
            x0[i] = 0.0

    return x0
