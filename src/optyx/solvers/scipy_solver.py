"""SciPy-based optimization solver.

Maps Optyx problems to scipy.optimize.minimize for solving.
"""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize

from optyx.core.errors import (
    IntegerVariableError,
    NoObjectiveError,
    UnsupportedOperationError,
)

if TYPE_CHECKING:
    from optyx.problem import Problem
    from optyx.solution import Solution


class _EarlyTermination(Exception):
    """Raised inside SciPy callback to terminate optimization early."""

    def __init__(self, x: np.ndarray, iteration: int, message: str) -> None:
        self.x = x
        self.iteration = iteration
        self.message = message


def solve_scipy(
    problem: Problem,
    method: str = "SLSQP",
    x0: np.ndarray | None = None,
    tol: float | None = None,
    maxiter: int | None = None,
    use_hessian: bool = True,
    strict: bool = False,
    warm_start: bool = True,
    callback: Callable | None = None,
    time_limit: float | None = None,
    **kwargs: Any,
) -> Solution:
    """Solve an optimization problem using SciPy.

    Args:
        problem: The optimization problem to solve.
        method: SciPy optimization method. Options:
            - "SLSQP": Sequential Least Squares Programming (default)
            - "trust-constr": Trust-region constrained optimization
            - "L-BFGS-B": Limited-memory BFGS with bounds (no constraints)
        x0: Initial point. If None, uses warm start (previous solution) when
            available, otherwise midpoint of bounds or zeros.
        tol: Solver tolerance.
        maxiter: Maximum number of iterations.
        use_hessian: Whether to compute and pass the symbolic Hessian to methods
            that support it (trust-constr, Newton-CG, etc.). Default True.
            Set to False if Hessian computation is too expensive.
        strict: Retained for API compatibility with Problem.solve(). Direct
            SciPy solves always reject integer/binary domains because SciPy
            cannot enforce them.
        warm_start: If True (default), use the previous solution stored on the
            Problem as the initial point when x0 is not explicitly provided.
        callback: Optional function receiving a SolverProgress object each
            iteration.  Return True to terminate early.
        time_limit: Maximum wall-clock seconds for the solve.  Terminates
            early with SolverStatus.TERMINATED when exceeded.
        **kwargs: Additional arguments passed to scipy.optimize.minimize.

    Returns:
        Solution object with optimization results.

    Raises:
        IntegerVariableError: If a linear discrete model is sent directly to a
            continuous SciPy solver.
        UnsupportedOperationError: If a nonlinear discrete model (MIQP/MINLP)
            reaches this solver directly.
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

    # Check for non-continuous domains — always raise for MINLP.
    # The caller (Problem.solve) should have caught this already, but
    # guard here as a safety net.
    non_continuous = [v for v in variables if v.domain != "continuous"]
    if non_continuous:
        if problem._is_linear_problem():
            raise IntegerVariableError(
                solver_name="SciPy",
                variable_names=[v.name for v in non_continuous],
            )

        raise UnsupportedOperationError(
            "MIQP/MINLP solve",
            solver_name="SciPy",
            problem_feature=(
                "nonlinear objective or constraints with integer/binary "
                f"variables {[v.name for v in non_continuous]}"
            ),
            suggestion=(
                "Use the MILP path for linear discrete models, or switch to a "
                "dedicated MIQP/MINLP solver"
            ),
        )

    # Check for cached compiled callables
    cache = problem._solver_cache
    if cache is None:
        cache = _build_solver_cache(problem, variables)
        problem._solver_cache = cache
    elif "scipy_constraints" not in cache:
        # Selective invalidation: objective cache preserved, rebuild constraints only
        _rebuild_constraint_cache(cache, problem, variables)

    # Extract cached callables
    obj_fn = cache["obj_fn"]
    grad_fn = cache["grad_fn"]
    scipy_constraints = cast(list[dict[str, Any]], cache["scipy_constraints"])
    linear_constraints = cast(
        list[LinearConstraint], cache.get("linear_constraints", [])
    )

    # Recompute bounds each time to ensure updates to variable properties are respected
    lb_arr = np.empty(n)
    ub_arr = np.empty(n)
    for i, v in enumerate(variables):
        lb_arr[i] = v.lb if v.lb is not None else -np.inf
        ub_arr[i] = v.ub if v.ub is not None else np.inf
    bounds = Bounds(lb=lb_arr, ub=ub_arr)  # type: ignore[arg-type]  # scipy stubs are wrong, Bounds accepts arrays

    def objective(x: np.ndarray) -> float:
        return float(obj_fn(x))

    def gradient(x: np.ndarray) -> np.ndarray:
        grad = grad_fn(x)
        if hasattr(grad, "toarray"):
            return np.asarray(grad.toarray(), dtype=np.float64).ravel()
        return np.asarray(grad, dtype=np.float64).ravel()

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

    # Initial point: explicit x0 > warm start > computed
    if x0 is None:
        if (
            warm_start
            and problem._last_solution is not None
            and len(problem._last_solution) == n
        ):
            x0 = problem._last_solution.copy()
        else:
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

    # For trust-constr, use the vector-valued NonlinearConstraint with the
    # batched sparse Jacobian when available.  Fall back to the old scalar
    # constraint dicts for all other methods (e.g. SLSQP).
    constraint_monitors: list[Any] = []
    if method == "trust-constr":
        tc_constraint = _build_trust_constr_constraints(cache)
        if tc_constraint is not None:
            constraint_monitors.append(tc_constraint)
    else:
        constraint_monitors.extend(scipy_constraints)

    constraint_monitors.extend(linear_constraints)
    constraints_arg: Any = constraint_monitors if constraint_monitors else ()

    # Build composite callback for user callback and/or time_limit
    scipy_callback = _build_scipy_callback(
        callback=callback,
        time_limit=time_limit,
        start_time=start_time,
        obj_fn=obj_fn,
        constraints=constraint_monitors,
        sense=problem.sense,
        method=method,
    )

    try:
        # Temporarily override warning handling during solve
        warnings.showwarning = warning_handler

        result = minimize(
            fun=objective,
            x0=x0,
            method=method,
            jac=gradient if use_gradient else None,
            hess=hess_fn if hess_fn is not None else None,
            bounds=bounds if method in BOUNDS_METHODS else None,
            constraints=constraints_arg,
            tol=tol,
            options=options if options else None,
            callback=scipy_callback,
            **kwargs,
        )
    except _EarlyTermination as et:
        warnings.showwarning = old_showwarning
        solve_time = time.perf_counter() - start_time
        obj_value = float(obj_fn(et.x))
        if problem.sense == "maximize":
            obj_value = -obj_value
        return Solution(
            status=SolverStatus.TERMINATED,
            objective_value=obj_value,
            values={v.name: float(et.x[i]) for i, v in enumerate(variables)},
            iterations=et.iteration,
            message=et.message,
            solve_time=solve_time,
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

    if result.success and constraint_monitors:
        max_violation = _compute_constraint_violation(
            result.x,
            constraint_monitors,
            atol=atol,
            rtol=rtol,
        )
        constraints_violated = max_violation > 0.0

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
            callback=callback,
            time_limit=time_limit,
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
    from optyx.core.autodiff import analyze_gradient_sparsity, compile_jacobian
    from optyx.core.compiler import (
        compile_expression,
        compile_sparse_gradient_dense_output,
    )
    from optyx.core.optimizer import flatten_expression

    cache: dict[str, Any] = {}

    # Build objective function
    obj_expr = problem.objective
    if obj_expr is None:
        raise NoObjectiveError(
            suggestion="Call minimize() or maximize() on the problem first.",
        )

    # Add Variable.obj contributions (linear objective coefficients set at creation)
    obj_vars = [v for v in variables if v.obj != 0.0]
    if obj_vars:
        from optyx.core.expressions import Constant

        obj_contrib = Constant(0.0)
        for v in obj_vars:
            obj_contrib = obj_contrib + Constant(v.obj) * v
        obj_expr = obj_expr + obj_contrib

    if problem.sense == "maximize":
        obj_expr = -obj_expr  # Negate for maximization

    # Flatten deep expression trees before compilation
    obj_expr = flatten_expression(obj_expr)

    cache["obj_fn"] = compile_expression(obj_expr, variables)

    # SciPy expects dense objective gradients. For sparse objectives, compile
    # only the non-zero partials and scatter them into a dense vector.
    obj_grad_sparsity = analyze_gradient_sparsity(obj_expr, variables)
    if obj_grad_sparsity.density <= 0.5:
        cache["grad_fn"] = compile_sparse_gradient_dense_output(obj_expr, variables)
    else:
        cache["grad_fn"] = compile_jacobian([obj_expr], variables)

    # Build constraints for SciPy
    scipy_constraints = []
    constraint_exprs = []
    constraint_fns = []
    constraint_senses = []

    for c in problem.constraints:
        c_expr = c.expr
        if c_expr is None:
            continue
        c_expr = flatten_expression(c_expr)
        c_fn = compile_expression(c_expr, variables)
        c_jac_fn = compile_jacobian([c_expr], variables)

        constraint_exprs.append(c_expr)
        constraint_fns.append(c_fn)
        constraint_senses.append(c.sense)

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
    cache["linear_constraints"] = _build_matrix_linear_constraints(problem, variables)

    # trust-constr builds the batched sparse Jacobian lazily so SLSQP-only
    # solves don't pay cold-start compilation cost for unused sparse data.
    if constraint_exprs:
        cache["constraint_exprs"] = constraint_exprs
        cache["constraint_fns"] = constraint_fns
        cache["constraint_senses"] = constraint_senses
        cache["constraint_variables"] = variables

    return cache


def _rebuild_constraint_cache(
    cache: dict[str, Any], problem: Problem, variables: list
) -> None:
    """Rebuild only the constraint portion of the solver cache.

    Called when constraints have been added/removed but the objective
    hasn't changed, so obj_fn/grad_fn/hess_fn are still valid.
    """
    from optyx.core.autodiff import compile_jacobian
    from optyx.core.compiler import compile_expression
    from optyx.core.optimizer import flatten_expression

    scipy_constraints: list[dict[str, Any]] = []
    constraint_exprs = []
    constraint_fns = []
    constraint_senses = []

    for c in problem.constraints:
        c_expr = c.expr
        if c_expr is None:
            continue
        c_expr = flatten_expression(c_expr)
        c_fn = compile_expression(c_expr, variables)
        c_jac_fn = compile_jacobian([c_expr], variables)

        constraint_exprs.append(c_expr)
        constraint_fns.append(c_fn)
        constraint_senses.append(c.sense)

        if c.sense == ">=":
            scipy_constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x, fn=c_fn: float(fn(x)),
                    "jac": lambda x, jfn=c_jac_fn: jfn(x).flatten(),
                }
            )
        elif c.sense == "<=":
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
    cache["linear_constraints"] = _build_matrix_linear_constraints(problem, variables)

    if constraint_exprs:
        cache["constraint_exprs"] = constraint_exprs
        cache["constraint_fns"] = constraint_fns
        cache["constraint_senses"] = constraint_senses
        cache["constraint_variables"] = variables
    else:
        cache.pop("sparse_constraint_jac_fn", None)
        cache.pop("constraint_exprs", None)
        cache.pop("constraint_fns", None)
        cache.pop("constraint_senses", None)
        cache.pop("constraint_variables", None)


def _build_trust_constr_constraints(
    cache: dict[str, Any],
) -> NonlinearConstraint | None:
    """Build a single NonlinearConstraint for trust-constr from cached data.

    Uses the batched sparse Jacobian compiled by ``compile_sparse_jacobian``
    and the vector of scalar constraint functions already stored in the cache.

    Returns ``None`` when there are no constraints.
    """
    constraint_fns = cache.get("constraint_fns")
    constraint_senses = cast(list[str] | None, cache.get("constraint_senses"))
    sparse_jac_fn = cache.get("sparse_constraint_jac_fn")

    if not constraint_fns or constraint_senses is None:
        return None

    if sparse_jac_fn is None:
        from optyx.core.autodiff import compile_sparse_jacobian

        constraint_exprs = cache.get("constraint_exprs")
        constraint_variables = cache.get("constraint_variables")
        if not constraint_exprs or constraint_variables is None:
            return None
        sparse_jac_fn = compile_sparse_jacobian(constraint_exprs, constraint_variables)
        cache["sparse_constraint_jac_fn"] = sparse_jac_fn

    m = len(constraint_fns)

    # Build lb / ub vectors from senses.
    # Stored expressions are normalised so that the constraint reads
    #   expr(x)  {>=, <=, ==}  0
    lb = np.full(m, -np.inf)
    ub = np.full(m, np.inf)
    for i, sense in enumerate(constraint_senses):
        if sense == ">=":
            lb[i] = 0.0
        elif sense == "<=":
            ub[i] = 0.0
        else:  # "=="
            lb[i] = 0.0
            ub[i] = 0.0

    # Vector-valued constraint function.
    def _constraint_vector(x: np.ndarray) -> np.ndarray:
        out = np.empty(m)
        for i, fn in enumerate(constraint_fns):
            out[i] = float(fn(x))
        return out

    jac_callback: Any = sparse_jac_fn

    return NonlinearConstraint(
        fun=_constraint_vector,
        lb=lb,
        ub=ub,
        jac=cast(Any, jac_callback),
    )


def _build_matrix_linear_constraints(
    problem: Problem,
    variables: list,
) -> list[LinearConstraint]:
    """Build SciPy LinearConstraint objects for stored matrix blocks."""
    from scipy import sparse as sp

    if not problem._matrix_constraints:
        return []

    n = len(variables)
    var_index = {var.name: i for i, var in enumerate(variables)}
    linear_constraints: list[LinearConstraint] = []

    for mc in problem._matrix_constraints:
        mc_n = len(mc.variables)
        col_indices = np.array([var_index[v.name] for v in mc.variables], dtype=np.intp)

        if sp.issparse(mc.A):
            if mc_n == n and np.array_equal(col_indices, np.arange(n)):
                A_full = mc.A
            else:
                permutation = sp.csr_matrix(
                    (np.ones(mc_n), (np.arange(mc_n), col_indices)),
                    shape=(mc_n, n),
                )
                A_full = (mc.A @ permutation).tocsr()
        else:
            A_base = np.asarray(mc.A, dtype=np.float64)
            if mc_n == n and np.array_equal(col_indices, np.arange(n)):
                A_full = A_base
            else:
                A_full = np.zeros((A_base.shape[0], n), dtype=np.float64)
                A_full[:, col_indices] = A_base

        m = A_full.shape[0]
        if mc.sense == "<=":
            lb = np.full(m, -np.inf)
            ub = mc.b
        elif mc.sense == ">=":
            lb = mc.b
            ub = np.full(m, np.inf)
        else:
            lb = mc.b
            ub = mc.b

        linear_constraints.append(
            LinearConstraint(A_full, lb=cast(Any, lb), ub=cast(Any, ub))
        )

    return linear_constraints


def _compute_constraint_violation(
    x: np.ndarray,
    constraints: list[Any],
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> float:
    """Compute maximum constraint violation for current iterate."""
    max_violation = 0.0
    for c in constraints:
        if isinstance(c, dict):
            val = float(c["fun"](x))
            scaled_tol = atol + rtol * max(1.0, abs(val))
            if c["type"] == "ineq":
                violation = -val if val < -scaled_tol else 0.0
            else:
                violation = abs(val) if abs(val) > scaled_tol else 0.0
        elif isinstance(c, LinearConstraint):
            values = np.asarray(c.A @ x, dtype=np.float64).reshape(-1)
            violation = _compute_bound_violation(
                values, c.lb, c.ub, atol=atol, rtol=rtol
            )
        elif isinstance(c, NonlinearConstraint):
            values = np.asarray(c.fun(x), dtype=np.float64).reshape(-1)
            violation = _compute_bound_violation(
                values, c.lb, c.ub, atol=atol, rtol=rtol
            )
        else:
            continue

        max_violation = max(max_violation, violation)
    return max_violation


def _compute_bound_violation(
    values: np.ndarray,
    lb: Any,
    ub: Any,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> float:
    """Compute maximum bound-style constraint violation for a vector of values."""
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    lb_arr = np.broadcast_to(np.asarray(lb, dtype=np.float64), vals.shape)
    ub_arr = np.broadcast_to(np.asarray(ub, dtype=np.float64), vals.shape)

    max_violation = 0.0

    lower_mask = np.isfinite(lb_arr)
    if np.any(lower_mask):
        lower_vals = vals[lower_mask]
        lower_bounds = lb_arr[lower_mask]
        lower_scale = np.maximum(
            1.0, np.maximum(np.abs(lower_vals), np.abs(lower_bounds))
        )
        lower_tol = atol + rtol * lower_scale
        lower_violation = lower_bounds - lower_vals
        lower_violation = lower_violation[lower_violation > lower_tol]
        if lower_violation.size:
            max_violation = max(max_violation, float(np.max(lower_violation)))

    upper_mask = np.isfinite(ub_arr)
    if np.any(upper_mask):
        upper_vals = vals[upper_mask]
        upper_bounds = ub_arr[upper_mask]
        upper_scale = np.maximum(
            1.0, np.maximum(np.abs(upper_vals), np.abs(upper_bounds))
        )
        upper_tol = atol + rtol * upper_scale
        upper_violation = upper_vals - upper_bounds
        upper_violation = upper_violation[upper_violation > upper_tol]
        if upper_violation.size:
            max_violation = max(max_violation, float(np.max(upper_violation)))

    return max_violation


def _build_scipy_callback(
    *,
    callback: Callable | None,
    time_limit: float | None,
    start_time: float,
    obj_fn: Callable,
    constraints: list[Any],
    sense: str,
    method: str,
) -> Callable | None:
    """Build a SciPy-compatible callback combining user callback and time limit.

    Returns None if neither callback nor time_limit is provided.
    """
    if callback is None and time_limit is None:
        return None

    from optyx.solution import SolverProgress

    # Mutable iteration counter
    state = {"iteration": 0}

    def _unified_callback(xk, *args):
        """Callback invoked by SciPy at each iteration.

        For trust-constr, args[0] is an OptimizeResult with state info.
        For other methods, only xk (current x) is provided.
        """
        state["iteration"] += 1
        elapsed = time.perf_counter() - start_time

        # Get current x — trust-constr passes OptimizeResult as first arg
        if hasattr(xk, "x"):
            # trust-constr passes OptimizeResult object
            x = np.asarray(xk.x)
        else:
            x = np.asarray(xk)

        # Compute objective value (undo negation for maximize)
        obj_val = float(obj_fn(x))
        if sense == "maximize":
            obj_val = -obj_val

        # Compute constraint violation
        cv = _compute_constraint_violation(x, constraints)

        # Check time limit first
        if time_limit is not None and elapsed >= time_limit:
            raise _EarlyTermination(
                x=x,
                iteration=state["iteration"],
                message=f"Time limit ({time_limit:.1f}s) exceeded",
            )

        # Invoke user callback
        if callback is not None:
            progress = SolverProgress(
                iteration=state["iteration"],
                objective_value=obj_val,
                constraint_violation=cv,
                elapsed_time=elapsed,
                x=x.copy(),
            )
            stop = callback(progress)
            if stop is True:
                raise _EarlyTermination(
                    x=x,
                    iteration=state["iteration"],
                    message="Terminated by user callback",
                )

    return _unified_callback
