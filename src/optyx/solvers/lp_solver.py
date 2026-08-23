"""Linear programming solver using scipy.optimize.linprog.

Provides a fast path for linear problems, bypassing gradient computation
and using the HiGHS solver for efficient LP solving.
"""

from __future__ import annotations

import time
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy

from optyx.core.errors import (
    NoObjectiveError,
    NonLinearError,
    SolverError,
)
from optyx.solvers.feasibility import (
    DEFAULT_FEASIBILITY_ATOL,
    compute_linear_problem_violation,
)

if TYPE_CHECKING:
    from optyx.problem import Problem
    from optyx.solution import Solution


# Minimum SciPy version for HiGHS solver
MIN_SCIPY_VERSION = "1.6.0"


def _check_scipy_version() -> bool:
    """Check if SciPy version supports HiGHS solver."""
    from packaging import version

    return version.parse(scipy.__version__) >= version.parse(MIN_SCIPY_VERSION)


def solve_lp(
    problem: Problem,
    method: str | None = None,
    strict: bool = False,
    **kwargs: Any,
) -> Solution:
    """Solve a linear programming problem using scipy.optimize.linprog.

    This function provides a fast path for linear problems, using the HiGHS
    solver which is significantly faster than general nonlinear solvers for LP.

    Args:
        problem: The optimization problem to solve. Must be a linear program
            (linear objective and linear constraints).
        method: LP solver method. Options:
            - None or "highs" (default): Automatic HiGHS method selection
            - "highs-ds": HiGHS dual simplex
            - "highs-ipm": HiGHS interior point method
        strict: If True, raise ValueError when the problem contains features
            that the solver cannot handle exactly (e.g., integer/binary
            variables). If False (default), emit a warning and relax.
        **kwargs: Additional arguments passed to scipy.optimize.linprog.

    Returns:
        Solution object with optimization results.

    Raises:
        ValueError: If the problem is not a valid linear program, or if
            strict=True and the problem contains unsupported features.
    """
    from scipy.optimize import linprog

    from optyx.analysis import LinearProgramExtractor, is_linear
    from optyx.solution import Solution, SolverStatus

    # Validate that the problem is linear
    if problem.objective is None:
        raise NoObjectiveError(
            suggestion="Call minimize() or maximize() on the problem first.",
        )

    if not is_linear(problem.objective):
        raise NonLinearError(
            expression=repr(problem.objective)[:100],
            context="LP solver",
            suggestion="Use solve() with a nonlinear solver for quadratic or nonlinear objectives.",
        )

    for constraint in problem.constraints:
        if not is_linear(constraint.expr):
            raise NonLinearError(
                expression=repr(constraint.expr)[:100],
                context="LP solver constraint",
                suggestion="Use solve() with a nonlinear solver for nonlinear constraints.",
            )

    # Check for non-continuous domains — route to MILP solver
    source_vector = problem._single_vector_source()
    variables = None
    non_continuous = []
    if source_vector is not None:
        if source_vector._has_non_continuous_domain():
            variables = problem.variables
            non_continuous = [v for v in variables if v.domain != "continuous"]
    else:
        variables = problem.variables
        non_continuous = [v for v in variables if v.domain != "continuous"]

    if non_continuous:
        from optyx.solvers.milp_solver import solve_milp

        # Extract LP coefficients (use cache if available)
        if (
            problem._lp_cache is not None
            and problem._lp_cache.parameters_are_current()
        ):
            lp_data = problem._lp_cache
        else:
            extractor = LinearProgramExtractor()
            try:
                lp_data = extractor.extract(problem)
                problem._lp_cache = lp_data
            except Exception as e:
                raise SolverError(
                    message=f"Failed to extract LP coefficients: {e}",
                    solver_name="milp",
                ) from e

        assert variables is not None
        return solve_milp(lp_data, variables, **kwargs)

    # Check SciPy version and select method
    if method is None:
        method = "highs"

    if not _check_scipy_version():
        warnings.warn(
            f"HiGHS solver requires SciPy >= {MIN_SCIPY_VERSION}. "
            f"Current version: {scipy.__version__}. Falling back to 'highs-ds'.",
            UserWarning,
            stacklevel=2,
        )
        # For older scipy, highs-ds is usually available
        method = "highs-ds"

    # Extract LP coefficients (use cache if available)
    if problem._lp_cache is not None and problem._lp_cache.parameters_are_current():
        lp_data = problem._lp_cache
    else:
        extractor = LinearProgramExtractor()
        try:
            lp_data = extractor.extract(problem)
            problem._lp_cache = lp_data  # Cache for future solves
        except Exception as e:
            raise SolverError(
                message=f"Failed to extract LP coefficients: {e}",
                solver_name="linprog",
            ) from e

    # Handle maximization by negating objective
    c = lp_data.c
    if lp_data.sense == "max":
        c = -c

    # Build linprog arguments
    linprog_kwargs: dict[str, Any] = {
        "c": c,
        "method": method,
    }

    if lp_data.A_ub is not None and lp_data.b_ub is not None:
        linprog_kwargs["A_ub"] = lp_data.A_ub
        linprog_kwargs["b_ub"] = lp_data.b_ub

    if lp_data.A_eq is not None and lp_data.b_eq is not None:
        linprog_kwargs["A_eq"] = lp_data.A_eq
        linprog_kwargs["b_eq"] = lp_data.b_eq

    # Always re-extract bounds from live variable properties to ensure
    # updates to v.lb/v.ub are respected even when LP data is cached.
    fresh_bounds = problem.get_bounds()
    if fresh_bounds:
        linprog_kwargs["bounds"] = fresh_bounds

    # Merge user kwargs (allow overriding)
    linprog_kwargs.update(kwargs)

    # Solve
    start_time = time.perf_counter()

    try:
        result = linprog(**linprog_kwargs)
    except Exception as e:
        return Solution(
            status=SolverStatus.FAILED,
            message=str(e),
            solve_time=time.perf_counter() - start_time,
        )

    solve_time = time.perf_counter() - start_time

    # Map linprog result to Solution
    if result.success:
        status = SolverStatus.OPTIMAL
    elif result.status == 2:  # Infeasible
        status = SolverStatus.INFEASIBLE
    elif result.status == 3:  # Unbounded
        status = SolverStatus.UNBOUNDED
    elif result.status == 1:  # Iteration limit
        status = SolverStatus.MAX_ITERATIONS
    else:
        status = SolverStatus.FAILED

    # Build values dictionary
    # For unbounded problems, result.x may contain a direction (ray)
    values: dict[str, float] = {}
    if result.x is not None:
        for i, var_name in enumerate(lp_data.variables):
            values[var_name] = float(result.x[i])

    # Compute actual objective value (undo negation for maximization)
    objective_value: float | None = None
    if result.fun is not None:
        objective_value = float(result.fun)
        if lp_data.sense == "max":
            objective_value = -objective_value

    # Build informative message for unbounded/infeasible cases
    message = result.message if hasattr(result, "message") else ""
    if status == SolverStatus.UNBOUNDED and values:
        message = f"{message} Unbounded direction available in solution.values."
    elif status == SolverStatus.INFEASIBLE:
        message = f"{message} No feasible solution exists."

    lb = np.array(
        [bound[0] if bound[0] is not None else -np.inf for bound in fresh_bounds]
    )
    ub = np.array(
        [bound[1] if bound[1] is not None else np.inf for bound in fresh_bounds]
    )
    constraint_violation = compute_linear_problem_violation(
        result.x,
        A_ub=lp_data.A_ub,
        b_ub=lp_data.b_ub,
        A_eq=lp_data.A_eq,
        b_eq=lp_data.b_eq,
        lb=lb,
        ub=ub,
    )
    feasibility_tolerance = (
        DEFAULT_FEASIBILITY_ATOL if constraint_violation is not None else None
    )

    return Solution(
        status=status,
        objective_value=objective_value,
        values=values,
        iterations=result.nit if hasattr(result, "nit") else None,
        message=message,
        solve_time=solve_time,
        constraint_violation=constraint_violation,
        feasibility_tolerance=feasibility_tolerance,
    )
