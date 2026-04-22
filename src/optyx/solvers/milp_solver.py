"""Mixed-Integer Linear Programming solver using scipy.optimize.milp.

Routes linear problems with integer/binary variables to scipy's HiGHS-based
MILP solver. Only supports linear objectives and constraints (no MIQP/MINLP).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

from optyx.solution import Solution, SolverStatus

if TYPE_CHECKING:
    from optyx.analysis import LPData
    from optyx.core.expressions import Variable


def solve_milp(
    lp_data: LPData,
    variables: list[Variable],
    **kwargs: Any,
) -> Solution:
    """Solve a mixed-integer linear program via scipy.optimize.milp().

    Args:
        lp_data: Extracted LP data (objective, constraints, bounds).
        variables: Ordered list of Variable objects (needed for domain info).
        **kwargs: Additional arguments passed to scipy.optimize.milp.

    Returns:
        Solution object with optimization results including mip_gap.
    """
    # Build objective coefficients (handle maximization)
    c = lp_data.c
    is_max = lp_data.sense == "max"
    if is_max:
        c = -c

    # Build integrality array: 0=continuous, 1=integer (vectorized)
    domains = np.array([var.domain for var in variables], dtype=object)
    integrality = ((domains == "integer") | (domains == "binary")).astype(int)

    # Always rebuild bounds from live variable attributes so cached LP
    # structure respects bound mutations across re-solves.
    raw_bounds = [(var.lb, var.ub) for var in variables]
    lb_arr = np.array([b[0] if b[0] is not None else -np.inf for b in raw_bounds])
    ub_arr = np.array([b[1] if b[1] is not None else np.inf for b in raw_bounds])
    bounds = Bounds(
        lb=cast(Any, lb_arr),
        ub=cast(Any, ub_arr),
    )

    # Build constraints for milp() using LinearConstraint
    constraints_list: list[LinearConstraint] = []

    if lp_data.A_ub is not None and lp_data.b_ub is not None:
        A_ub = lp_data.A_ub
        b_ub = lp_data.b_ub
        m_ub = b_ub.shape[0] if hasattr(b_ub, "shape") else len(b_ub)
        # A_ub @ x <= b_ub  →  -inf <= A_ub @ x <= b_ub
        constraints_list.append(
            LinearConstraint(
                A_ub,
                cast(Any, np.full(m_ub, -np.inf)),
                cast(Any, b_ub),
            )
        )

    if lp_data.A_eq is not None and lp_data.b_eq is not None:
        A_eq = lp_data.A_eq
        b_eq = lp_data.b_eq
        # A_eq @ x == b_eq  →  b_eq <= A_eq @ x <= b_eq
        constraints_list.append(
            LinearConstraint(A_eq, cast(Any, b_eq), cast(Any, b_eq))
        )

    # Solve
    start_time = time.perf_counter()

    milp_kwargs: dict[str, Any] = {}
    milp_kwargs.update(kwargs)

    try:
        result = milp(
            c,
            constraints=constraints_list if constraints_list else None,
            integrality=integrality,
            bounds=bounds,
            **milp_kwargs,
        )
    except Exception as e:
        return Solution(
            status=SolverStatus.FAILED,
            message=str(e),
            solve_time=time.perf_counter() - start_time,
        )

    solve_time = time.perf_counter() - start_time

    # Map milp result status
    status = _map_milp_status(result)

    # Build values dictionary
    values: dict[str, float] = {}
    if result.x is not None:
        for i, var_name in enumerate(lp_data.variables):
            values[var_name] = float(result.x[i])

    # Compute objective value (undo negation for maximization)
    objective_value: float | None = None
    if result.fun is not None and np.isfinite(result.fun):
        objective_value = float(result.fun)
        if is_max:
            objective_value = -objective_value

    # Extract MIP gap if available
    mip_gap: float | None = None
    if hasattr(result, "mip_gap") and result.mip_gap is not None:
        mip_gap = float(result.mip_gap)

    # Extract best bound if available
    best_bound: float | None = None
    if hasattr(result, "mip_dual_bound") and result.mip_dual_bound is not None:
        best_bound = float(result.mip_dual_bound)
        if is_max:
            best_bound = -best_bound

    message = result.message if hasattr(result, "message") else ""

    return Solution(
        status=status,
        objective_value=objective_value,
        values=values,
        message=message,
        solve_time=solve_time,
        mip_gap=mip_gap,
        best_bound=best_bound,
    )


def _map_milp_status(result: Any) -> SolverStatus:
    """Map scipy.optimize.milp result to SolverStatus."""
    if result.success:
        return SolverStatus.OPTIMAL

    # scipy milp status codes:
    # 0: Optimal
    # 1: Iteration or time limit
    # 2: Infeasible
    # 3: Unbounded
    # 4: Error
    status_code = getattr(result, "status", -1)

    if status_code == 1:
        return SolverStatus.MAX_ITERATIONS
    elif status_code == 2:
        return SolverStatus.INFEASIBLE
    elif status_code == 3:
        return SolverStatus.UNBOUNDED
    else:
        return SolverStatus.FAILED
