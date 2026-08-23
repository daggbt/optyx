"""LP format file export for optimization problems.

Exports Problem objects to the standard LP file format, compatible with
solvers like CPLEX, Gurobi, GLPK, and HiGHS.

LP format reference:
    https://www.ibm.com/docs/en/icos/22.1.1?topic=cplex-lp-file-format-algebraic-representation

Supported features:
    - Linear and quadratic objectives
    - Linear constraints (<=, >=, ==)
    - Variable bounds
    - Integer (Generals) and binary (Binaries) variable types
    - Named constraints
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from optyx.analysis import (
    _get_scalar_constant_value,
    compute_degree,
    is_linear,
    extract_linear_coefficient,
    extract_constant_term,
)
from optyx.core.errors import InvalidOperationError
from optyx.core.parameters import Parameter
from optyx.core.expressions import (
    BinaryOp,
    Constant,
    Expression,
    NaryProduct,
    NarySum,
    UnaryOp,
    Variable,
)

if TYPE_CHECKING:
    from optyx.problem import Problem


def _compute_poly_degree(expr: Expression) -> int | None:
    """Return polynomial degree using the canonical analysis implementation."""
    return compute_degree(expr)


def _is_at_most_quadratic(expr: Expression) -> bool:
    """Check if an expression is at most quadratic (degree <= 2)."""
    d = _compute_poly_degree(expr)
    return d is not None and d <= 2


def write_lp(problem: Problem, filename: str) -> None:
    """Export a Problem to LP file format.

    Args:
        problem: The optimization problem to export.
        filename: Path to the output .lp file.

    Raises:
        InvalidOperationError: If the problem contains nonlinear (non-quadratic)
            expressions that cannot be represented in LP format.
        NoObjectiveError: If no objective has been set.
    """
    content = _format_lp(problem)
    with open(filename, "w") as f:
        f.write(content)


def format_lp(problem: Problem) -> str:
    """Format a Problem as an LP format string (without writing to file).

    Args:
        problem: The optimization problem to format.

    Returns:
        The LP format string.

    Raises:
        InvalidOperationError: If the problem contains nonlinear expressions.
    """
    return _format_lp(problem)


def _format_lp(problem: Problem) -> str:
    """Build the LP format string for a problem."""
    from optyx.core.errors import NoObjectiveError

    if problem.objective is None:
        raise NoObjectiveError(
            suggestion="Set an objective with minimize() or maximize() before exporting."
        )

    variables = problem.variables
    if not variables:
        raise InvalidOperationError(
            operation="LP export",
            operand_types="Problem",
            reason="Problem has no variables.",
        )

    # Validate: objective and constraints must be at most quadratic
    obj = problem.objective
    if not _is_at_most_quadratic(obj):
        raise InvalidOperationError(
            operation="LP export",
            operand_types=type(obj).__name__,
            reason="LP format only supports linear and quadratic objectives.",
            suggestion="Nonlinear objectives cannot be exported to LP format.",
        )

    for i, c in enumerate(problem.constraints):
        if not is_linear(c.expr):
            name = c.name or f"c{i}"
            raise InvalidOperationError(
                operation="LP export",
                operand_types=type(c.expr).__name__,
                reason=f"Constraint '{name}' is not linear. LP format only supports linear constraints.",
                suggestion="Only linear constraints can be exported to LP format.",
            )

    lines: list[str] = []

    # Comment header
    name = problem.name or "optyx_model"
    lines.append(f"\\ Model {name}")
    lines.append("")

    # Objective section
    _write_objective(lines, problem, variables)

    # Constraints section
    _write_constraints(lines, problem, variables)

    # Bounds section
    _write_bounds(lines, variables)

    # Integer / Binary variable sections
    _write_variable_types(lines, variables)

    lines.append("End")
    lines.append("")

    return "\n".join(lines)


def _write_objective(
    lines: list[str], problem: Problem, variables: list[Variable]
) -> None:
    """Write the objective function section."""
    obj = problem.objective
    assert obj is not None

    sense = "Minimize" if problem.sense == "minimize" else "Maximize"
    lines.append(sense)

    # Extract linear part
    linear_terms = _extract_linear_terms(obj, variables)
    constant = (
        extract_constant_term(obj)
        if is_linear(obj)
        else _extract_constant_from_quadratic(obj)
    )

    # Extract quadratic part (if any)
    quad_terms: list[tuple[str, str, float]] = []
    if not is_linear(obj):
        quad_terms = _extract_quadratic_terms(obj, variables)

    # Format the objective line
    obj_str = _format_expression(linear_terms, quad_terms, constant)
    lines.append(f"  obj: {obj_str}")
    lines.append("")


def _write_constraints(
    lines: list[str], problem: Problem, variables: list[Variable]
) -> None:
    """Write the constraints section."""
    all_constraints = problem.constraints
    matrix_constraints = problem._matrix_constraints

    if not all_constraints and not matrix_constraints:
        return

    lines.append("Subject To")

    # Expression-based constraints
    for i, c in enumerate(all_constraints):
        name = c.name or f"c{i}"
        linear_terms = _extract_linear_terms(c.expr, variables)
        constant = extract_constant_term(c.expr)

        # Constraint is: expr sense 0, so expr - constant sense -constant
        # We write: linear_part sense -constant
        lhs_str = _format_expression(linear_terms, [], 0.0)
        rhs = -constant

        if not lhs_str or lhs_str.strip() == "0":
            lhs_str = "0"

        sense_str = c.sense
        lines.append(f"  {name}: {lhs_str} {sense_str} {_format_number(rhs)}")

    # Matrix constraints: A @ x sense b
    constraint_offset = len(all_constraints)
    for mc in matrix_constraints:
        A = mc.A
        b = mc.b
        sense_str = mc.sense
        mc_vars = mc.variables

        # Build variable name list for this matrix constraint
        var_names = [v.name for v in mc_vars]

        m = A.shape[0]
        for row_idx in range(m):
            name = f"c{constraint_offset + row_idx}"

            # Extract row coefficients
            from scipy import sparse as sp

            if sp.issparse(A):
                row = A.getrow(row_idx)
                indices = row.indices
                data = row.data
                terms: list[tuple[str, float]] = [
                    (var_names[j], float(data[k]))
                    for k, j in enumerate(indices)
                    if abs(data[k]) > 1e-15
                ]
            else:
                row_data = A[row_idx]
                terms = [
                    (var_names[j], float(row_data[j]))
                    for j in range(len(var_names))
                    if abs(row_data[j]) > 1e-15
                ]

            lhs_str = _format_linear_terms(terms)
            if not lhs_str:
                lhs_str = "0"

            rhs = float(b[row_idx])
            lines.append(f"  {name}: {lhs_str} {sense_str} {_format_number(rhs)}")

        constraint_offset += m

    lines.append("")


def _write_bounds(lines: list[str], variables: list[Variable]) -> None:
    """Write the bounds section."""
    lines.append("Bounds")

    for v in variables:
        lb = v.lb
        ub = v.ub

        # Skip binary variables (handled in Binaries section with implicit 0/1 bounds)
        if v.domain == "binary":
            lines.append(f"  0 <= {v.name} <= 1")
            continue

        if lb is None and ub is None:
            # Free variable
            lines.append(f"  {v.name} free")
        elif lb is not None and ub is not None:
            lines.append(f"  {_format_number(lb)} <= {v.name} <= {_format_number(ub)}")
        elif lb is not None:
            lines.append(f"  {_format_number(lb)} <= {v.name}")
        else:
            # ub only
            assert ub is not None
            lines.append(f"  -inf <= {v.name} <= {_format_number(ub)}")

    lines.append("")


def _write_variable_types(lines: list[str], variables: list[Variable]) -> None:
    """Write Generals and Binaries sections."""
    generals = [v.name for v in variables if v.domain == "integer"]
    binaries = [v.name for v in variables if v.domain == "binary"]

    if generals:
        lines.append("Generals")
        for name in generals:
            lines.append(f"  {name}")
        lines.append("")

    if binaries:
        lines.append("Binaries")
        for name in binaries:
            lines.append(f"  {name}")
        lines.append("")


# =============================================================================
# Expression formatting helpers
# =============================================================================


def _extract_linear_terms(
    expr: Expression, variables: list[Variable]
) -> list[tuple[str, float]]:
    """Extract linear coefficient for each variable from a (possibly quadratic) expression.

    For quadratic expressions, this extracts only the linear part.
    """
    terms: list[tuple[str, float]] = []

    if is_linear(expr):
        for v in variables:
            coeff = extract_linear_coefficient(expr, v)
            if abs(coeff) > 1e-15:
                terms.append((v.name, coeff))
    else:
        # Quadratic expression — extract linear terms by walking the tree
        linear_coeffs = _extract_linear_from_quadratic(expr, variables)
        for v, coeff in zip(variables, linear_coeffs):
            if abs(coeff) > 1e-15:
                terms.append((v.name, coeff))

    return terms


def _extract_quadratic_terms(
    expr: Expression, variables: list[Variable]
) -> list[tuple[str, str, float]]:
    """Extract quadratic terms as (var_i, var_j, coeff) triples.

    Returns terms where i <= j (upper triangle), with coefficients
    adjusted for LP format conventions:
    - Diagonal: x_i^2 has coefficient as-is
    - Off-diagonal: x_i * x_j has coefficient (summed for both orderings)
    """
    n = len(variables)
    var_index = {v.name: i for i, v in enumerate(variables)}
    Q = np.zeros((n, n), dtype=np.float64)

    _collect_quadratic_coefficients(expr, var_index, Q, 1.0)

    # Convert Q matrix to list of (var_i, var_j, coeff) with i <= j
    terms: list[tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i, n):
            if i == j:
                coeff = Q[i, j]
            else:
                # Combine symmetric entries
                coeff = Q[i, j] + Q[j, i]
            if abs(coeff) > 1e-15:
                terms.append((variables[i].name, variables[j].name, coeff))

    return terms


def _collect_quadratic_coefficients(
    expr: Expression,
    var_index: dict[str, int],
    Q: np.ndarray,
    multiplier: float,
) -> None:
    """Recursively collect quadratic coefficients into matrix Q.

    Walks the expression tree accounting for + - * and known quadratic patterns.
    """
    from optyx.core.matrices import QuadraticForm
    from optyx.core.vectors import (
        DotProduct,
        VectorVariable,
        VectorPowerSum,
    )

    # Constant leaves and variables have no quadratic contribution.
    if isinstance(expr, (Constant, Parameter, Variable)):
        return

    # QuadraticForm: x' Q x
    if isinstance(expr, QuadraticForm):
        if isinstance(expr.vector, VectorVariable):
            qvars = expr.vector._variables
            matrix = expr.matrix
            for i, vi in enumerate(qvars):
                idx_i = var_index.get(vi.name)
                if idx_i is None:
                    continue
                for j, vj in enumerate(qvars):
                    idx_j = var_index.get(vj.name)
                    if idx_j is None:
                        continue
                    Q[idx_i, idx_j] += multiplier * matrix[i, j]
        return

    # DotProduct: x.dot(x) = sum(x_i^2) when left is right
    if isinstance(expr, DotProduct):
        if (
            isinstance(expr.left, VectorVariable)
            and isinstance(expr.right, VectorVariable)
            and expr.left is expr.right
        ):
            for v in expr.left._variables:
                idx = var_index.get(v.name)
                if idx is not None:
                    Q[idx, idx] += multiplier
        return

    # VectorPowerSum: sum(x ** 2)
    if isinstance(expr, VectorPowerSum):
        if expr.power == 2.0 and isinstance(expr.vector, VectorVariable):
            for v in expr.vector._variables:
                idx = var_index.get(v.name)
                if idx is not None:
                    Q[idx, idx] += multiplier
        return

    # BinaryOp
    if isinstance(expr, BinaryOp):
        if expr.op == "+":
            _collect_quadratic_coefficients(expr.left, var_index, Q, multiplier)
            _collect_quadratic_coefficients(expr.right, var_index, Q, multiplier)
            return
        if expr.op == "-":
            _collect_quadratic_coefficients(expr.left, var_index, Q, multiplier)
            _collect_quadratic_coefficients(expr.right, var_index, Q, -multiplier)
            return
        if expr.op == "*":
            # scalar * quadratic_expr or quadratic_expr * scalar
            left_value = _get_scalar_constant_value(expr.left)
            if left_value is not None:
                _collect_quadratic_coefficients(
                    expr.right, var_index, Q, multiplier * left_value
                )
                return
            right_value = _get_scalar_constant_value(expr.right)
            if right_value is not None:
                _collect_quadratic_coefficients(
                    expr.left, var_index, Q, multiplier * right_value
                )
                return
            # var * var — quadratic term
            left_vars = expr.left.get_variables()
            right_vars = expr.right.get_variables()
            if len(left_vars) == 1 and len(right_vars) == 1:
                lv = next(iter(left_vars))
                rv = next(iter(right_vars))
                li = var_index.get(lv.name)
                ri = var_index.get(rv.name)
                if li is not None and ri is not None:
                    # Handle coefficient*var * coefficient*var
                    lc = _get_scalar_linear_coeff(expr.left, lv)
                    rc = _get_scalar_linear_coeff(expr.right, rv)
                    Q[li, ri] += multiplier * lc * rc
            return
        if expr.op == "**":
            # var ** 2
            if isinstance(expr.right, Constant) and float(expr.right.value) == 2.0:
                left_vars = expr.left.get_variables()
                if len(left_vars) == 1:
                    lv = next(iter(left_vars))
                    li = var_index.get(lv.name)
                    if li is not None:
                        lc = _get_scalar_linear_coeff(expr.left, lv)
                        Q[li, li] += multiplier * lc * lc
            return
        if expr.op == "/":
            right_value = _get_scalar_constant_value(expr.right)
            if right_value is not None:
                _collect_quadratic_coefficients(
                    expr.left, var_index, Q, multiplier / right_value
                )
            return

    # UnaryOp
    if isinstance(expr, UnaryOp):
        if expr.op == "neg":
            _collect_quadratic_coefficients(expr.operand, var_index, Q, -multiplier)
        return

    # NarySum
    if isinstance(expr, NarySum):
        for term in expr.terms:
            _collect_quadratic_coefficients(term, var_index, Q, multiplier)
        return

    if isinstance(expr, NaryProduct):
        product_multiplier = multiplier
        variable_factors: list[Expression] = []
        for factor in expr.factors:
            value = _get_scalar_constant_value(factor)
            if value is not None:
                product_multiplier *= value
            else:
                variable_factors.append(factor)

        if len(variable_factors) == 1:
            _collect_quadratic_coefficients(
                variable_factors[0], var_index, Q, product_multiplier
            )
        elif len(variable_factors) == 2:
            left, right = variable_factors
            left_vars = left.get_variables()
            right_vars = right.get_variables()
            if len(left_vars) == 1 and len(right_vars) == 1:
                left_var = next(iter(left_vars))
                right_var = next(iter(right_vars))
                left_idx = var_index.get(left_var.name)
                right_idx = var_index.get(right_var.name)
                if left_idx is not None and right_idx is not None:
                    left_coeff = _get_scalar_linear_coeff(left, left_var)
                    right_coeff = _get_scalar_linear_coeff(right, right_var)
                    Q[left_idx, right_idx] += (
                        product_multiplier * left_coeff * right_coeff
                    )
        return


def _get_scalar_linear_coeff(expr: Expression, var: Variable) -> float:
    """Get the linear coefficient of a single-variable linear expression.

    For expressions like `3*x`, returns 3.0. For just `x`, returns 1.0.
    """
    if isinstance(expr, Variable):
        return 1.0
    if isinstance(expr, BinaryOp):
        if expr.op == "*":
            left_value = _get_scalar_constant_value(expr.left)
            if left_value is not None:
                return left_value * _get_scalar_linear_coeff(expr.right, var)
            right_value = _get_scalar_constant_value(expr.right)
            if right_value is not None:
                return _get_scalar_linear_coeff(expr.left, var) * right_value
        if expr.op == "/":
            right_value = _get_scalar_constant_value(expr.right)
            if right_value is not None:
                return _get_scalar_linear_coeff(expr.left, var) / right_value
    if isinstance(expr, NaryProduct):
        coefficient = 1.0
        for factor in expr.factors:
            value = _get_scalar_constant_value(factor)
            if value is not None:
                coefficient *= value
            else:
                coefficient *= _get_scalar_linear_coeff(factor, var)
        return coefficient
    if isinstance(expr, UnaryOp) and expr.op == "neg":
        return -_get_scalar_linear_coeff(expr.operand, var)
    return 1.0


def _extract_linear_from_quadratic(
    expr: Expression, variables: list[Variable]
) -> list[float]:
    """Extract only the linear coefficients from a quadratic expression.

    Walks the tree, ignoring quadratic terms, and collects linear contributions.
    """
    n = len(variables)
    var_index = {v.name: i for i, v in enumerate(variables)}
    result = [0.0] * n
    _collect_linear_from_quadratic(expr, var_index, result, 1.0)
    return result


def _collect_linear_from_quadratic(
    expr: Expression,
    var_index: dict[str, int],
    result: list[float],
    multiplier: float,
) -> None:
    """Recursively collect linear coefficients, skipping quadratic terms."""
    from optyx.core.matrices import QuadraticForm
    from optyx.core.vectors import (
        DotProduct,
        VectorVariable,
        VectorSum,
        LinearCombination,
        VectorPowerSum,
    )

    # Constant leaves have no linear contribution.
    if isinstance(expr, (Constant, Parameter)):
        return

    # Variable — linear contribution
    if isinstance(expr, Variable):
        idx = var_index.get(expr.name)
        if idx is not None:
            result[idx] += multiplier
        return

    # VectorSum: sum(x) — each variable coefficient is 1
    if isinstance(expr, VectorSum):
        if isinstance(expr.vector, VectorVariable):
            for v in expr.vector._variables:
                idx = var_index.get(v.name)
                if idx is not None:
                    result[idx] += multiplier
        return

    # LinearCombination: c @ x
    if isinstance(expr, LinearCombination):
        if isinstance(expr.vector, VectorVariable):
            for i, v in enumerate(expr.vector._variables):
                idx = var_index.get(v.name)
                if idx is not None:
                    result[idx] += multiplier * float(expr.coefficients[i])
        return

    # Purely quadratic — no linear contribution
    if isinstance(expr, (QuadraticForm, DotProduct, VectorPowerSum)):
        return

    # BinaryOp
    if isinstance(expr, BinaryOp):
        if expr.op == "+":
            _collect_linear_from_quadratic(expr.left, var_index, result, multiplier)
            _collect_linear_from_quadratic(expr.right, var_index, result, multiplier)
            return
        if expr.op == "-":
            _collect_linear_from_quadratic(expr.left, var_index, result, multiplier)
            _collect_linear_from_quadratic(expr.right, var_index, result, -multiplier)
            return
        if expr.op == "*":
            left_value = _get_scalar_constant_value(expr.left)
            if left_value is not None:
                _collect_linear_from_quadratic(
                    expr.right, var_index, result, multiplier * left_value
                )
                return
            right_value = _get_scalar_constant_value(expr.right)
            if right_value is not None:
                _collect_linear_from_quadratic(
                    expr.left, var_index, result, multiplier * right_value
                )
                return
            # var * var — quadratic, no linear contribution
            return
        if expr.op == "/":
            right_value = _get_scalar_constant_value(expr.right)
            if right_value is not None:
                _collect_linear_from_quadratic(
                    expr.left, var_index, result, multiplier / right_value
                )
            return
        if expr.op == "**":
            # x**2 is quadratic, skip; x**1 is handled as Variable
            if isinstance(expr.right, Constant) and float(expr.right.value) == 1.0:
                _collect_linear_from_quadratic(expr.left, var_index, result, multiplier)
            return

    # UnaryOp
    if isinstance(expr, UnaryOp):
        if expr.op == "neg":
            _collect_linear_from_quadratic(expr.operand, var_index, result, -multiplier)
        return

    # NarySum
    if isinstance(expr, NarySum):
        for term in expr.terms:
            _collect_linear_from_quadratic(term, var_index, result, multiplier)
        return

    if isinstance(expr, NaryProduct):
        product_multiplier = multiplier
        variable_factors: list[Expression] = []
        for factor in expr.factors:
            value = _get_scalar_constant_value(factor)
            if value is not None:
                product_multiplier *= value
            else:
                variable_factors.append(factor)
        if len(variable_factors) == 1:
            _collect_linear_from_quadratic(
                variable_factors[0], var_index, result, product_multiplier
            )
        return


def _extract_constant_from_quadratic(expr: Expression) -> float:
    """Extract constant term from a quadratic expression."""
    if isinstance(expr, (Constant, Parameter)):
        constant_value = _get_scalar_constant_value(expr)
        assert constant_value is not None
        return constant_value
    if isinstance(expr, Variable):
        return 0.0
    if isinstance(expr, BinaryOp):
        if expr.op == "+":
            return _extract_constant_from_quadratic(
                expr.left
            ) + _extract_constant_from_quadratic(expr.right)
        if expr.op == "-":
            return _extract_constant_from_quadratic(
                expr.left
            ) - _extract_constant_from_quadratic(expr.right)
        if expr.op == "*":
            return _extract_constant_from_quadratic(
                expr.left
            ) * _extract_constant_from_quadratic(expr.right)
        if expr.op == "/":
            right_value = _get_scalar_constant_value(expr.right)
            if right_value is not None:
                return _extract_constant_from_quadratic(expr.left) / right_value
            return 0.0
        if expr.op == "**":
            if isinstance(expr.right, Constant) and float(expr.right.value) == 0.0:
                return 1.0
            return 0.0
    if isinstance(expr, UnaryOp):
        if expr.op == "neg":
            return -_extract_constant_from_quadratic(expr.operand)
        return 0.0
    if isinstance(expr, NarySum):
        return sum(_extract_constant_from_quadratic(t) for t in expr.terms)
    if isinstance(expr, NaryProduct):
        result = 1.0
        for factor in expr.factors:
            result *= _extract_constant_from_quadratic(factor)
        return result
    return 0.0


def _format_expression(
    linear_terms: list[tuple[str, float]],
    quad_terms: list[tuple[str, str, float]],
    constant: float,
) -> str:
    """Format an expression as LP format string.

    LP format for quadratic objectives uses [ ... ] / 2 notation.
    """
    parts: list[str] = []

    # Linear terms
    linear_str = _format_linear_terms(linear_terms)
    if linear_str:
        parts.append(linear_str)

    # Quadratic terms ([ ... ] / 2 notation)
    if quad_terms:
        quad_str = _format_quadratic_section(quad_terms)
        parts.append(quad_str)

    # Constant term
    if abs(constant) > 1e-15:
        const_str = _format_number(constant)
        if parts:
            if constant > 0:
                parts.append(f"+ {const_str}")
            else:
                parts.append(f"- {_format_number(-constant)}")
        else:
            parts.append(const_str)

    if not parts:
        return "0"

    return " ".join(parts)


def _format_linear_terms(terms: list[tuple[str, float]]) -> str:
    """Format linear terms like '2 x0 + 3 x1 - x2'."""
    if not terms:
        return ""

    parts: list[str] = []
    for i, (name, coeff) in enumerate(terms):
        if i == 0:
            # First term: no leading space/sign unless negative
            if coeff == 1.0:
                parts.append(name)
            elif coeff == -1.0:
                parts.append(f"- {name}")
            elif coeff < 0:
                parts.append(f"- {_format_number(-coeff)} {name}")
            else:
                parts.append(f"{_format_number(coeff)} {name}")
        else:
            # Subsequent terms: always have + or -
            if coeff == 1.0:
                parts.append(f"+ {name}")
            elif coeff == -1.0:
                parts.append(f"- {name}")
            elif coeff < 0:
                parts.append(f"- {_format_number(-coeff)} {name}")
            else:
                parts.append(f"+ {_format_number(coeff)} {name}")

    return " ".join(parts)


def _format_quadratic_section(
    quad_terms: list[tuple[str, str, float]],
) -> str:
    """Format quadratic terms in LP format: [ 2 x0 ^2 + 4 x0 * x1 ] / 2.

    LP format convention: the quadratic section is [ Q_terms ] / 2,
    where Q_terms use doubled coefficients (so the actual contribution is halved).
    """
    parts: list[str] = []
    for i, (vi, vj, coeff) in enumerate(quad_terms):
        # Double the coefficient for LP [ ... ] / 2 convention
        doubled = 2.0 * coeff

        if vi == vj:
            term = f"{vi} ^2"
        else:
            term = f"{vi} * {vj}"

        if i == 0:
            if doubled == 1.0:
                parts.append(term)
            elif doubled == -1.0:
                parts.append(f"- {term}")
            elif doubled < 0:
                parts.append(f"- {_format_number(-doubled)} {term}")
            else:
                parts.append(f"{_format_number(doubled)} {term}")
        else:
            if doubled == 1.0:
                parts.append(f"+ {term}")
            elif doubled == -1.0:
                parts.append(f"- {term}")
            elif doubled < 0:
                parts.append(f"- {_format_number(-doubled)} {term}")
            else:
                parts.append(f"+ {_format_number(doubled)} {term}")

    return "[ " + " ".join(parts) + " ] / 2"


def _format_number(value: float) -> str:
    """Format a number for LP output, removing unnecessary trailing zeros."""
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    if value == int(value):
        return str(int(value))
    # Format with reasonable precision, strip trailing zeros
    formatted = f"{value:.10g}"
    return formatted
