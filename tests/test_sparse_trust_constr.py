"""Tests for sparse trust-constr constraint wiring."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from optyx import Problem, Variable, VectorVariable, as_matrix
from optyx.solvers.scipy_solver import (
    _build_solver_cache,
    _build_trust_constr_constraints,
)


def _build_sparse_chain_problem(n: int = 6) -> tuple[Problem, np.ndarray]:
    """Build a sparse nonlinear problem with mixed constraint senses."""
    variables = [Variable(f"x{i}", lb=0.0, ub=2.0) for i in range(n)]

    prob = Problem(name=f"sparse_chain_{n}")
    prob.minimize(sum((var - 1.0) ** 2 for var in variables))

    for i in range(n - 1):
        prob.subject_to(variables[i + 1] - variables[i] >= 0)
    prob.subject_to(variables[0] <= 0.5)
    prob.subject_to(variables[-1].eq(1.0))

    x0 = np.linspace(0.2, 1.0, n)
    return prob, x0


def test_trust_constr_helper_builds_sparse_mixed_sense_constraint():
    """Helper should build a mixed-sense NonlinearConstraint with CSR Jacobian."""
    prob, x0 = _build_sparse_chain_problem(n=6)
    cache = _build_solver_cache(prob, prob.variables)

    assert "sparse_constraint_jac_fn" not in cache

    constraint = _build_trust_constr_constraints(cache)
    assert constraint is not None
    assert "sparse_constraint_jac_fn" in cache

    jac = constraint.jac(x0)
    assert sparse.isspmatrix_csr(jac)

    expected_lb = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -np.inf, 0.0])
    expected_ub = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, 0.0, 0.0])
    np.testing.assert_allclose(constraint.lb, expected_lb)
    np.testing.assert_allclose(constraint.ub, expected_ub)
    assert constraint.fun(x0).shape == (7,)


def test_solver_cache_uses_dense_output_sparse_eval_for_sparse_objective():
    """Sparse objective gradients should compute sparsely but return dense arrays."""
    x = Variable("x", lb=0.0, ub=2.0)
    y = Variable("y", lb=0.0, ub=2.0)
    z = Variable("z", lb=0.0, ub=2.0)

    prob = Problem(name="sparse_objective_grad")
    prob.minimize((x - 1.0) ** 2)
    prob.subject_to(y >= 0.25)
    prob.subject_to(z <= 1.75)

    cache = _build_solver_cache(prob, prob.variables)
    grad = cache["grad_fn"](np.array([1.5, 0.3, 0.8]))

    assert isinstance(grad, np.ndarray)
    expected = np.array([1.0 if var.name == "x" else 0.0 for var in prob.variables])
    np.testing.assert_allclose(grad, expected)


@pytest.mark.filterwarnings("ignore:delta_grad == 0.0:UserWarning")
def test_trust_constr_honors_sparse_matrix_constraints():
    """trust-constr should honor sparse matrix blocks on the SciPy path."""
    n = 8
    x = VectorVariable("x", n, lb=0.0, ub=2.0)
    A = as_matrix(sparse.eye(n, format="csr"), storage="sparse")

    prob = Problem(name="sparse_matrix_trust_constr")
    prob.minimize(sum((x[i] - 1.0) ** 2 for i in range(n)))
    prob.subject_to(A @ x >= np.full(n, 0.25))

    sol = prob.solve(method="trust-constr")

    assert sol.is_optimal
    values = np.array([sol[f"x[{i}]"] for i in range(n)])
    assert np.all(values >= 0.25 - 1e-5)


def test_slsqp_honors_matrix_constraints_with_variable_reordering():
    """Matrix blocks should be aligned to the global variable order for SciPy."""
    x = VectorVariable("x", 2, lb=0.0, ub=2.0)
    a = Variable("a", lb=0.0, ub=2.0)

    prob = Problem(name="matrix_reordering")
    prob.minimize((x[0] - 0.2) ** 2 + (x[1] - 0.8) ** 2 + (a - 0.5) ** 2)
    prob.subject_to(
        as_matrix(np.array([[1.0, 1.0]]), storage="dense") @ x >= np.array([1.0])
    )

    sol = prob.solve(method="SLSQP")

    assert sol.is_optimal
    assert sol["a"] == pytest.approx(0.5, abs=1e-4)
    assert sol["x[0]"] + sol["x[1]"] >= 1.0 - 1e-5


def test_auto_select_prefers_trust_constr_for_large_sparse_matrix_nlp():
    """Large sparse matrix-constrained NLPs should prefer trust-constr."""
    n = 64
    x = VectorVariable("x", n, lb=0.0, ub=2.0)

    prob = Problem(name="auto_sparse_matrix_nlp")
    prob.minimize(x.dot(x))
    prob.subject_to(as_matrix(np.eye(n), storage="auto") @ x >= np.zeros(n))

    assert prob._auto_select_method() == "trust-constr"


@pytest.mark.filterwarnings("ignore:delta_grad == 0.0:UserWarning")
def test_sparse_chain_problem_solves_with_trust_constr():
    """trust-constr should solve sparse chain problems through the new path."""
    prob, x0 = _build_sparse_chain_problem(n=8)

    sol = prob.solve(method="trust-constr", x0=x0)

    assert sol.is_optimal
    values = np.array([sol[f"x{i}"] for i in range(8)])
    assert np.all(np.diff(values) >= -1e-6)
    assert values[0] <= 0.5 + 1e-6
    assert values[-1] == pytest.approx(1.0, abs=1e-3)
