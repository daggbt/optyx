"""LP format export.

Demonstrates how to export optimization models to the standard LP file
format using Problem.write() and Problem.to_lp().
"""

from optyx import (
    BinaryVariable,
    IntegerVariable,
    Problem,
    Variable,
    VectorVariable,
    quadratic_form,
)
import numpy as np
import os

# --- Example 1: Simple LP ---
print("=== Example 1: Simple linear program ===\n")

x = Variable("x", lb=0)
y = Variable("y", lb=0)

prob = Problem("simple_lp")
prob.minimize(2 * x + 3 * y)
prob.subject_to(x + y >= 1)
prob.subject_to(x - y <= 5)

print(prob.to_lp())


# --- Example 2: Quadratic objective ---
print("=== Example 2: Quadratic program ===\n")

x0 = Variable("x0", lb=0)
x1 = Variable("x1", lb=0)
x2 = Variable("x2", lb=0)

prob2 = Problem("dense_qp")
prob2.minimize(x0 + x1 + x0**2 + x0 * x1 + x1**2 + x1 * x2 + x2**2)
prob2.subject_to(x0 + 2 * x1 + 3 * x2 >= 4)
prob2.subject_to(x0 + x1 >= 1)

print(prob2.to_lp())


# --- Example 3: Portfolio optimization (QuadraticForm) ---
print("=== Example 3: Portfolio optimization ===\n")

n_assets = 4
w = VectorVariable("w", n_assets, lb=0, ub=1)

# Covariance matrix
cov = np.array([
    [0.04, 0.006, 0.002, 0.001],
    [0.006, 0.09, 0.004, 0.003],
    [0.002, 0.004, 0.01, 0.002],
    [0.001, 0.003, 0.002, 0.06],
])
expected_return = np.array([0.10, 0.15, 0.08, 0.12])

prob3 = Problem("portfolio")
prob3.minimize(quadratic_form(w, cov))
prob3.subject_to(w.sum().eq(1))
prob3.subject_to(expected_return @ w >= 0.10)

print(prob3.to_lp())


# --- Example 4: Mixed-integer program ---
print("=== Example 4: Mixed-integer program ===\n")

# Knapsack-like problem
items = ["A", "B", "C", "D"]
from optyx import VariableDict

select = VariableDict("select", items, domain="binary")
quantity = VariableDict("qty", items, lb=0, ub=10, domain="integer")

weights = {"A": 3, "B": 5, "C": 2, "D": 7}
values = {"A": 4, "B": 7, "C": 3, "D": 9}

prob4 = Problem("knapsack_mip")
prob4.maximize(sum(values[i] * select[i] for i in items))
prob4.subject_to(sum(weights[i] * select[i] for i in items) <= 12)

print(prob4.to_lp())


# --- Example 5: Write to file ---
print("=== Example 5: Writing to file ===\n")

_dir = os.path.dirname(os.path.abspath(__file__))

prob.write(os.path.join(_dir, "simple_lp.lp"))
print("Wrote simple LP to examples/simple_lp.lp")

prob3.write(os.path.join(_dir, "portfolio.lp"))
print("Wrote portfolio QP to examples/portfolio.lp")

prob4.write(os.path.join(_dir, "knapsack.lp"))
print("Wrote knapsack MIP to examples/knapsack.lp")

print("\nDone!")
