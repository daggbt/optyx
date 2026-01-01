"""Example: Road Infrastructure Maintenance Optimization

Demonstrates optimizing budget allocation for a road network using non-linear
deterioration and satisfaction models.

Problem:
- We manage a network of road segments with different traffic volumes.
- Roads deteriorate over time (decay).
- We have a limited maintenance budget that must be fully allocated.
- Spending money improves condition, but with diminishing returns (Exponential).
- User satisfaction depends on condition non-linearly (Sigmoid/S-curve).

Why Optyx?
- The objective function combines Exponentials (repair physics) and
  Sigmoids (user satisfaction).
- The chain rule for the derivatives is complex:
  d(Satisfaction)/d(Budget) = d(Sat)/d(PCI) * d(PCI)/d(Budget)
- Optyx handles this symbolic complexity automatically, providing the exact
  gradients and Hessians to the solver for robust convergence.
"""

import numpy as np
from optyx import VectorVariable, Problem, exp
from optyx.core.compiler import compile_gradient

print("=" * 70)
print("OPTYX - Road Asset Management Demo")
print("=" * 70)

# =============================================================================
# 1. Problem Data
# =============================================================================
# 5 Road Segments
# PCI = Pavement Condition Index (0-100)
roads = ["Hwy 101", "Route 66", "I-95", "Main St", "Broadway"]
traffic = np.array([50000, 12000, 85000, 5000, 15000])  # Daily vehicles
current_pci = np.array([45, 60, 35, 70, 55])  # Current condition
decay_rate = np.array([5, 3, 8, 2, 4])  # Annual deterioration

total_budget = 10.0  # $10M available
max_per_road = 5.0  # Max $5M per road (practical constraint)

print("Road Network Status:")
print("-" * 50)
for i, road in enumerate(roads):
    print(
        f"{road:<12} Traffic: {traffic[i]:>6,}  PCI: {current_pci[i]:>3}  Decay: {decay_rate[i]}"
    )

# =============================================================================
# 2. Define Decision Variables
# =============================================================================
# x[i] = Budget allocated to road i (in $M)
budget = VectorVariable("budget", len(roads), lb=0.0, ub=max_per_road)

# =============================================================================
# 3. Define Non-Linear Models
# =============================================================================


def repair_model(spend):
    """
    Physics Model: Diminishing returns on investment.
    Spending $1M improves PCI, but spending $2M doesn't double the improvement.
    Formula: Gain = MaxGain * (1 - exp(-k * spend))
    """
    max_gain = 40.0  # Max possible PCI increase
    efficiency = 0.5  # How fast we reach max gain
    return max_gain * (1 - exp(-efficiency * spend))


def satisfaction_model(pci):
    """
    Social Model: User satisfaction is an S-curve (Sigmoid).
    - Below PCI 40: Everyone is unhappy (near 0%)
    - Above PCI 70: Everyone is happy (near 100%)
    - The transition is steep around PCI 50-60.
    """
    # Sigmoid: 1 / (1 + exp(-k * (x - center)))
    return 100.0 / (1 + exp(-0.12 * (pci - 50)))


# Build the objective expression
print("\n📊 Building Symbolic Model...")

# Build objective: maximize total weighted satisfaction
# Vectorized operations:
future_pci = current_pci - decay_rate + repair_model(budget)
user_sat = satisfaction_model(future_pci)
total_utility = traffic @ user_sat

print("✓ Symbolic model built with vectorized operations")

# =============================================================================
# 4. THE SELLING POINT: Show Auto-Diff in Action
# =============================================================================
print("\n🔍 Why Auto-Diff Matters:")
print("-" * 50)
print("The objective involves nested non-linearities:")

# Compile the gradient function
grad_fn = compile_gradient(total_utility, budget)

# Evaluate at uniform $1M allocation
test_point = np.ones(len(roads))
grad_values = grad_fn(test_point)

print("Marginal Utility at $1M uniform allocation:")
print("-" * 45)
for i, road in enumerate(roads):
    print(f"∂U/∂{road:<10}: {grad_values[i]:>12,.0f}")
print("-" * 45)
print("Higher gradient = more value from additional spending")

# =============================================================================
# 5. Solve
# =============================================================================

# Create and solve problem
# We rely on the constraint to enforce the budget limit.
# The objective (utility) is increasing, so the solver will naturally use the full budget.
prob = Problem().maximize(total_utility)
prob.subject_to(budget.sum() <= total_budget)

sol = prob.solve()  # method="trust-constr")

print(f"Solver status: {sol.status.value}")
print(f"Solve time: {sol.solve_time*1000:.1f} ms")

# =============================================================================
# 6. Results
# =============================================================================

print("\nOptimal Budget Allocation:")
print("-" * 70)
print(
    f"{'Road':<12} {'Traffic':>10} {'Old PCI':>10} {'Budget':>10} {'New PCI':>10} {'Satisf%':>10}"
)
print("-" * 70)

# Get all budget values at once
budget_vals = sol[budget]

total_spent = 0
for i, road in enumerate(roads):
    spend = budget_vals[i]
    total_spent += spend

    pci_gain = 40.0 * (1 - np.exp(-0.5 * spend))
    final_pci = current_pci[i] - decay_rate[i] + pci_gain
    sat = 100.0 / (1 + np.exp(-0.12 * (final_pci - 50)))

    print(
        f"{road:<12} {traffic[i]:>10,} {current_pci[i]:>10} ${spend:>9.2f} {final_pci:>10.1f} {sat:>10.1f}"
    )

print("-" * 70)
print(f"Total Spent: ${total_spent:.2f}M / ${total_budget:.2f}M")
