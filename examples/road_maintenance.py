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
from optyx import Variable, Problem, exp
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
traffic = [50000, 12000, 85000, 5000, 15000]  # Daily vehicles
current_pci = [45, 60, 35, 70, 55]  # Current condition (poor to fair)
decay_rate = [5, 3, 8, 2, 4]  # Expected drop in PCI next year

total_budget = 10.0  # $10M available
max_per_road = 5.0  # Max $5M per road (practical constraint)

# =============================================================================
# 2. Define Decision Variables
# =============================================================================
# x[i] = Budget allocated to road i (in $M)
# Use underscores in variable names for consistency
road_ids = ["hwy101", "route66", "i95", "main_st", "broadway"]
vars = []
for rid in road_ids:
    vars.append(Variable(f"budget_{rid}", lb=0.0, ub=max_per_road))

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
total_utility = 0
print("\n📊 Building Symbolic Model...")

for i in range(len(roads)):
    # 1. Predict future condition
    # Future = Current - Decay + Repair
    future_pci = current_pci[i] - decay_rate[i] + repair_model(vars[i])

    # 2. Calculate user satisfaction (0-100)
    user_sat = satisfaction_model(future_pci)

    # 3. Weighted by traffic (Maximize total happy drivers)
    total_utility += traffic[i] * user_sat

# =============================================================================
# 4. THE SELLING POINT: Show Auto-Diff in Action
# =============================================================================
print("\n🔍 Why Auto-Diff Matters:")
print("-" * 50)
print("The objective involves nested non-linearities:")
print("  Utility = Σ Traffic × Sigmoid( PCI₀ - Decay + 40×(1 - e^(-0.5×Budget)) )")
print("\nManually deriving ∂Utility/∂Budget requires the chain rule through")
print("both the Sigmoid and Exponential. Optyx does this automatically.")

# Compile the gradient function and evaluate at a test point
grad_fn = compile_gradient(total_utility, vars)
test_point = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # $1M each
grad_values = grad_fn(test_point)

print("\n📐 Gradient at uniform $1M allocation:")
print("   (Higher = more marginal utility from additional spending)")
for i, r in enumerate(roads):
    print(f"   ∂U/∂{r:<10}: {grad_values[i]:>10.0f}")
print("-" * 50)

# =============================================================================
# 5. Solve
# =============================================================================
print("\n🚀 Solving Optimization Problem...")

# Add penalty for unspent budget to encourage full allocation
# Utility = Original Utility - Penalty * (Budget - Spent)^2
budget_sum = vars[0]
for v in vars[1:]:
    budget_sum = budget_sum + v

# Large penalty for not using all budget encourages full allocation
penalty = 1e6 * (budget_sum - total_budget) ** 2
penalized_utility = total_utility - penalty

prob = Problem().maximize(penalized_utility)

# Also add hard upper bound
prob.subject_to(budget_sum <= total_budget)

sol = prob.solve()

# =============================================================================
# 6. Results
# =============================================================================
print("\n✅ Optimal Budget Allocation:")
print("-" * 70)
print(
    f"{'Road':<12} {'Traffic':>10} {'Old PCI':>10} {'Budget':>10} {'New PCI':>10} {'Satisf%':>10}"
)
print("-" * 70)

total_spent = 0
for i, r in enumerate(roads):
    spend = sol[vars[i]]
    total_spent += spend

    # Re-calculate final state values
    pci_gain = 40.0 * (1 - np.exp(-0.5 * spend))
    final_pci = current_pci[i] - decay_rate[i] + pci_gain
    sat = 100.0 / (1 + np.exp(-0.12 * (final_pci - 50)))

    print(
        f"{r:<12} {traffic[i]:>10,} {current_pci[i]:>10} ${spend:>9.2f} {final_pci:>10.1f} {sat:>10.1f}"
    )

print("-" * 70)
print(f"Total Spent: ${total_spent:.2f}M / ${total_budget:.2f}M")
print(f"Network Utility Score: {sol.objective_value:,.0f}")

print("""
======================================================================
Demo complete! Key insights:
  • Roads with high traffic AND improvable PCI get priority.
  • Roads already in good condition (Main St) get less because they're
    in the 'flat' top of the S-curve (diminishing satisfaction gains).
  • Roads in very poor condition may need multi-year investment plans.
  
This model can be extended with:
  • Multi-year planning horizons
  • Different repair action types (patching vs. resurfacing)
  • Stochastic deterioration models
  • Network connectivity / criticality weights
======================================================================
""")
