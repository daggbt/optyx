"""Example: Open-Pit Mining Production Scheduling

Demonstrates a multi-period mine production scheduling problem using optyx.
This is a simplified but realistic model for open-pit mining operations.

Problem:
- Schedule extraction from multiple mining blocks over multiple periods
- Maximize NPV (Net Present Value) of extracted ore
- Subject to:
  - Processing plant capacity limits
  - Mining equipment capacity limits
  - Grade blending constraints (min/max ore grade)
  - Precedence constraints (can't mine lower blocks before upper blocks)
"""

import numpy as np
from optyx import Variable, Problem

print("=" * 70)
print("OPTYX - Mining Production Scheduling Demo")
print("=" * 70)

# =============================================================================
# Problem Data
# =============================================================================
print("\n📊 Problem Setup")
print("-" * 50)

# Time periods (e.g., years)
n_periods = 4
discount_rate = 0.10  # 10% annual discount rate

# Mining blocks (simplified 2D pit with 3 levels, 3 columns = 9 blocks)
# Block layout (cross-section view):
#   [0] [1] [2]    <- Level 0 (surface)
#     [3] [4]      <- Level 1 (must mine 0,1 or 1,2 first)
#       [5]        <- Level 2 (must mine 3,4 first)
#   [6] [7] [8]    <- Level 0 (surface, separate area)

n_blocks = 9

# Block properties: tonnage (kt), grade (% copper), mining cost ($/t)
block_tonnage = np.array([100, 120, 110, 80, 90, 60, 95, 105, 100])  # kt
block_grade = np.array([0.8, 1.2, 0.9, 1.5, 2.0, 2.5, 0.7, 1.1, 0.85])  # % Cu
block_mining_cost = np.array([3.5, 3.8, 3.6, 4.2, 4.5, 5.0, 3.4, 3.7, 3.5])  # $/t

# Economic parameters
copper_price = 8000  # $/tonne Cu
processing_cost = 15  # $/t ore
recovery_rate = 0.90  # 90% Cu recovery

# Capacity constraints
max_mining_capacity = 250  # kt/period (mining equipment limit)
max_processing_capacity = 200  # kt/period (plant capacity)
min_blend_grade = 1.0  # Minimum average grade to process (% Cu)
max_blend_grade = 2.0  # Maximum average grade (avoid overloading plant)

# Precedence: block i must be mined before block j
# (upper blocks before lower blocks)
precedence = [
    (0, 3),
    (1, 3),  # Blocks 0,1 must precede block 3
    (1, 4),
    (2, 4),  # Blocks 1,2 must precede block 4
    (3, 5),
    (4, 5),  # Blocks 3,4 must precede block 5
]

print(f"Periods: {n_periods} years")
print(f"Blocks: {n_blocks}")
print(f"Total ore: {block_tonnage.sum():.0f} kt")
print(f"Average grade: {np.average(block_grade, weights=block_tonnage):.2f}% Cu")
print(f"Mining capacity: {max_mining_capacity} kt/period")
print(f"Processing capacity: {max_processing_capacity} kt/period")

# =============================================================================
# Decision Variables
# =============================================================================
print("\n🔧 Creating Decision Variables")
print("-" * 50)

# x[i,t] = fraction of block i mined in period t (0 to 1)
# Using continuous relaxation (would be binary in full MILP)
x = {}
for i in range(n_blocks):
    for t in range(n_periods):
        x[i, t] = Variable(f"x_{i}_{t}", lb=0, ub=1)

print(f"Variables created: {len(x)} (blocks × periods)")

# =============================================================================
# Objective: Maximize NPV
# =============================================================================
print("\n💰 Building Objective Function")
print("-" * 50)

# NPV = Σ (revenue - cost) * discount_factor
# Revenue = tonnage × grade × recovery × price
# Cost = tonnage × (mining_cost + processing_cost)

npv = 0
for t in range(n_periods):
    discount_factor = 1 / (1 + discount_rate) ** t

    for i in range(n_blocks):
        tonnage = block_tonnage[i]
        grade = block_grade[i] / 100  # Convert to decimal

        # Revenue from copper ($ thousands since tonnage in kt)
        revenue = tonnage * 1000 * grade * recovery_rate * copper_price / 1000

        # Costs ($ thousands)
        cost = tonnage * 1000 * (block_mining_cost[i] + processing_cost) / 1000

        # NPV contribution
        npv = npv + (revenue - cost) * discount_factor * x[i, t]

print(f"NPV expression built with {n_blocks * n_periods} terms")

# =============================================================================
# Constraints
# =============================================================================
print("\n📋 Adding Constraints")
print("-" * 50)

prob = Problem(name="mine_scheduling")
prob.maximize(npv)

# 1. Each block can only be mined once (sum over periods ≤ 1)
for i in range(n_blocks):
    total_extraction = sum(x[i, t] for t in range(n_periods))
    prob.subject_to(total_extraction <= 1)
print(f"  ✓ {n_blocks} block extraction limits (each block mined at most once)")

# 2. Mining capacity per period
for t in range(n_periods):
    period_mining = sum(block_tonnage[i] * x[i, t] for i in range(n_blocks))
    prob.subject_to(period_mining <= max_mining_capacity)
print(f"  ✓ {n_periods} mining capacity constraints ({max_mining_capacity} kt/period)")

# 3. Processing capacity per period
for t in range(n_periods):
    period_processing = sum(block_tonnage[i] * x[i, t] for i in range(n_blocks))
    prob.subject_to(period_processing <= max_processing_capacity)
print(
    f"  ✓ {n_periods} processing capacity constraints ({max_processing_capacity} kt/period)"
)

# 4. Precedence constraints (cumulative: can't mine j until i is done)
for i, j in precedence:
    for t in range(n_periods):
        # Cumulative extraction of i up to period t must be ≥ cumulative of j
        cum_i = sum(x[i, s] for s in range(t + 1))
        cum_j = sum(x[j, s] for s in range(t + 1))
        prob.subject_to(cum_i >= cum_j)
print(f"  ✓ {len(precedence) * n_periods} precedence constraints")

# 5. Grade blending constraints (linearized)
# We need: min_grade ≤ (Σ tonnage × grade × x) / (Σ tonnage × x) ≤ max_grade
# Linearize: Σ tonnage × (grade - min_grade) × x ≥ 0
#           Σ tonnage × (max_grade - grade) × x ≥ 0
for t in range(n_periods):
    # Minimum grade constraint
    min_grade_expr = sum(
        block_tonnage[i] * (block_grade[i] - min_blend_grade) * x[i, t]
        for i in range(n_blocks)
    )
    prob.subject_to(min_grade_expr >= 0)

    # Maximum grade constraint
    max_grade_expr = sum(
        block_tonnage[i] * (max_blend_grade - block_grade[i]) * x[i, t]
        for i in range(n_blocks)
    )
    prob.subject_to(max_grade_expr >= 0)
print(
    f"  ✓ {n_periods * 2} grade blending constraints ({min_blend_grade}-{max_blend_grade}% Cu)"
)

total_constraints = (
    n_blocks  # extraction limits
    + n_periods  # mining capacity
    + n_periods  # processing capacity
    + len(precedence) * n_periods  # precedence
    + n_periods * 2  # grade blending
)
print(f"\nTotal constraints: {total_constraints}")

# =============================================================================
# Solve
# =============================================================================
print("\n🚀 Solving...")
print("-" * 50)

solution = prob.solve(method="trust-constr")

print(f"Status: {solution.status.value}")
print(f"NPV: ${solution.objective_value:,.0f} thousand")
print(f"Iterations: {solution.iterations}")
if solution.solve_time >= 1.0:
    print(f"Solve time: {solution.solve_time:.2f} s")
else:
    print(f"Solve time: {solution.solve_time * 1000:.1f} ms")

# =============================================================================
# Results Visualization
# =============================================================================
print("\n📊 Extraction Schedule")
print("-" * 50)

# Print schedule table
print(f"\n{'Block':<8}", end="")
for t in range(n_periods):
    print(f"{'Period ' + str(t + 1):>10}", end="")
print(f"{'Total':>10}")
print("-" * (8 + 10 * (n_periods + 1)))

total_by_period = [0] * n_periods
for i in range(n_blocks):
    print(f"Block {i:<3}", end="")
    block_total = 0
    for t in range(n_periods):
        val = solution[f"x_{i}_{t}"]
        print(f"{val:>10.2f}", end="")
        block_total += val
        total_by_period[t] += val * block_tonnage[i]
    print(f"{block_total:>10.2f}")

print("-" * (8 + 10 * (n_periods + 1)))
print(f"{'Tonnage':<8}", end="")
for t in range(n_periods):
    print(f"{total_by_period[t]:>10.0f}", end="")
print(f"{sum(total_by_period):>10.0f} kt")

# Grade by period
print(f"\n{'Grade':<8}", end="")
for t in range(n_periods):
    weighted_grade = sum(
        block_grade[i] * block_tonnage[i] * solution[f"x_{i}_{t}"]
        for i in range(n_blocks)
    )
    if total_by_period[t] > 0:
        avg_grade = weighted_grade / total_by_period[t]
        print(f"{avg_grade:>9.2f}%", end="")
    else:
        print(f"{'N/A':>10}", end="")
print()

# =============================================================================
# Economic Summary
# =============================================================================
print("\n💵 Economic Summary")
print("-" * 50)

total_tonnage = sum(total_by_period)
total_cu = sum(
    block_grade[i]
    / 100
    * block_tonnage[i]
    * sum(solution[f"x_{i}_{t}"] for t in range(n_periods))
    for i in range(n_blocks)
)

print(f"Total ore mined: {total_tonnage:,.0f} kt")
print(f"Total copper content: {total_cu:,.1f} kt Cu")
print(f"Average grade: {total_cu / total_tonnage * 100:.2f}% Cu")
print(f"NPV: ${solution.objective_value:,.0f} thousand")
print(f"NPV per tonne ore: ${solution.objective_value / total_tonnage:.2f}/t")

print("\n" + "=" * 70)
print("Demo complete! This model can be extended with:")
print("  • Binary variables for discrete block extraction (MILP)")
print("  • More realistic pit geometry and bench constraints")
print("  • Stockpile management and blending")
print("  • Equipment scheduling and haul routes")
print("  • Stochastic prices and geological uncertainty")
print("=" * 70)
