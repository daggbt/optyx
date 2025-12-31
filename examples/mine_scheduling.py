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
from optyx import MatrixVariable, Problem

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
x = MatrixVariable("x", rows=n_blocks, cols=n_periods, lb=0, ub=1)

print(f"Variables created: {x.size} (blocks × periods)")

# =============================================================================
# Objective: Maximize NPV
# =============================================================================
print("\n💰 Building Objective Function")
print("-" * 50)

# NPV = Σ (revenue - cost) * discount_factor
# Revenue = tonnage × grade × recovery × price
# Cost = tonnage × (mining_cost + processing_cost)

# Calculate profit per block ($ thousands)
# Revenue = tonnage * grade * recovery * price
revenue_per_block = block_tonnage * (block_grade / 100) * recovery_rate * copper_price
# Cost = tonnage * (mining_cost + processing_cost)
cost_per_block = block_tonnage * (block_mining_cost + processing_cost)
profit_per_block = revenue_per_block - cost_per_block

# Discount factors for each period
discount_factors = np.array([1 / (1 + discount_rate) ** t for t in range(n_periods)])

# NPV = sum(profit[i] * x[i,t] * discount[t])
npv = 0
for t in range(n_periods):
    # Vectorized over blocks for each period: profit @ x_column
    period_profit = profit_per_block @ x[:, t]
    npv += period_profit * discount_factors[t]

print("NPV objective built with vectorized operations")

# =============================================================================
# Constraints
# =============================================================================
print("\n🚧 Adding Constraints")
print("-" * 50)

prob = Problem(name="mine_scheduling")
prob.maximize(npv)

# 1. Each block mined at most once
# Sum across columns (time) for each row (block) must be <= 1
for i in range(n_blocks):
    # x[i, :] returns a VectorVariable representing the row
    prob.subject_to(x[i, :].sum() <= 1)

# 2. Mining capacity per period
for t in range(n_periods):
    # Vectorized dot product: tonnage @ x_column
    period_mining = block_tonnage @ x[:, t]
    prob.subject_to(period_mining <= max_mining_capacity)

# 3. Processing capacity per period
for t in range(n_periods):
    period_processing = block_tonnage @ x[:, t]
    prob.subject_to(period_processing <= max_processing_capacity)

# 4. Precedence constraints
for i, j in precedence:
    for t in range(n_periods):
        # Cumulative extraction up to time t
        # We use slice indexing x[i, :t+1] which returns a VectorVariable
        cum_i = x[i, : t + 1].sum()
        cum_j = x[j, : t + 1].sum()
        prob.subject_to(cum_i >= cum_j)

# 5. Grade blending constraints
for t in range(n_periods):
    # Minimum grade: sum(tonnage * (grade - min) * x) >= 0
    min_grade_coefs = block_tonnage * (block_grade - min_blend_grade)
    prob.subject_to(min_grade_coefs @ x[:, t] >= 0)

    # Maximum grade: sum(tonnage * (max - grade) * x) >= 0
    max_grade_coefs = block_tonnage * (max_blend_grade - block_grade)
    prob.subject_to(max_grade_coefs @ x[:, t] >= 0)

total_constraints = (
    n_blocks  # extraction limits
    + n_periods * 2  # capacity constraints
    + len(precedence) * n_periods  # precedence
    + n_periods * 2  # grade blending
)
print(f"Total constraints: {total_constraints}")

# =============================================================================
# Solve
# =============================================================================
print("\n🚀 Solving...")
print("-" * 50)

# Auto-detects that this is a Linear Programming (LP) problem
# and uses the fast HiGHS solver via scipy.optimize.linprog
solution = prob.solve()

print(f"Status: {solution.status.value}")
print(f"NPV: ${solution.objective_value:,.0f} thousand")
print(f"Iterations: {solution.iterations}")
print(f"Solve time: {solution.solve_time*1000:.1f} ms")
