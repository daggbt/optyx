"""Example: Mine Production Planning with VariableDict

Demonstrates VariableDict for a mine production planning problem where
decision variables are naturally indexed by pit/zone names rather than
integer indices.

Problem:
- A mining operation has multiple ore zones, each with different grades,
  extraction costs, and tonnage limits.
- Plan extraction quantities to maximize profit while meeting:
  - Mill blend grade targets (min/max acceptable feed grade)
  - Mill throughput capacity
  - Zone-specific extraction limits
  - Minimum extraction commitments (keep equipment utilized)
  - Stockpile balance: high-grade and low-grade zones tracked separately

This showcases VariableDict's key methods:
  - Creation with per-key bounds
  - __getitem__ for individual variable access
  - sum() and sum(subset) for total and partial sums
  - prod(dict) for weighted sums (grade blending, cost calculation)
  - keys(), values(), items() for iteration
  - Solution[VariableDict] for result extraction
"""

from optyx import VariableDict, Problem

print("=" * 70)
print("OPTYX - Mine Production Planning with VariableDict")
print("=" * 70)

# =============================================================================
# Problem Data — Ore Zones
# =============================================================================
zones = ["North Pit", "South Pit", "East Bench", "West Cutback", "Stockpile"]

# Ore grade (% copper) per zone
grade = {
    "North Pit": 0.85,
    "South Pit": 0.62,
    "East Bench": 1.20,
    "West Cutback": 0.45,
    "Stockpile": 0.55,
}

# Extraction cost ($/tonne) per zone
cost = {
    "North Pit": 12.50,
    "South Pit": 9.80,
    "East Bench": 18.00,
    "West Cutback": 8.50,
    "Stockpile": 5.00,
}

# Revenue per unit of contained metal ($/tonne of ore × grade)
cu_price = 85.0  # $/unit grade-tonne

# Maximum extractable tonnage per zone (kt)
max_tonnes = {
    "North Pit": 500,
    "South Pit": 800,
    "East Bench": 300,
    "West Cutback": 600,
    "Stockpile": 200,
}

# Mill capacity
mill_capacity = 1500  # kt total throughput
min_feed_grade = 0.60  # minimum blend grade (% Cu)
max_feed_grade = 1.00  # maximum blend grade (% Cu)

# Define high-grade and low-grade zone groups
high_grade_zones = ["North Pit", "East Bench"]
low_grade_zones = ["South Pit", "West Cutback", "Stockpile"]

print("\n--- Ore Zone Data ---")
print(f"{'Zone':<18} {'Grade (%Cu)':>12} {'Cost ($/t)':>12} {'Max (kt)':>10}")
print("-" * 55)
for z in zones:
    print(f"{z:<18} {grade[z]:>12.2f} {cost[z]:>12.2f} {max_tonnes[z]:>10}")

print(f"\nMill capacity: {mill_capacity} kt")
print(f"Feed grade window: {min_feed_grade}% - {max_feed_grade}% Cu")

# =============================================================================
# Model Setup — VariableDict creation with per-key bounds
# =============================================================================
print("\n--- Building Model ---")

# Extraction tonnes per zone (per-key upper bounds from max_tonnes)
extract = VariableDict("extract", zones, lb=0, ub=max_tonnes)

print(f"Created VariableDict: {extract}")
print(f"  Keys: {extract.keys()}")
print(f"  Variable count: {len(extract)}")

# Show individual variable access via __getitem__
print(f"\n  extract['North Pit'] → {extract['North Pit']}")
print(f"  extract['Stockpile'] → {extract['Stockpile']}")

# Check membership via __contains__
print(f"\n  'East Bench' in extract → {'East Bench' in extract}")
print(f"  'Underground' in extract → {'Underground' in extract}")

# =============================================================================
# Objective — Maximize profit using prod()
# =============================================================================
prob = Problem(name="mine_plan")

# Revenue = cu_price × (grade-weighted sum of extraction)
revenue_coeffs = {z: cu_price * grade[z] for z in zones}
revenue = extract.prod(revenue_coeffs)

# Cost = cost-weighted sum of extraction
total_cost = extract.prod(cost)

# Profit = revenue - cost
prob.maximize(revenue - total_cost)

print("\nObjective: maximize Σ(price × grade - cost) × extract[zone]")
print("  Revenue coefficients ($/t):")
for z in zones:
    net = revenue_coeffs[z] - cost[z]
    print(f"    {z:<18} revenue={revenue_coeffs[z]:6.2f}  cost={cost[z]:5.2f}  net={net:6.2f}")

# =============================================================================
# Constraints — Using sum(), sum(subset), and prod()
# =============================================================================

# 1. Mill throughput: total extraction <= mill capacity
#    Showcases: sum() — full sum over all keys
prob.subject_to(extract.sum() <= mill_capacity)
print(f"\nConstraint 1: Total extraction ≤ {mill_capacity} kt  [sum()]")

# 2. Minimum feed grade (blend constraint)
#    grade-weighted sum >= min_grade × total sum
#    Showcases: prod(dict) for weighted sum
prob.subject_to(extract.prod(grade) >= min_feed_grade * extract.sum())
print(f"Constraint 2: Blend grade ≥ {min_feed_grade}%  [prod()]")

# 3. Maximum feed grade
prob.subject_to(extract.prod(grade) <= max_feed_grade * extract.sum())
print(f"Constraint 3: Blend grade ≤ {max_feed_grade}%  [prod()]")

# 4. High-grade zones must supply at least 300 kt
#    Showcases: sum(subset) — partial sum over selected keys
prob.subject_to(extract.sum(high_grade_zones) >= 300)
print(f"Constraint 4: High-grade zones ≥ 300 kt  [sum(subset)]")

# 5. Low-grade zones capped at 60% of total feed
prob.subject_to(extract.sum(low_grade_zones) <= 0.6 * extract.sum())
print(f"Constraint 5: Low-grade zones ≤ 60% of feed  [sum(subset)]")

# 6. Minimum extraction per zone (keep equipment busy)
#    Showcases: items() for iteration over (key, variable) pairs
min_extract = 50  # kt minimum per zone
for key, var in extract.items():
    prob.subject_to(var >= min_extract)
print(f"Constraint 6: Each zone ≥ {min_extract} kt  [items() iteration]")

# =============================================================================
# Solve
# =============================================================================
print("\n--- Solving ---")
sol = prob.solve()

if not sol.is_optimal:
    print(f"Solver status: {sol.status}")
    exit(1)

# =============================================================================
# Results — Solution extraction via Solution[VariableDict]
# =============================================================================
print("\n--- Optimal Production Plan ---")

# Extract all results at once: Solution[VariableDict] → dict
result = sol[extract]
print(f"\nSolution type: {type(result).__name__}")  # dict

print(f"\n{'Zone':<18} {'Extract (kt)':>14} {'Grade (%Cu)':>12} {'Revenue ($k)':>14} {'Cost ($k)':>12}")
print("-" * 73)

total_tonnes = 0
total_metal = 0
total_revenue = 0
total_cost_val = 0

for zone in zones:
    t = result[zone]
    metal = t * grade[zone]
    rev = t * revenue_coeffs[zone]
    cst = t * cost[zone]
    total_tonnes += t
    total_metal += metal
    total_revenue += rev
    total_cost_val += cst
    print(f"{zone:<18} {t:>14.1f} {grade[zone]:>12.2f} {rev:>14.1f} {cst:>12.1f}")

print("-" * 73)
blend_grade = total_metal / total_tonnes if total_tonnes > 0 else 0
print(f"{'TOTAL':<18} {total_tonnes:>14.1f} {blend_grade:>12.2f} {total_revenue:>14.1f} {total_cost_val:>12.1f}")

profit = total_revenue - total_cost_val
print(f"\nProfit: ${profit:,.0f}k")
print(f"Blend grade: {blend_grade:.3f}% Cu (target: {min_feed_grade}-{max_feed_grade}%)")
print(f"Mill utilization: {total_tonnes/mill_capacity*100:.1f}%")

# Show subset sums from results
hg_total = sum(result[z] for z in high_grade_zones)
lg_total = sum(result[z] for z in low_grade_zones)
print(f"\nHigh-grade zones: {hg_total:.1f} kt ({hg_total/total_tonnes*100:.0f}%)")
print(f"Low-grade zones:  {lg_total:.1f} kt ({lg_total/total_tonnes*100:.0f}%)")

# Also show individual variable access from solution
print(f"\nIndividual access: extract['East Bench'] = {sol[extract['East Bench']]:.1f} kt")

# Show get_variables() — returns list of Variable objects
all_vars = extract.get_variables()
print(f"\nget_variables() returned {len(all_vars)} Variable objects")

# Show values() — returns list of Variable objects in key order
print(f"values() returns: {[v.name for v in extract.values()]}")

print("\n" + "=" * 70)
print("Demo complete — VariableDict methods showcased:")
print("  VariableDict(name, keys, lb, ub)  — creation with per-key bounds")
print("  vd['key']                         — individual variable access")
print("  vd.sum()                          — full sum")
print("  vd.sum(subset)                    — partial sum")
print("  vd.prod(coefficients)             — weighted sum")
print("  vd.keys(), values(), items()      — dict-like iteration")
print("  vd.get_variables()                — list of Variable objects")
print("  len(vd), 'key' in vd             — length and membership")
print("  solution[vd]                      — extract all results as dict")
print("=" * 70)
