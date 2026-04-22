"""Example: Mine Equipment Selection (Mixed-Integer Linear Programming)

Demonstrates MILP capabilities in Optyx using BinaryVariable, IntegerVariable,
and VectorVariable with domain="binary"/"integer".

Problem:
- A mining company must select which equipment to purchase/lease and decide
  how many shifts to operate each machine to meet production targets.
- Binary decisions: whether to acquire each piece of equipment
- Integer decisions: number of shifts per week for each machine
- Continuous decisions: tonnes allocated from each machine to each ore type

This showcases:
  - BinaryVariable for yes/no decisions
  - IntegerVariable for discrete quantities
  - VectorVariable with domain="binary" and domain="integer"
  - Mixed continuous + integer formulations
  - Big-M constraints linking binary and continuous variables
  - Solution MIP-specific fields: mip_gap, best_bound
"""

import numpy as np
from optyx import (
    BinaryVariable,
    IntegerVariable,
    Variable,
    VectorVariable,
    Problem,
)

print("=" * 70)
print("OPTYX - Mine Equipment Selection (MILP)")
print("=" * 70)

# =============================================================================
# Problem Data
# =============================================================================
equipment = ["Excavator A", "Excavator B", "Loader C", "Drill Rig D"]
n_equip = len(equipment)

# Fixed cost to acquire/lease each machine ($k/month)
fixed_cost = [120, 180, 85, 95]

# Capacity per shift (tonnes/shift)
capacity_per_shift = [500, 800, 350, 200]

# Operating cost per shift ($k/shift)
op_cost_per_shift = [8, 12, 5, 6]

# Maximum shifts per week
max_shifts = [14, 14, 21, 21]  # Some machines can run 3 shifts/day

# Production targets
min_production = 8000  # tonnes/week minimum
target_production = 12000  # tonnes/week ideal

print("\n--- Equipment Options ---")
print(f"{'Machine':<16} {'Fixed Cost':>12} {'Cap/Shift':>12} {'Op Cost':>10} {'Max Shifts':>12}")
print("-" * 65)
for i in range(n_equip):
    print(f"{equipment[i]:<16} ${fixed_cost[i]:>10}k {capacity_per_shift[i]:>10} t "
          f"${op_cost_per_shift[i]:>7}k {max_shifts[i]:>10}/wk")

print(f"\nMin production: {min_production} t/week")
print(f"Target production: {target_production} t/week")

# =============================================================================
# Part 1: Basic MILP — Binary Equipment Selection
# =============================================================================
print("\n" + "=" * 70)
print("Part 1: Binary Equipment Selection (BinaryVariable)")
print("=" * 70)

# Binary: acquire this equipment? (0 or 1)
acquire = [BinaryVariable(f"acquire_{equipment[i]}") for i in range(n_equip)]

prob1 = Problem(name="equipment_selection")

# Minimize fixed costs of acquired equipment
total_fixed = sum(fixed_cost[i] * acquire[i] for i in range(n_equip))
prob1.minimize(total_fixed)

# Must acquire enough capacity to meet minimum production
# (assuming max shifts for feasibility check)
total_capacity = sum(
    capacity_per_shift[i] * max_shifts[i] * acquire[i] for i in range(n_equip)
)
prob1.subject_to(total_capacity >= min_production)

# Need at least 2 machines for redundancy
prob1.subject_to(sum(acquire) >= 2)

sol1 = prob1.solve()

print(f"\nStatus: {sol1.status.name}")
print(f"MIP gap:    {sol1.mip_gap}")
print(f"Best bound: {sol1.best_bound}")

print(f"\nOptimal fixed cost: ${sol1.objective_value:.0f}k/month")
print("\nEquipment acquired:")
for i in range(n_equip):
    selected = sol1[acquire[i].name]
    marker = "YES" if selected > 0.5 else "no"
    print(f"  {equipment[i]:<16} → {marker}")

# =============================================================================
# Part 2: Integer Shift Scheduling
# =============================================================================
print("\n" + "=" * 70)
print("Part 2: Integer Shift Scheduling (IntegerVariable)")
print("=" * 70)

# Integer: how many shifts per week for each machine?
shifts = [
    IntegerVariable(f"shifts_{equipment[i]}", lb=0, ub=max_shifts[i])
    for i in range(n_equip)
]

prob2 = Problem(name="shift_scheduling")

# Minimize total operating cost
total_op_cost = sum(op_cost_per_shift[i] * shifts[i] for i in range(n_equip))
prob2.minimize(total_op_cost)

# Meet production target
production = sum(capacity_per_shift[i] * shifts[i] for i in range(n_equip))
prob2.subject_to(production >= target_production)

# Each machine needs at least 2 shifts if used at all (maintenance window)
# (simplified: just set minimum 2 shifts for all)
for i in range(n_equip):
    prob2.subject_to(shifts[i] >= 2)

sol2 = prob2.solve()

print(f"\nStatus: {sol2.status.name}")
print(f"Optimal operating cost: ${sol2.objective_value:.0f}k/week")

print(f"\n{'Machine':<16} {'Shifts/wk':>10} {'Production':>12} {'Cost':>10}")
print("-" * 52)
total_prod = 0
for i in range(n_equip):
    s = sol2[shifts[i].name]
    prod = s * capacity_per_shift[i]
    cost = s * op_cost_per_shift[i]
    total_prod += prod
    print(f"{equipment[i]:<16} {s:>10.0f} {prod:>10.0f} t ${cost:>7.0f}k")
print("-" * 52)
print(f"{'TOTAL':<16} {'':>10} {total_prod:>10.0f} t ${sol2.objective_value:>7.0f}k")

# =============================================================================
# Part 3: VectorVariable with domain="binary"
# =============================================================================
print("\n" + "=" * 70)
print("Part 3: Binary Knapsack (VectorVariable domain='binary')")
print("=" * 70)

# Spare parts selection: pick which parts to stock, maximize coverage
n_parts = 8
part_names = [f"Part-{chr(65+i)}" for i in range(n_parts)]
part_coverage = np.array([15, 22, 8, 30, 12, 18, 25, 10])  # coverage score
part_cost = np.array([5, 8, 3, 12, 4, 7, 10, 4])  # cost ($k)
budget = 30  # $k

# Binary vector: stock this part?
stock = VectorVariable("stock", n_parts, domain="binary")

prob3 = Problem(name="parts_selection")
prob3.maximize(part_coverage @ stock)  # maximize coverage
prob3.subject_to(part_cost @ stock <= budget)  # budget constraint
prob3.subject_to(stock.sum() >= 3)  # minimum 3 different parts

sol3 = prob3.solve()

print(f"\nStatus: {sol3.status.name}")
print(f"Maximum coverage score: {sol3.objective_value:.0f}")

print(f"\n{'Part':<10} {'Coverage':>10} {'Cost ($k)':>10} {'Selected':>10}")
print("-" * 43)
total_cost_parts = 0
for i in range(n_parts):
    selected = sol3[f"stock[{i}]"]
    sel_str = "YES" if selected > 0.5 else "-"
    if selected > 0.5:
        total_cost_parts += part_cost[i]
    print(f"{part_names[i]:<10} {part_coverage[i]:>10} {part_cost[i]:>10} {sel_str:>10}")
print(f"\nTotal cost: ${total_cost_parts}k / ${budget}k budget")

# =============================================================================
# Part 4: Mixed-Integer — Facility Location
# =============================================================================
print("\n" + "=" * 70)
print("Part 4: Mixed-Integer — Depot Location")
print("=" * 70)

# Mining depots: binary open/close + continuous allocation
depots = ["Central", "North", "South"]
sites = ["Pit A", "Pit B", "Pit C", "Pit D"]
n_depots = len(depots)
n_sites = len(sites)

depot_fixed_cost = [50, 35, 40]  # $k/month to operate
depot_capacity = [200, 150, 180]  # tonnes/day

# Transport cost per tonne from depot j to site k ($)
transport_cost = np.array([
    [3, 8, 5, 7],   # Central
    [6, 2, 9, 4],   # North
    [7, 5, 3, 2],   # South
])

site_demand = [60, 45, 55, 40]  # tonnes/day

prob4 = Problem(name="depot_location")

# Binary: open depot j?
open_depot = [BinaryVariable(f"open_{depots[j]}") for j in range(n_depots)]

# Continuous: tonnes from depot j to site k
alloc = {}
for j in range(n_depots):
    for k in range(n_sites):
        alloc[j, k] = Variable(f"alloc_{depots[j]}_{sites[k]}", lb=0)

# Minimize: fixed costs + transport costs
obj = sum(depot_fixed_cost[j] * open_depot[j] for j in range(n_depots))
for j in range(n_depots):
    for k in range(n_sites):
        obj = obj + transport_cost[j, k] * alloc[j, k]
prob4.minimize(obj)

# Demand satisfaction
for k in range(n_sites):
    prob4.subject_to(
        sum(alloc[j, k] for j in range(n_depots)) >= site_demand[k]
    )

# Capacity & linking (Big-M: can only allocate from open depots)
for j in range(n_depots):
    prob4.subject_to(
        sum(alloc[j, k] for k in range(n_sites)) <= depot_capacity[j] * open_depot[j]
    )

sol4 = prob4.solve()

print(f"\nStatus: {sol4.status.name}")
print(f"Total cost: ${sol4.objective_value:.1f}k")
print(f"MIP gap: {sol4.mip_gap}")

print("\nDepot decisions:")
for j in range(n_depots):
    opened = sol4[open_depot[j].name]
    status = "OPEN" if opened > 0.5 else "closed"
    print(f"  {depots[j]:<10} → {status}")

print(f"\n{'From → To':<25} {'Tonnes/day':>12}")
print("-" * 40)
for j in range(n_depots):
    if sol4[open_depot[j].name] > 0.5:
        for k in range(n_sites):
            val = sol4[alloc[j, k].name]
            if val > 0.1:
                print(f"  {depots[j]} → {sites[k]:<12} {val:>10.1f}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("MILP features demonstrated:")
print("  BinaryVariable('name')              — 0/1 decisions")
print("  IntegerVariable('name', lb, ub)     — discrete quantities")
print("  VectorVariable('x', n, domain='binary')  — binary vectors")
print("  VectorVariable('x', n, domain='integer') — integer vectors")
print("  Big-M linking constraints            — binary × continuous")
print("  sol.mip_gap, sol.best_bound          — MIP solution quality")
print("  Automatic MILP routing               — detected from domains")
print("=" * 70)
