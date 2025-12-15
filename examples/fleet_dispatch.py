"""Example: Fleet Dispatch Optimization

Demonstrates a truck-shovel fleet dispatch optimization problem using optyx.
This is a simplified but realistic model for open-pit mining operations.

Problem:
- Assign haul trucks to loading units (shovels/excavators) 
- Maximize total throughput (tonnes per hour)
- Subject to:
  - Total truck fleet size constraint
  - Shovel dig rate capacity limits
  - Truck cycle time considerations
  - Crusher/dump destination capacity

Target Audience: Mining operations roles at BHP, Fortescue, Rio Tinto
"""

import numpy as np
from optyx import Variable, Problem
import time

print("=" * 70)
print("OPTYX - Fleet Dispatch Optimization Demo")
print("=" * 70)

# =============================================================================
# Problem Data
# =============================================================================
print("\n📊 Fleet Configuration")
print("-" * 50)

# Shovels (loading units)
n_shovels = 4
shovel_names = ["Shovel A", "Shovel B", "Excavator C", "Excavator D"]
shovel_dig_rates = np.array([3500, 4000, 2800, 3200])  # tonnes/hour capacity
shovel_locations = ["Pit North", "Pit South", "Pit East", "Pit West"]

# Trucks
n_trucks = 12  # Total available trucks
truck_payload = 220  # tonnes per load (CAT 793 class)

# Cycle times from each shovel to dump (minutes)
# Includes: travel to shovel + queue + load + travel to dump + dump + return
cycle_times = np.array([
    18,  # Shovel A - close to crusher
    22,  # Shovel B - medium distance
    25,  # Excavator C - far pit area
    20,  # Excavator D - medium distance
])

# Calculate trucks needed to match shovel capacity
# trucks_needed = dig_rate * cycle_time / (payload * 60)
trucks_to_match = shovel_dig_rates * cycle_times / (truck_payload * 60)

# Crusher capacity (destination constraint)
crusher_capacity = 10000  # tonnes/hour max

print(f"Shovels: {n_shovels}")
print(f"Total trucks: {n_trucks}")
print(f"Truck payload: {truck_payload} tonnes")
print(f"Crusher capacity: {crusher_capacity} t/h")

print("\n📍 Shovel Details:")
print(f"{'Shovel':<15} {'Location':<12} {'Dig Rate':>10} {'Cycle':>8} {'Match':>8}")
print("-" * 55)
for i in range(n_shovels):
    print(f"{shovel_names[i]:<15} {shovel_locations[i]:<12} "
          f"{shovel_dig_rates[i]:>8} t/h {cycle_times[i]:>6} min "
          f"{trucks_to_match[i]:>7.1f}")

# =============================================================================
# Decision Variables
# =============================================================================
print("\n🔧 Creating Decision Variables")
print("-" * 50)

# x[i] = number of trucks assigned to shovel i (continuous relaxation)
x = []
for i in range(n_shovels):
    x.append(Variable(f"trucks_{i}", lb=0, ub=n_trucks))

print(f"Variables: {n_shovels} (trucks per shovel)")

# =============================================================================
# Objective: Maximize Throughput
# =============================================================================
print("\n💰 Building Objective Function")
print("-" * 50)

# Throughput from shovel i = min(dig_rate, truck_delivery_rate)
# truck_delivery_rate = x[i] * payload * 60 / cycle_time[i]
# 
# For linear model, assume trucks are the bottleneck (common case):
# throughput[i] = x[i] * payload * 60 / cycle_time[i]
# 
# But we also cap at shovel dig rate via constraints

throughput = 0
for i in range(n_shovels):
    # Tonnes per hour delivered by trucks assigned to this shovel
    truck_rate = x[i] * truck_payload * 60 / cycle_times[i]
    throughput = throughput + truck_rate

print("Objective: Maximize total throughput (t/h)")

# =============================================================================
# Constraints
# =============================================================================
print("\n📋 Adding Constraints")
print("-" * 50)

prob = Problem(name="fleet_dispatch")
prob.maximize(throughput)

# 1. Total trucks available
total_trucks = sum(x[i] for i in range(n_shovels))
prob.subject_to(total_trucks <= n_trucks)
print(f"  ✓ Fleet size constraint: ≤ {n_trucks} trucks total")

# 2. Shovel dig rate limits (can't deliver more than shovel can dig)
for i in range(n_shovels):
    truck_rate = x[i] * truck_payload * 60 / cycle_times[i]
    prob.subject_to(truck_rate <= shovel_dig_rates[i])
print(f"  ✓ {n_shovels} shovel capacity constraints")

# 3. Crusher capacity (total throughput limit)
prob.subject_to(throughput <= crusher_capacity)
print(f"  ✓ Crusher capacity constraint: ≤ {crusher_capacity} t/h")

# 4. Minimum trucks per active shovel (operational requirement)
min_trucks_per_shovel = 1
for i in range(n_shovels):
    prob.subject_to(x[i] >= min_trucks_per_shovel)
print(f"  ✓ Minimum {min_trucks_per_shovel} truck(s) per shovel")

total_constraints = 1 + n_shovels + 1 + n_shovels
print(f"\nTotal constraints: {total_constraints}")

# =============================================================================
# Solve - Initial Dispatch
# =============================================================================
print("\n🚀 Solving Initial Dispatch...")
print("-" * 50)

solution = prob.solve(method="trust-constr")

print(f"Status: {solution.status.value}")
if solution.solve_time >= 1.0:
    print(f"Solve time: {solution.solve_time:.2f} s")
else:
    print(f"Solve time: {solution.solve_time*1000:.1f} ms")

# =============================================================================
# Results
# =============================================================================
print("\n📊 Optimal Fleet Assignment")
print("-" * 50)

print(f"\n{'Shovel':<15} {'Trucks':>8} {'Truck Rate':>12} {'Shovel Cap':>12} {'Utilization':>12}")
print("-" * 62)

total_throughput = 0
for i in range(n_shovels):
    trucks = solution[f"trucks_{i}"]
    truck_rate = trucks * truck_payload * 60 / cycle_times[i]
    shovel_cap = shovel_dig_rates[i]
    utilization = min(truck_rate, shovel_cap) / shovel_cap * 100
    actual_rate = min(truck_rate, shovel_cap)
    total_throughput += actual_rate
    
    print(f"{shovel_names[i]:<15} {trucks:>8.1f} {truck_rate:>10.0f} t/h "
          f"{shovel_cap:>10} t/h {utilization:>10.1f}%")

print("-" * 62)
print(f"{'TOTAL':<15} {sum(solution[f'trucks_{i}'] for i in range(n_shovels)):>8.1f} "
      f"{solution.objective_value:>10.0f} t/h")

print("\n📈 Performance Summary:")
print(f"  Total throughput: {solution.objective_value:,.0f} t/h")
print(f"  Crusher utilization: {solution.objective_value / crusher_capacity * 100:.1f}%")
print(f"  Fleet utilization: {sum(solution[f'trucks_{i}'] for i in range(n_shovels)) / n_trucks * 100:.1f}%")

# =============================================================================
# Real-time Re-optimization Scenario
# =============================================================================
print("\n" + "=" * 70)
print("⚡ REAL-TIME RE-OPTIMIZATION SCENARIO")
print("=" * 70)

print("\n🔴 Event: Shovel A breakdown! Taken offline for maintenance.")
print("-" * 50)

# Update shovel A capacity - completely offline
shovel_dig_rates_updated = shovel_dig_rates.copy()
shovel_dig_rates_updated[0] = 0  # Shovel A offline

# Rebuild problem with new variables and updated data
x2 = []
for i in range(n_shovels):
    x2.append(Variable(f"trucks_{i}", lb=0, ub=n_trucks))

throughput2 = sum(x2[i] * truck_payload * 60 / cycle_times[i] for i in range(n_shovels))

prob2 = Problem(name="fleet_dispatch_reoptimized")
prob2.maximize(throughput2)

# Same constraints with updated shovel capacity
prob2.subject_to(sum(x2[i] for i in range(n_shovels)) <= n_trucks)
for i in range(n_shovels):
    truck_rate = x2[i] * truck_payload * 60 / cycle_times[i]
    prob2.subject_to(truck_rate <= shovel_dig_rates_updated[i])
prob2.subject_to(throughput2 <= crusher_capacity)
# Minimum trucks only for active shovels
for i in range(n_shovels):
    if shovel_dig_rates_updated[i] > 0:
        prob2.subject_to(x2[i] >= min_trucks_per_shovel)

# Solve with timing (SLSQP is fast for re-optimization)
start = time.perf_counter()
solution2 = prob2.solve(method="SLSQP")
reopt_time = time.perf_counter() - start

print(f"Re-optimization time: {reopt_time*1000:.1f} ms  ⚡ Fast enough for real-time dispatch")

print("\n📊 Updated Fleet Assignment")
print(f"\n{'Shovel':<15} {'Before':>8} {'After':>8} {'Change':>10}")
print("-" * 45)

for i in range(n_shovels):
    before = solution[f"trucks_{i}"]
    after = solution2[f"trucks_{i}"]
    change = after - before
    sign = "+" if change > 0 else ""
    print(f"{shovel_names[i]:<15} {before:>8.1f} {after:>8.1f} {sign}{change:>9.1f}")

print("-" * 45)
throughput_loss = solution.objective_value - solution2.objective_value
print("\n📉 Impact Analysis:")
print(f"  Original throughput: {solution.objective_value:,.0f} t/h")
print(f"  New throughput: {solution2.objective_value:,.0f} t/h")
print(f"  Throughput loss: {throughput_loss:,.0f} t/h ({throughput_loss/solution.objective_value*100:.1f}%)")

# =============================================================================
# Shift Planning Scenario
# =============================================================================
print("\n" + "=" * 70)
print("📅 SHIFT PLANNING: What if we add 2 more trucks?")
print("=" * 70)

# Increase fleet size
n_trucks_new = n_trucks + 2

prob3 = Problem(name="fleet_expanded")

# Recreate with original dig rates but more trucks
x3 = []
for i in range(n_shovels):
    x3.append(Variable(f"trucks_{i}", lb=0, ub=n_trucks_new))

throughput3 = sum(x3[i] * truck_payload * 60 / cycle_times[i] for i in range(n_shovels))
prob3.maximize(throughput3)

prob3.subject_to(sum(x3[i] for i in range(n_shovels)) <= n_trucks_new)
for i in range(n_shovels):
    truck_rate = x3[i] * truck_payload * 60 / cycle_times[i]
    prob3.subject_to(truck_rate <= shovel_dig_rates[i])
prob3.subject_to(throughput3 <= crusher_capacity)
for i in range(n_shovels):
    prob3.subject_to(x3[i] >= min_trucks_per_shovel)

solution3 = prob3.solve(method="SLSQP")

improvement = solution3.objective_value - solution.objective_value
print(f"\n📈 With {n_trucks_new} trucks (was {n_trucks}):")
print(f"  New throughput: {solution3.objective_value:,.0f} t/h")
print(f"  Improvement: +{improvement:,.0f} t/h ({improvement/solution.objective_value*100:.1f}%)")

if improvement < 100:
    print("\n  ⚠️  Limited improvement - shovels or crusher may be the bottleneck")
else:
    print("\n  ✅ Significant improvement - trucks were the bottleneck")

print("\n" + "=" * 70)
print("Demo complete! This model can be extended with:")
print("  • Integer constraints for discrete truck assignments (MILP)")
print("  • Multiple dump destinations (crusher, stockpile, waste)")
print("  • Fuel consumption and cost optimization")
print("  • Stochastic cycle times and breakdowns")
print("  • Multi-shift scheduling with crew constraints")
print("=" * 70)
