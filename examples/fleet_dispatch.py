"""Example: Fleet Dispatch Optimization

Demonstrates a truck-shovel fleet dispatch optimization problem using optyx.
This is a simplified but realistic model for open-pit mining operations.

Problem:
- Assign haul trucks to loading units (shovels/excavators)
- Maximize throughput (tonnes per hour)
- Subject to:
  - Total truck fleet size constraint
  - Shovel dig rate capacity limits
  - Truck cycle time considerations
  - Crusher/dump destination capacity
"""

import numpy as np
from optyx import VectorVariable, Problem
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
cycle_times = np.array(
    [
        18,  # Shovel A - close to crusher
        22,  # Shovel B - medium distance
        25,  # Excavator C - far pit area
        20,  # Excavator D - medium distance
    ]
)

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
    print(
        f"{shovel_names[i]:<15} {shovel_locations[i]:<12} "
        f"{shovel_dig_rates[i]:>8} t/h {cycle_times[i]:>6} min "
        f"{trucks_to_match[i]:>7.1f}"
    )

# =============================================================================
# Decision Variables
# =============================================================================
print("\n🔧 Creating Decision Variables")
print("-" * 50)

# Productivity (t/h per truck)
productivity = truck_payload * 60 / cycle_times

# Decision variables: trucks per shovel (integer allocation)
# We set lb=1 to ensure minimum trucks per active shovel
x = VectorVariable("trucks", n_shovels, lb=1, ub=n_trucks, domain="integer")

print(f"Variables: {n_shovels} (trucks per shovel, integer)")

# =============================================================================
# Objective: Maximize Throughput
# =============================================================================
print("\n💰 Building Objective Function")
print("-" * 50)

# Total throughput: dot product of productivity and trucks
total_throughput = productivity @ x

# Build problem
prob = Problem("fleet_dispatch").maximize(total_throughput)

print("Objective: Maximize Total Throughput")

# =============================================================================
# Constraints
# =============================================================================
print("\n🚧 Adding Constraints")
print("-" * 50)

# 1. Fleet size constraint
prob.subject_to(x.sum() <= n_trucks)
print(f"  1. Total trucks <= {n_trucks}")

# 2. Crusher capacity constraint
prob.subject_to(total_throughput <= crusher_capacity)
print(f"  2. Total throughput <= {crusher_capacity} t/h")

# 3. Shovel capacity constraints
# Each shovel's throughput must not exceed its dig rate
for i in range(n_shovels):
    prob.subject_to(x[i] * productivity[i] <= shovel_dig_rates[i])
    print(f"  3.{i + 1} {shovel_names[i]} throughput <= {shovel_dig_rates[i]} t/h")

# =============================================================================
# Solve
# =============================================================================
print("\n🚀 Solving...")
print("-" * 50)

start = time.time()
solution = prob.solve()
solve_time = (time.time() - start) * 1000

print("=" * 55)
print("FLEET DISPATCH SOLUTION")
print("=" * 55)
print(f"Status: {solution.status.name}")
print(f"Solve time: {solve_time:.1f}ms")
print()

print("Optimal Truck Assignments:")
print(f"{'Shovel':<15} {'Trucks':>8} {'Throughput':>12} {'Utilization':>12}")
print("-" * 50)

# Get solution values as numpy array
x_opt = x.to_numpy(solution.values)
total_trucks = 0

for i in range(n_shovels):
    trucks = x_opt[i]
    throughput = trucks * productivity[i]
    utilization = throughput / shovel_dig_rates[i] * 100
    total_trucks += trucks
    print(
        f"{shovel_names[i]:<15} {trucks:>8.1f} {throughput:>10.0f} t/h {utilization:>10.0f}%"
    )

print("-" * 50)
print(f"{'TOTAL':<15} {total_trucks:>8.1f} {solution.objective_value:>10.0f} t/h")
