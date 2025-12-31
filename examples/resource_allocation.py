"""Example: Large-Scale Resource Allocation

Demonstrates Optyx v1.2.0's scalability with VectorVariable for problems
with hundreds of decision variables.

This example solves a resource allocation problem where a company must
allocate budget across 100+ projects, considering:
- Expected returns per project
- Risk factors
- Resource constraints (budget, personnel, equipment)
- Project dependencies
- Sector diversification

NEW IN v1.2.0: VectorVariable makes this problem tractable with clean code.
"""

import numpy as np
import time
from optyx import VectorVariable, Problem, Parameter

print("=" * 70)
print("OPTYX v1.2.0 - Large-Scale Resource Allocation Demo")
print("=" * 70)

# =============================================================================
# Problem Scale
# =============================================================================
n_projects = 100  # 100 projects - not possible cleanly before v1.2.0!
n_sectors = 5  # Projects grouped into 5 sectors
n_resources = 3  # 3 resource types: budget, personnel, equipment

np.random.seed(42)

print("\n📊 Problem Scale")
print("-" * 50)
print(f"  Projects: {n_projects}")
print(f"  Sectors: {n_sectors}")
print(f"  Resource types: {n_resources}")
print(f"  Total decision variables: {n_projects}")

# =============================================================================
# Generate Problem Data
# =============================================================================
print("\n📈 Generating Problem Data...")

# Project expected returns (5% to 25% annually)
expected_returns = np.random.uniform(0.05, 0.25, n_projects)

# Project risk scores (higher = riskier)
risk_scores = np.random.uniform(0.1, 0.5, n_projects)

# Resource requirements per project (3 resources x 100 projects)
# Budget in $M, Personnel in FTEs, Equipment in units
resource_requirements = np.random.uniform(0.5, 5.0, (n_resources, n_projects))

# Total available resources
total_resources = np.array(
    [
        150.0,  # $150M total budget
        200.0,  # 200 FTEs available
        100.0,  # 100 equipment units
    ]
)
resource_names = ["Budget ($M)", "Personnel (FTE)", "Equipment (units)"]

# Assign projects to sectors (20 projects per sector)
sector_assignments = np.zeros((n_sectors, n_projects))
for i in range(n_projects):
    sector = i % n_sectors
    sector_assignments[sector, i] = 1
sector_names = ["Tech", "Healthcare", "Finance", "Energy", "Manufacturing"]

# Sector exposure limits
max_sector_exposure = 0.30  # Max 30% in any sector

# Project-level constraints
min_allocation = 0.0  # Can't invest negative
max_allocation = 0.10  # Max 10% per project (diversification)

print(
    f"  Expected returns: {expected_returns.min()*100:.1f}% - {expected_returns.max()*100:.1f}%"
)
print(f"  Risk scores: {risk_scores.min():.2f} - {risk_scores.max():.2f}")
print(f"  Max per project: {max_allocation*100:.0f}%")
print(f"  Max per sector: {max_sector_exposure*100:.0f}%")

# =============================================================================
# Build the Model (v1.2.0: VectorVariable)
# =============================================================================
print("\n🔧 Building Optimization Model...")
build_start = time.perf_counter()

# v1.2.0: Create 100 decision variables in ONE LINE
allocation = VectorVariable("alloc", n_projects, lb=min_allocation, ub=max_allocation)

# Risk-adjusted return parameter (for sensitivity analysis)
risk_aversion = Parameter("risk_aversion", value=1.0)

# Objective: Maximize risk-adjusted return
# = Expected return - (risk_aversion * weighted risk)
expected_return_expr = expected_returns @ allocation
weighted_risk = risk_scores @ allocation
objective = expected_return_expr - risk_aversion * weighted_risk

# Build problem
problem = (
    Problem("resource_allocation")
    .maximize(objective)
    .subject_to(allocation.sum().eq(1))  # Fully allocated
)

# Resource constraints (3 constraints)
for r in range(n_resources):
    resource_usage = resource_requirements[r] @ allocation
    problem = problem.subject_to(resource_usage <= total_resources[r])

# Sector exposure constraints (5 constraints)
for s in range(n_sectors):
    sector_exposure = sector_assignments[s] @ allocation
    problem = problem.subject_to(sector_exposure <= max_sector_exposure)

build_time = time.perf_counter() - build_start
print(f"  Model built in {build_time*1000:.1f} ms")
print(f"  Variables: {n_projects}")
print(f"  Constraints: {1 + n_resources + n_sectors} (budget + resources + sectors)")

# =============================================================================
# Solve the Problem
# =============================================================================
print("\n🚀 Solving...")
solve_start = time.perf_counter()

solution = problem.solve(method="SLSQP")

solve_time = time.perf_counter() - solve_start
print(f"  Status: {solution.status}")
print(f"  Solve time: {solve_time*1000:.1f} ms")

# =============================================================================
# Analyze Results
# =============================================================================
print("\n" + "=" * 70)
print("📊 SOLUTION ANALYSIS")
print("=" * 70)

# Extract optimal allocations
opt_alloc = np.array([solution[f"alloc[{i}]"] for i in range(n_projects)])

# Summary statistics
print("\n📈 Allocation Summary:")
print(f"  Total allocated: {opt_alloc.sum()*100:.2f}%")
print(f"  Non-zero allocations: {np.sum(opt_alloc > 0.001)}")
print(f"  Max allocation: {opt_alloc.max()*100:.2f}%")
print(f"  Mean allocation (non-zero): {opt_alloc[opt_alloc > 0.001].mean()*100:.2f}%")

# Portfolio metrics
portfolio_return = opt_alloc @ expected_returns
portfolio_risk = opt_alloc @ risk_scores
risk_adj_return = solution.objective_value

print("\n📊 Portfolio Metrics:")
print(f"  Expected Return: {portfolio_return*100:.2f}%")
print(f"  Weighted Risk Score: {portfolio_risk:.4f}")
print(f"  Risk-Adjusted Return: {risk_adj_return*100:.2f}%")

# Resource utilization
print("\n📦 Resource Utilization:")
for r in range(n_resources):
    usage = opt_alloc @ resource_requirements[r]
    utilization = usage / total_resources[r] * 100
    print(
        f"  {resource_names[r]}: {usage:.1f} / {total_resources[r]:.1f} ({utilization:.1f}%)"
    )

# Sector exposure
print("\n🏢 Sector Exposure:")
for s in range(n_sectors):
    exposure = opt_alloc @ sector_assignments[s]
    print(f"  {sector_names[s]}: {exposure*100:.1f}%")

# Top 10 projects by allocation
print("\n🏆 Top 10 Projects by Allocation:")
print(f"  {'Rank':<6} {'Project':<10} {'Allocation':>12} {'Return':>10} {'Risk':>10}")
print("  " + "-" * 50)

sorted_indices = np.argsort(opt_alloc)[::-1]
for rank, idx in enumerate(sorted_indices[:10], 1):
    print(
        f"  {rank:<6} Project {idx:<4} {opt_alloc[idx]*100:>10.2f}% "
        f"{expected_returns[idx]*100:>8.1f}% {risk_scores[idx]:>10.2f}"
    )

# =============================================================================
# Sensitivity Analysis: Risk Aversion
# =============================================================================
print("\n" + "=" * 70)
print("⚖️ SENSITIVITY ANALYSIS: Risk Aversion")
print("=" * 70)
print("\nUsing Parameter for fast re-solves...")

risk_levels = [0.5, 1.0, 2.0, 5.0]
sensitivity_results = []

print(f"\n{'Risk Aversion':>15} {'Return':>10} {'Risk':>10} {'Solve Time':>12}")
print("-" * 50)

for risk_level in risk_levels:
    risk_aversion.set(risk_level)  # v1.2.0: Fast parameter update

    start = time.perf_counter()
    sol = problem.solve(method="SLSQP")
    elapsed = time.perf_counter() - start

    alloc = np.array([sol[f"alloc[{i}]"] for i in range(n_projects)])
    ret = alloc @ expected_returns
    risk = alloc @ risk_scores

    sensitivity_results.append((risk_level, ret, risk, elapsed))
    print(
        f"{risk_level:>15.1f} {ret*100:>9.2f}% {risk:>10.4f} {elapsed*1000:>10.1f} ms"
    )

# =============================================================================
# Comparison: v1.2.0 vs Loop-Based Approach
# =============================================================================
print("\n" + "=" * 70)
print("⚡ PERFORMANCE: VectorVariable vs Loop-Based")
print("=" * 70)


def build_loop_based(n):
    """Build problem using loop-based variables (pre-v1.2.0 style)."""
    from optyx import Variable, Problem

    # Create variables one by one
    alloc = [Variable(f"alloc_{i}", lb=0, ub=0.1) for i in range(n)]

    # Objective via loops
    obj = sum(alloc[i] * (expected_returns[i] - risk_scores[i]) for i in range(n))

    # Budget constraint via loop
    budget = sum(alloc[i] for i in range(n))

    prob = Problem().maximize(obj).subject_to(budget.eq(1))

    # Resource constraints via loops
    for r in range(n_resources):
        usage = sum(alloc[i] * resource_requirements[r, i] for i in range(n))
        prob = prob.subject_to(usage <= total_resources[r])

    return prob


def build_vector_based(n):
    """Build problem using VectorVariable (v1.2.0 style)."""
    from optyx import VectorVariable, Problem

    alloc = VectorVariable("alloc", n, lb=0, ub=0.1)
    obj = (expected_returns[:n] - risk_scores[:n]) @ alloc

    prob = Problem().maximize(obj).subject_to(alloc.sum().eq(1))

    for r in range(n_resources):
        usage = resource_requirements[r, :n] @ alloc
        prob = prob.subject_to(usage <= total_resources[r])

    return prob


# Benchmark at different scales
scales = [20, 50, 100]
print(f"\n{'Scale':>8} {'Loop-Based':>15} {'VectorVariable':>15} {'Speedup':>10}")
print("-" * 50)

for n in scales:
    # Loop-based timing
    start = time.perf_counter()
    prob_loop = build_loop_based(n)
    loop_time = time.perf_counter() - start

    # Vector-based timing
    start = time.perf_counter()
    prob_vec = build_vector_based(n)
    vec_time = time.perf_counter() - start

    speedup = loop_time / vec_time if vec_time > 0 else float("inf")
    print(
        f"{n:>8} {loop_time*1000:>13.2f} ms {vec_time*1000:>13.2f} ms {speedup:>9.1f}x"
    )

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("✨ v1.2.0 Features Demonstrated")
print("=" * 70)
print("""
This large-scale example showcases:

1. VectorVariable for 100 decision variables:
   allocation = VectorVariable("alloc", 100, lb=0, ub=0.1)

2. Vectorized constraints:
   allocation @ resource_requirements[r] <= total_resources[r]

3. Parameter for fast sensitivity analysis:
   risk_aversion.set(new_value)  # No problem rebuild

4. Clean sector exposure constraints:
   allocation @ sector_assignments[s] <= max_exposure

5. Scalable performance:
   - Build 100-variable problem in milliseconds
   - Solve with SLSQP in < 100ms

Without VectorVariable, this would require:
- 100 individual Variable() calls
- Nested loops for objective and constraints
- Error-prone index bookkeeping
""")

print("=" * 70)
print("Demo complete!")
print("=" * 70)
