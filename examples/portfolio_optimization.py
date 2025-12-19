"""Example: Commodity Portfolio Optimization

Demonstrates a mean-variance portfolio optimization problem using optyx.
This is a classic Markowitz-style model adapted for commodity investments.

Problem:
- Allocate capital across multiple commodities
- Maximize expected return for a given risk level
- Subject to:
  - Budget constraint (weights sum to 1)
  - No short selling (non-negative weights)
  - Risk constraint (portfolio variance limit)
  - Optional sector exposure limits
"""

import numpy as np
from optyx import Variable, Problem

print("=" * 70)
print("OPTYX - Commodity Portfolio Optimization Demo")
print("=" * 70)

# =============================================================================
# Problem Data
# =============================================================================
print("\n📊 Commodity Universe")
print("-" * 50)

# Commodities
commodities = ["Gold", "Silver", "Copper", "Oil", "Natural Gas", "Wheat"]
n_assets = len(commodities)

# Expected annual returns (based on historical/forecast data)
expected_returns = np.array([0.08, 0.10, 0.12, 0.15, 0.18, 0.09])  # 8-18%

# Covariance matrix (annualized)
# Constructed to reflect realistic commodity correlations
correlation = np.array(
    [
        [1.00, 0.75, 0.30, 0.10, 0.05, 0.15],  # Gold
        [0.75, 1.00, 0.40, 0.15, 0.10, 0.20],  # Silver
        [0.30, 0.40, 1.00, 0.50, 0.35, 0.25],  # Copper
        [0.10, 0.15, 0.50, 1.00, 0.60, 0.30],  # Oil
        [0.05, 0.10, 0.35, 0.60, 1.00, 0.20],  # Natural Gas
        [0.15, 0.20, 0.25, 0.30, 0.20, 1.00],  # Wheat
    ]
)

# Standard deviations (annualized volatility)
std_devs = np.array([0.15, 0.25, 0.28, 0.35, 0.45, 0.22])

# Covariance = correlation * outer(std, std)
covariance = correlation * np.outer(std_devs, std_devs)

print(f"Assets: {n_assets} commodities")
print(f"\n{'Commodity':<12} {'Exp Return':>12} {'Volatility':>12} {'Sharpe':>10}")
print("-" * 48)
risk_free_rate = 0.04  # 4% risk-free rate
for i in range(n_assets):
    sharpe = (expected_returns[i] - risk_free_rate) / std_devs[i]
    print(
        f"{commodities[i]:<12} {expected_returns[i] * 100:>10.1f}% "
        f"{std_devs[i] * 100:>10.1f}% {sharpe:>10.2f}"
    )

# =============================================================================
# Decision Variables
# =============================================================================
print("\n🔧 Creating Decision Variables")
print("-" * 50)

# w[i] = weight allocated to commodity i
w = []
for i in range(n_assets):
    w.append(Variable(f"w_{commodities[i]}", lb=0, ub=1))

print(f"Variables: {n_assets} portfolio weights")

# =============================================================================
# Helper: Build Portfolio Expressions
# =============================================================================


def portfolio_return(weights, returns):
    """Expected portfolio return: Σ w_i * r_i"""
    return sum(weights[i] * returns[i] for i in range(len(weights)))


def portfolio_variance(weights, cov_matrix):
    """Portfolio variance: Σ Σ w_i * w_j * cov_ij"""
    n = len(weights)
    variance = 0
    for i in range(n):
        for j in range(n):
            variance = variance + weights[i] * weights[j] * cov_matrix[i, j]
    return variance


# =============================================================================
# Problem 1: Maximum Return for Given Risk
# =============================================================================
print("\n" + "=" * 70)
print("📈 SCENARIO 1: Maximum Return Portfolio")
print("=" * 70)
print("Objective: Maximize return subject to risk constraint")

target_volatility = 0.20  # 20% max volatility
target_variance = target_volatility**2

prob1 = Problem(name="max_return")

# Objective: maximize expected return
exp_return = portfolio_return(w, expected_returns)
prob1.maximize(exp_return)

# Constraints
# 1. Budget: weights sum to 1
prob1.subject_to(sum(w[i] for i in range(n_assets)) <= 1)
prob1.subject_to(sum(w[i] for i in range(n_assets)) >= 0.99)  # Approximately equality

# 2. Risk: variance ≤ target
variance_expr = portfolio_variance(w, covariance)
prob1.subject_to(variance_expr <= target_variance)

print("\nConstraints:")
print("  • Budget: Fully invested (Σw = 1)")
print(f"  • Risk: Volatility ≤ {target_volatility * 100:.0f}%")
print("  • No short selling: w ≥ 0")

# Solve
solution1 = prob1.solve(method="trust-constr")

print("\n🎯 Optimal Portfolio (Max Return):")
print(f"{'Commodity':<12} {'Weight':>10} {'Contribution':>14}")
print("-" * 38)

total_weight = 0
for i in range(n_assets):
    weight = solution1[f"w_{commodities[i]}"]
    if weight > 0.001:  # Only show non-zero allocations
        contribution = weight * expected_returns[i]
        print(
            f"{commodities[i]:<12} {weight * 100:>9.1f}% {contribution * 100:>12.2f}%"
        )
        total_weight += weight

print("-" * 38)
print(f"{'Total':<12} {total_weight * 100:>9.1f}%")

# Calculate portfolio metrics
port_return = solution1.objective_value
port_var = sum(
    solution1[f"w_{commodities[i]}"]
    * solution1[f"w_{commodities[j]}"]
    * covariance[i, j]
    for i in range(n_assets)
    for j in range(n_assets)
)
port_vol = np.sqrt(port_var)
port_sharpe = (port_return - risk_free_rate) / port_vol

print("\n📊 Portfolio Metrics:")
print(f"  Expected Return: {port_return * 100:.2f}%")
print(f"  Volatility: {port_vol * 100:.2f}%")
print(f"  Sharpe Ratio: {port_sharpe:.2f}")
if solution1.solve_time >= 1.0:
    print(f"  Solve time: {solution1.solve_time:.2f} s")
else:
    print(f"  Solve time: {solution1.solve_time * 1000:.1f} ms")

# =============================================================================
# Problem 2: Minimum Risk for Given Return
# =============================================================================
print("\n" + "=" * 70)
print("📉 SCENARIO 2: Minimum Variance Portfolio")
print("=" * 70)
print("Objective: Minimize risk subject to return constraint")

target_return = 0.12  # 12% minimum return

# Create new variables for this problem
w2 = []
for i in range(n_assets):
    w2.append(Variable(f"w_{commodities[i]}", lb=0, ub=1))

prob2 = Problem(name="min_variance")

# Objective: minimize variance (negative for maximization framework)
variance_expr2 = portfolio_variance(w2, covariance)
prob2.minimize(variance_expr2)

# Constraints
prob2.subject_to(sum(w2[i] for i in range(n_assets)) <= 1)
prob2.subject_to(sum(w2[i] for i in range(n_assets)) >= 0.99)
exp_return2 = portfolio_return(w2, expected_returns)
prob2.subject_to(exp_return2 >= target_return)

print("\nConstraints:")
print("  • Budget: Fully invested (Σw = 1)")
print(f"  • Return: Expected return ≥ {target_return * 100:.0f}%")
print("  • No short selling: w ≥ 0")

solution2 = prob2.solve(method="trust-constr")

print("\n🎯 Optimal Portfolio (Min Variance):")
print(f"{'Commodity':<12} {'Weight':>10}")
print("-" * 24)

for i in range(n_assets):
    weight = solution2[f"w_{commodities[i]}"]
    if weight > 0.001:
        print(f"{commodities[i]:<12} {weight * 100:>9.1f}%")

# Calculate metrics
port_return2 = sum(
    solution2[f"w_{commodities[i]}"] * expected_returns[i] for i in range(n_assets)
)
port_var2 = (
    -solution2.objective_value
    if solution2.objective_value < 0
    else solution2.objective_value
)
port_vol2 = np.sqrt(port_var2)
port_sharpe2 = (port_return2 - risk_free_rate) / port_vol2

print("\n📊 Portfolio Metrics:")
print(f"  Expected Return: {port_return2 * 100:.2f}%")
print(f"  Volatility: {port_vol2 * 100:.2f}%")
print(f"  Sharpe Ratio: {port_sharpe2:.2f}")

# =============================================================================
# Efficient Frontier
# =============================================================================
print("\n" + "=" * 70)
print("📊 EFFICIENT FRONTIER")
print("=" * 70)

# Generate efficient frontier by varying target return
target_returns_range = np.linspace(0.08, 0.17, 10)
frontier_returns = []
frontier_volatilities = []

print("\nComputing efficient frontier points...")

for target_ret in target_returns_range:
    # Create fresh variables for each point
    w_ef = []
    for i in range(n_assets):
        w_ef.append(Variable(f"w_{i}", lb=0, ub=1))

    prob_ef = Problem()
    variance_ef = portfolio_variance(w_ef, covariance)
    prob_ef.minimize(variance_ef)

    prob_ef.subject_to(sum(w_ef[i] for i in range(n_assets)) <= 1)
    prob_ef.subject_to(sum(w_ef[i] for i in range(n_assets)) >= 0.99)
    return_ef = portfolio_return(w_ef, expected_returns)
    prob_ef.subject_to(return_ef >= target_ret)

    sol_ef = prob_ef.solve(method="SLSQP")

    # Calculate actual metrics
    actual_return = sum(sol_ef[f"w_{i}"] * expected_returns[i] for i in range(n_assets))
    actual_var = sum(
        sol_ef[f"w_{i}"] * sol_ef[f"w_{j}"] * covariance[i, j]
        for i in range(n_assets)
        for j in range(n_assets)
    )
    actual_vol = np.sqrt(max(0, actual_var))

    frontier_returns.append(actual_return)
    frontier_volatilities.append(actual_vol)

# Display frontier as ASCII art
print("\n  Efficient Frontier (Risk vs Return)")
print("  " + "-" * 52)

# Simple ASCII visualization
max_ret = max(frontier_returns)
min_ret = min(frontier_returns)
max_vol = max(frontier_volatilities)
min_vol = min(frontier_volatilities)

height = 12
width = 50

# Create grid
grid = [[" " for _ in range(width)] for _ in range(height)]

# Plot points
for ret, vol in zip(frontier_returns, frontier_volatilities):
    x = int((vol - min_vol) / (max_vol - min_vol + 0.001) * (width - 1))
    y = int((ret - min_ret) / (max_ret - min_ret + 0.001) * (height - 1))
    y = height - 1 - y  # Flip y-axis
    x = min(max(0, x), width - 1)
    y = min(max(0, y), height - 1)
    grid[y][x] = "●"

# Print grid
print(f"  {max_ret * 100:5.1f}% │{''.join(grid[0])}")
for row in grid[1:-1]:
    print(f"        │{''.join(row)}")
print(f"  {min_ret * 100:5.1f}% │{''.join(grid[-1])}")
print(f"        └{'─' * width}")
print(f"         {min_vol * 100:5.1f}%{' ' * (width - 12)}{max_vol * 100:5.1f}%")
print("                    Volatility →")

# Print data table
print("\n  Efficient Frontier Data:")
print(f"  {'Return':>8} {'Volatility':>12} {'Sharpe':>10}")
print("  " + "-" * 32)
for ret, vol in zip(frontier_returns, frontier_volatilities):
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
    print(f"  {ret * 100:>7.2f}% {vol * 100:>10.2f}% {sharpe:>10.2f}")

# =============================================================================
# Scenario Analysis: Commodity Price Shock
# =============================================================================
print("\n" + "=" * 70)
print("⚡ SCENARIO 3: Oil Price Shock Analysis")
print("=" * 70)

print("\nWhat if oil expected return drops from 15% to 5%?")

# Update expected returns
expected_returns_shocked = expected_returns.copy()
expected_returns_shocked[3] = 0.05  # Oil drops to 5%

# Reoptimize with max return objective
w3 = []
for i in range(n_assets):
    w3.append(Variable(f"w_{commodities[i]}", lb=0, ub=1))

prob3 = Problem(name="shocked_portfolio")
exp_return3 = portfolio_return(w3, expected_returns_shocked)
prob3.maximize(exp_return3)

prob3.subject_to(sum(w3[i] for i in range(n_assets)) <= 1)
prob3.subject_to(sum(w3[i] for i in range(n_assets)) >= 0.99)
variance_expr3 = portfolio_variance(w3, covariance)
prob3.subject_to(variance_expr3 <= target_variance)

solution3 = prob3.solve(method="trust-constr")

print("\n📊 Portfolio Rebalancing:")
print(f"{'Commodity':<12} {'Before':>10} {'After':>10} {'Change':>10}")
print("-" * 45)

for i in range(n_assets):
    before = solution1[f"w_{commodities[i]}"]
    after = solution3[f"w_{commodities[i]}"]
    change = after - before
    if abs(before) > 0.001 or abs(after) > 0.001:
        sign = "+" if change > 0 else ""
        print(
            f"{commodities[i]:<12} {before * 100:>9.1f}% {after * 100:>9.1f}% {sign}{change * 100:>8.1f}%"
        )

# New metrics
new_return = solution3.objective_value
new_var = sum(
    solution3[f"w_{commodities[i]}"]
    * solution3[f"w_{commodities[j]}"]
    * covariance[i, j]
    for i in range(n_assets)
    for j in range(n_assets)
)
new_vol = np.sqrt(new_var)

print("\n📉 Impact:")
print(
    f"  Return: {port_return * 100:.2f}% → {new_return * 100:.2f}% (Δ {(new_return - port_return) * 100:+.2f}%)"
)
print(f"  Volatility: {port_vol * 100:.2f}% → {new_vol * 100:.2f}%")

print("\n" + "=" * 70)
print("Demo complete! This model can be extended with:")
print("  • Transaction costs and turnover constraints")
print("  • Sector/geography exposure limits")
print("  • Conditional Value-at-Risk (CVaR) optimization")
print("  • Multi-period rebalancing strategies")
print("  • Factor-based risk models")
print("=" * 70)
