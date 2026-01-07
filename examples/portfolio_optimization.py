"""Example: Commodity Portfolio Optimization

Demonstrates a mean-variance portfolio optimization problem using Optyx.
This is a classic Markowitz-style model adapted for commodity investments.

Math-like syntax: Uses `w.dot(Σ @ w)` for variance (reads as wᵀΣw).

Problem:
- Allocate capital across multiple commodities
- Maximize expected return for a given risk level
- Subject to:
  - Budget constraint (weights sum to 1)
  - No short selling (non-negative weights)
  - Risk constraint (portfolio variance limit)
  - Optional sector exposure limits
"""

import time

import numpy as np
from optyx import VectorVariable, Problem, Parameter

print("=" * 70)
print("OPTYX v1.2.0 - Commodity Portfolio Optimization Demo")
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
# Decision Variables (v1.2.0: VectorVariable)
# =============================================================================
print("\n🔧 Creating Decision Variables")
print("-" * 50)

# v1.2.0: Create all weights in one line with VectorVariable
w = VectorVariable("w", n_assets, lb=0, ub=1)

print(f"Variables: VectorVariable 'w' with {len(w)} elements")
print(f"Bounds: [{w.lb}, {w.ub}]")

# =============================================================================
# Portfolio Expressions (v1.2.0: Vectorized Operations)
# =============================================================================

# Portfolio return: μᵀw via dot product (no loops!)
portfolio_return = expected_returns @ w  # Clean and efficient

# Portfolio variance: wᵀΣw via math-like syntax
portfolio_variance = w.dot(covariance @ w)  # Reads as w · (Σw)

print("\nExpressions created:")
print(f"  Return: {type(portfolio_return).__name__}")
print(f"  Variance: {type(portfolio_variance).__name__} (with analytic gradient)")

# =============================================================================
# Problem 1: Maximum Return for Given Risk
# =============================================================================
print("\n" + "=" * 70)
print("📈 SCENARIO 1: Maximum Return Portfolio")
print("=" * 70)
print("Objective: Maximize return subject to risk constraint")

target_volatility = 0.20  # 20% max volatility
target_variance = target_volatility**2

# v1.2.0: Clean constraint syntax
prob1 = (
    Problem("max_return")
    .maximize(portfolio_return)
    .subject_to(w.sum().eq(1))  # Budget: fully invested
    .subject_to(portfolio_variance <= target_variance)  # Risk limit
)

print("\nConstraints:")
print("  • Budget: Fully invested (w.sum().eq(1))")
print(f"  • Risk: Volatility ≤ {target_volatility * 100:.0f}%")
print("  • No short selling: w ≥ 0 (via bounds)")

# Solve
solution1 = prob1.solve(method="trust-constr")

print("\n🎯 Optimal Portfolio (Max Return):")
print(f"{'Commodity':<12} {'Weight':>10} {'Contribution':>14}")
print("-" * 38)

# Extract weights efficiently
opt_weights = np.array([solution1[f"w[{i}]"] for i in range(n_assets)])
total_weight = 0

for i in range(n_assets):
    weight = opt_weights[i]
    if weight > 0.001:  # Only show non-zero allocations
        contribution = weight * expected_returns[i]
        print(
            f"{commodities[i]:<12} {weight * 100:>9.1f}% {contribution * 100:>12.2f}%"
        )
        total_weight += weight

print("-" * 38)
print(f"{'Total':<12} {total_weight * 100:>9.1f}%")

# Calculate portfolio metrics using vectorized operations
port_return = opt_weights @ expected_returns
port_var = opt_weights @ covariance @ opt_weights
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

# Fresh VectorVariable for new problem
w2 = VectorVariable("w", n_assets, lb=0, ub=1)
port_ret2 = expected_returns @ w2
port_var2 = w2.dot(covariance @ w2)  # wᵀΣw

prob2 = (
    Problem("min_variance")
    .minimize(port_var2)
    .subject_to(w2.sum().eq(1))
    .subject_to(port_ret2 >= target_return)
)

print("\nConstraints:")
print("  • Budget: Fully invested (w.sum().eq(1))")
print(f"  • Return: Expected return ≥ {target_return * 100:.0f}%")
print("  • No short selling: w ≥ 0 (via bounds)")

solution2 = prob2.solve(method="trust-constr")

print("\n🎯 Optimal Portfolio (Min Variance):")
print(f"{'Commodity':<12} {'Weight':>10}")
print("-" * 24)

weights2 = np.array([solution2[f"w[{i}]"] for i in range(n_assets)])
for i in range(n_assets):
    if weights2[i] > 0.001:
        print(f"{commodities[i]:<12} {weights2[i] * 100:>9.1f}%")

# Metrics
port_return2 = weights2 @ expected_returns
port_var2_val = weights2 @ covariance @ weights2
port_vol2 = np.sqrt(port_var2_val)
port_sharpe2 = (port_return2 - risk_free_rate) / port_vol2

print("\n📊 Portfolio Metrics:")
print(f"  Expected Return: {port_return2 * 100:.2f}%")
print(f"  Volatility: {port_vol2 * 100:.2f}%")
print(f"  Sharpe Ratio: {port_sharpe2:.2f}")

# =============================================================================
# Efficient Frontier (v1.2.0: Using Parameter for fast re-solves)
# =============================================================================
print("\n" + "=" * 70)
print("📊 EFFICIENT FRONTIER (with Parameters)")
print("=" * 70)

# Use Parameter for efficient frontier computation
# Build problem once, update parameter for each point
w_ef = VectorVariable("w", n_assets, lb=0, ub=1)
target_param = Parameter("target", value=0.08)  # Updatable target return

port_ret_ef = expected_returns @ w_ef
port_var_ef = w_ef.dot(covariance @ w_ef)  # wᵀΣw

prob_ef = (
    Problem("efficient_frontier")
    .minimize(port_var_ef)
    .subject_to(w_ef.sum().eq(1))
    .subject_to(port_ret_ef >= target_param)
)

# Compute frontier points (much faster with Parameter!)
target_returns_range = np.linspace(0.08, 0.17, 10)
frontier_returns = []
frontier_volatilities = []

print("\nComputing efficient frontier points with Parameter (fast re-solves)...")

start_time = time.perf_counter()

for i, target_ret in enumerate(target_returns_range):
    target_param.set(target_ret)  # v1.2.0: Update without rebuilding problem
    sol_ef = prob_ef.solve(method="SLSQP")

    weights_ef = np.array([sol_ef[f"w[{i}]"] for i in range(n_assets)])
    actual_return = weights_ef @ expected_returns
    actual_vol = np.sqrt(weights_ef @ covariance @ weights_ef)

    frontier_returns.append(actual_return)
    frontier_volatilities.append(actual_vol)

total_time = time.perf_counter() - start_time
print(f"Computed {len(target_returns_range)} points in {total_time * 1000:.1f} ms")
print(f"Average: {total_time / len(target_returns_range) * 1000:.2f} ms per point")

# Display frontier as ASCII art
print("\n  Efficient Frontier (Risk vs Return)")
print("  " + "-" * 52)

max_ret = max(frontier_returns)
min_ret = min(frontier_returns)
max_vol = max(frontier_volatilities)
min_vol = min(frontier_volatilities)

height = 12
width = 50

grid = [[" " for _ in range(width)] for _ in range(height)]

for ret, vol in zip(frontier_returns, frontier_volatilities):
    x = int((vol - min_vol) / (max_vol - min_vol + 0.001) * (width - 1))
    y = int((ret - min_ret) / (max_ret - min_ret + 0.001) * (height - 1))
    y = height - 1 - y
    x = min(max(0, x), width - 1)
    y = min(max(0, y), height - 1)
    grid[y][x] = "●"

print(f"  {max_ret * 100:5.1f}% │{''.join(grid[0])}")
for row in grid[1:-1]:
    print(f"        │{''.join(row)}")
print(f"  {min_ret * 100:5.1f}% │{''.join(grid[-1])}")
print(f"        └{'─' * width}")
print(f"         {min_vol * 100:5.1f}%{' ' * (width - 12)}{max_vol * 100:5.1f}%")
print("                    Volatility →")

# Data table
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

# Shocked expected returns
expected_returns_shocked = expected_returns.copy()
expected_returns_shocked[3] = 0.05  # Oil drops to 5%

# New VectorVariable for shocked scenario
w3 = VectorVariable("w", n_assets, lb=0, ub=1)
port_ret3 = expected_returns_shocked @ w3
port_var3 = w3.dot(covariance @ w3)  # wᵀΣw

prob3 = (
    Problem("shocked_portfolio")
    .maximize(port_ret3)
    .subject_to(w3.sum().eq(1))
    .subject_to(port_var3 <= target_variance)
)

solution3 = prob3.solve(method="trust-constr")
weights3 = np.array([solution3[f"w[{i}]"] for i in range(n_assets)])

print("\n📊 Portfolio Rebalancing:")
print(f"{'Commodity':<12} {'Before':>10} {'After':>10} {'Change':>10}")
print("-" * 45)

for i in range(n_assets):
    before = opt_weights[i]
    after = weights3[i]
    change = after - before
    if abs(before) > 0.001 or abs(after) > 0.001:
        sign = "+" if change > 0 else ""
        print(
            f"{commodities[i]:<12} {before * 100:>9.1f}% {after * 100:>9.1f}% {sign}{change * 100:>8.1f}%"
        )

# New metrics
new_return = weights3 @ expected_returns_shocked
new_vol = np.sqrt(weights3 @ covariance @ weights3)

print("\n📉 Impact:")
print(
    f"  Return: {port_return * 100:.2f}% → {new_return * 100:.2f}% (Δ {(new_return - port_return) * 100:+.2f}%)"
)
print(f"  Volatility: {port_vol * 100:.2f}% → {new_vol * 100:.2f}%")

# =============================================================================
# v1.2.0 Highlights
# =============================================================================
print("\n" + "=" * 70)
print("✨ Optyx Features Demonstrated")
print("=" * 70)
print("""
This example showcases Optyx's math-like syntax:

1. VectorVariable - Create many variables in one line:
   w = VectorVariable("w", n_assets, lb=0, ub=1)

2. Vectorized operations - No loops for dot products:
   portfolio_return = expected_returns @ w  # μᵀw

3. Math-like quadratic form - Reads as w·(Σw) = wᵀΣw:
   portfolio_variance = w.dot(covariance @ w)

4. Parameter - Fast re-solves for efficient frontier:
   target_param.set(new_value)  # Update without rebuilding

5. Clean constraints - Natural syntax:
   .subject_to(w.sum().eq(1))

The result: Cleaner code that runs faster.
""")

print("Demo complete! This model can be extended with:")
print("  • Transaction costs and turnover constraints")
print("  • Sector/geography exposure limits")
print("  • Conditional Value-at-Risk (CVaR) optimization")
print("  • Multi-period rebalancing strategies")
print("  • Factor-based risk models")
print("=" * 70)
