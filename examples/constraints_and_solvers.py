"""Example: Constraint System and Problem Solving (Phase 3)

Demonstrates:
- Creating optimization problems with natural syntax
- Adding constraints using <=, >=, and eq()
- Solving with different methods
- Accessing solution values
"""

from optyx import Variable, Problem

print("=" * 60)
print("OPTYX - Phase 3: Constraints & Solvers Demo")
print("=" * 60)

# =============================================================================
# Example 1: Simple Constrained Quadratic
# =============================================================================
print("\n📐 Example 1: Constrained Quadratic")
print("-" * 40)

x = Variable("x", lb=0)
y = Variable("y", lb=0)

# min x² + y² subject to x + y ≥ 1
prob = Problem()
prob.minimize(x**2 + y**2)
prob.subject_to(x + y >= 1)

print("Problem: min x² + y² s.t. x + y ≥ 1, x,y ≥ 0")
print(f"Variables: {[v.name for v in prob.variables]}")
print(f"Constraints: {prob.n_constraints}")

solution = prob.solve()
print(f"\nSolution: {solution.status.value}")
print(f"  x* = {solution['x']:.6f}")
print(f"  y* = {solution['y']:.6f}")
print(f"  Objective = {solution.objective_value:.6f}")
print(f"  Iterations: {solution.iterations}")
print(f"  Solve time: {solution.solve_time * 1000:.2f} ms")

# =============================================================================
# Example 2: Maximization Problem
# =============================================================================
print("\n📈 Example 2: Maximization")
print("-" * 40)

# max x + 2y s.t. x + y ≤ 4, x ≤ 2, y ≤ 3, x,y ≥ 0
prob2 = (
    Problem(name="linear_max")
    .maximize(x + 2 * y)
    .subject_to(x + y <= 4)
    .subject_to(x <= 2)
    .subject_to(y <= 3)
)

print("Problem: max x + 2y s.t. x + y ≤ 4, x ≤ 2, y ≤ 3")

sol2 = prob2.solve()
print(f"\nSolution: {sol2.status.value}")
print(f"  x* = {sol2['x']:.6f}")
print(f"  y* = {sol2['y']:.6f}")
print(f"  Objective = {sol2.objective_value:.6f}")

# =============================================================================
# Example 3: Equality Constraint
# =============================================================================
print("\n🔗 Example 3: Equality Constraint")
print("-" * 40)

# min x² + y² s.t. x + y = 2
a = Variable("a")
b = Variable("b")

prob3 = Problem().minimize(a**2 + b**2).subject_to((a + b).eq(2))

print("Problem: min a² + b² s.t. a + b = 2")

sol3 = prob3.solve()
print(f"\nSolution: {sol3.status.value}")
print(f"  a* = {sol3['a']:.6f}")
print(f"  b* = {sol3['b']:.6f}")
print(f"  Objective = {sol3.objective_value:.6f}")
print(f"  Check: a + b = {sol3['a'] + sol3['b']:.6f}")

# =============================================================================
# Example 4: Rosenbrock Function
# =============================================================================
print("\n🌹 Example 4: Rosenbrock Function")
print("-" * 40)

r_x = Variable("x")
r_y = Variable("y")

rosenbrock = (1 - r_x) ** 2 + 100 * (r_y - r_x**2) ** 2

prob4 = Problem().minimize(rosenbrock)
print("Problem: min (1-x)² + 100(y-x²)²")
print("Known optimum: (1, 1) with objective = 0")

sol4 = prob4.solve()
print(f"\nSolution: {sol4.status.value}")
print(f"  x* = {sol4['x']:.6f} (expected: 1.0)")
print(f"  y* = {sol4['y']:.6f} (expected: 1.0)")
print(f"  Objective = {sol4.objective_value:.2e} (expected: 0)")

# =============================================================================
# Example 5: Different Solver Methods
# =============================================================================
print("\n⚙️ Example 5: Solver Methods Comparison")
print("-" * 40)

test_prob = Problem().minimize(x**2 + y**2).subject_to(x + y >= 1)

for method in ["SLSQP", "trust-constr"]:
    sol = test_prob.solve(method=method)
    print(
        f"{method:15} → x={sol['x']:.4f}, y={sol['y']:.4f}, time={sol.solve_time * 1000:.2f}ms"
    )

print("\n" + "=" * 60)
print("✅ Phase 3 complete! Constraint system and solvers working.")
print("=" * 60)
