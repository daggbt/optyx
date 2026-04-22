"""Modify and Re-solve: Adapting a Model to Changing Conditions.

Shows how to modify an optimization problem between solves without
rebuilding from scratch — add/remove constraints, update variable bounds,
and leverage warm starting for faster re-solves.

Scenario: A factory production plan is adjusted as operating conditions
change (new regulations, capacity upgrades, equipment maintenance).
"""

from optyx import Variable, Problem
from optyx.constraints import Constraint

a = Variable("a", lb=0, ub=100)
b = Variable("b", lb=0, ub=100)

prob = Problem(name="factory_production")
prob.maximize(5 * a + 8 * b)  # $5/unit A, $8/unit B

# Named constraints via Constraint() so we can remove them by name later
mh = Constraint(expr=(a + 2 * b - 120), sense="<=", name="machine_hours")
prob.subject_to(mh)

rm = Constraint(expr=(3 * a + 2 * b - 150), sense="<=", name="raw_material")
prob.subject_to(rm)

print("=" * 60)
print("MODIFY AND RE-SOLVE DEMO")
print("=" * 60)

print("\n--- Solve 1: Base case ---")
sol = prob.solve()
print(f"Status: {sol.status}")
print(f"Profit: ${sol.objective_value:.2f}")
print(f"Production: A={sol.values['a']:.1f}, B={sol.values['b']:.1f}")

print("\n--- Solve 2: New regulation limits product B to 40 units ---")
reg = Constraint(expr=(b - 40), sense="<=", name="regulation_b")  # Named for later removal
prob.subject_to(reg)
sol = prob.solve()
print(f"Status: {sol.status}")
print(f"Profit: ${sol.objective_value:.2f}")
print(f"Production: A={sol.values['a']:.1f}, B={sol.values['b']:.1f}")

print("\n--- Solve 3: Regulation lifted ---")
prob.remove_constraint("regulation_b")
sol = prob.solve()
print(f"Status: {sol.status}")
print(f"Profit: ${sol.objective_value:.2f}")
print(f"Production: A={sol.values['a']:.1f}, B={sol.values['b']:.1f}")

print("\n--- Solve 4: Extra machine hours — capacity from 120h to 200h ---")
prob.remove_constraint("machine_hours")  # Remove old constraint by name
mh2 = Constraint(expr=(a + 2 * b - 200), sense="<=", name="machine_hours")
prob.subject_to(mh2)
sol = prob.solve()
print(f"Status: {sol.status}")
print(f"Profit: ${sol.objective_value:.2f}")
print(f"Production: A={sol.values['a']:.1f}, B={sol.values['b']:.1f}")

print("\n--- Solve 5: Maintenance — product A line shut down ---")
a.lb = 0
a.ub = 0
sol = prob.solve()
print(f"Status: {sol.status}")
print(f"Profit: ${sol.objective_value:.2f}")
print(f"Production: A={sol.values['a']:.1f}, B={sol.values['b']:.1f}")

print("\n--- Solve 6: Maintenance complete — line A restored ---")
a.lb = 0
a.ub = 100
sol = prob.solve()
print(f"Status: {sol.status}")
print(f"Profit: ${sol.objective_value:.2f}")
print(f"Production: A={sol.values['a']:.1f}, B={sol.values['b']:.1f}")

# --- Warm start with a nonlinear problem ---
print("\n" + "=" * 60)
print("WARM START DEMO (NLP)")
print("=" * 60)

x = Variable("x", lb=0, ub=10)
y = Variable("y", lb=0, ub=10)

nlp = Problem(name="warm_start_demo")
nlp.minimize((x - 3) ** 2 + (y - 4) ** 2)
nlp.subject_to(x + y >= 5)

print("\n--- NLP Solve 1: Initial solve ---")
sol1 = nlp.solve(method="SLSQP")
print(f"Solution: x={sol1.values['x']:.4f}, y={sol1.values['y']:.4f}")
print(f"Objective: {sol1.objective_value:.6f}")

print("\n--- NLP Solve 2: Re-solve (warm start reuses previous solution) ---")
sol2 = nlp.solve(method="SLSQP")
print(f"Solution: x={sol2.values['x']:.4f}, y={sol2.values['y']:.4f}")
print(f"Iterations: {sol2.iterations}")

print("\n--- NLP Solve 3: Add constraint and re-solve ---")
nlp.subject_to(x >= 4)
sol3 = nlp.solve(method="SLSQP")
print(f"Solution: x={sol3.values['x']:.4f}, y={sol3.values['y']:.4f}")
print(f"Objective: {sol3.objective_value:.6f}")

print("\n--- NLP Solve 4: Remove constraint and re-solve ---")
nlp.remove_constraint(1)
sol4 = nlp.solve(method="SLSQP")
print(f"Solution: x={sol4.values['x']:.4f}, y={sol4.values['y']:.4f}")
print(f"Objective: {sol4.objective_value:.6f}")

print("\n--- Reset (forces cold start on next solve) ---")
nlp.reset()
sol5 = nlp.solve(method="SLSQP")
print(f"Solution: x={sol5.values['x']:.4f}, y={sol5.values['y']:.4f}")

print("\n" + "=" * 60)
print("All demos completed successfully!")
print("=" * 60)
