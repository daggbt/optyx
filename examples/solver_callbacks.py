"""Solver progress callbacks and time limits.

Demonstrates how to monitor solver progress during optimization and
apply time limits for long-running solves.
"""

from optyx import Problem, SolverProgress, VectorVariable

# --- Problem setup: Rosenbrock in 10 dimensions ---
n = 10
v = VectorVariable("v", n, lb=-5, ub=5)
prob = Problem("rosenbrock")

# Vectorized Rosenbrock using VectorVariable slicing
v_head = v[:-1]  # first n-1 elements
v_tail = v[1:]   # last  n-1 elements
obj = ((1 - v_head) ** 2 + 100 * (v_tail - v_head ** 2) ** 2).sum()
prob.minimize(obj)


# --- Example 1: Logging progress ---
print("=== Example 1: Logging solver progress ===\n")


def log_progress(p: SolverProgress) -> None:
    print(
        f"  iter {p.iteration:3d}  |  obj {p.objective_value:12.4f}"
        f"  |  violation {p.constraint_violation:.2e}"
        f"  |  time {p.elapsed_time:.3f}s"
    )


sol = prob.solve(method="SLSQP", callback=log_progress)
print(f"\nStatus: {sol.status.value}  |  Objective: {sol.objective_value:.6f}\n")


# --- Example 2: Early termination via callback ---
print("=== Example 2: Stop when objective < threshold ===\n")

THRESHOLD = 1.0


def stop_when_good_enough(p: SolverProgress) -> bool:
    if p.objective_value < THRESHOLD:
        print(f"  Objective {p.objective_value:.4f} < {THRESHOLD} — stopping early")
        return True  # signal termination
    return False


sol = prob.solve(method="SLSQP", callback=stop_when_good_enough)
print(f"Status: {sol.status.value}  |  Objective: {sol.objective_value:.6f}\n")


# --- Example 3: Time limit ---
print("=== Example 3: Solve with a 0.01 s time limit ===\n")

sol = prob.solve(method="SLSQP", time_limit=0.01)
print(f"Status: {sol.status.value}  |  Objective: {sol.objective_value:.6f}")
print(f"Solve time: {sol.solve_time:.4f}s  |  Iterations: {sol.iterations}\n")


# --- Example 4: Combining callback and time limit ---
print("=== Example 4: Callback + time limit together ===\n")

history: list[float] = []


def record_objective(p: SolverProgress) -> None:
    history.append(p.objective_value)


sol = prob.solve(method="SLSQP", callback=record_objective, time_limit=0.05)
print(f"Status: {sol.status.value}  |  Objective: {sol.objective_value:.6f}")
print(f"Recorded {len(history)} objective snapshots")
