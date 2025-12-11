# optyx

**Symbolic optimization for people who hate writing gradients.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-161%20passing-brightgreen.svg)](tests/)

Write optimization problems in natural Python. Optyx handles the gradients, constraints, and solver plumbing for you.

```python
from optyx import Variable, Problem

x = Variable("x", lb=0)
y = Variable("y", lb=0)

# Define and solve in one fluent chain
solution = (
    Problem()
    .minimize(x**2 + y**2)
    .subject_to(x + y >= 1)
    .solve()
)

print(f"x* = {solution['x']:.4f}")  # 0.5000
print(f"y* = {solution['y']:.4f}")  # 0.5000
```

---

## Installation

```bash
pip install git+https://github.com/daggbt/optyx.git
```

Or for development:
```bash
git clone https://github.com/daggbt/optyx.git
cd optyx
uv sync  # or: pip install -e .
```

---

## Features

### ✅ Expression System
Build mathematical expressions with natural Python operators.

```python
from optyx import Variable, sin, exp, log

x = Variable("x", lb=0, ub=10)
y = Variable("y", lb=0)

f = 2*x**2 + 3*y**2 + sin(x*y) + exp(-x) * log(y + 1)
print(f.evaluate({"x": 1.5, "y": 2.5}))  # 22.957968
```

**Supported functions:** `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs_`, `tanh`, `sinh`, `cosh`

### ✅ Automatic Differentiation
Symbolic gradients via chain rule. No manual derivatives.

```python
from optyx import Variable
from optyx.core.autodiff import gradient, jacobian, hessian

x = Variable("x")
f = x**3 + 2*x**2 - 5*x + 3

df = gradient(f, x)  # 3x² + 4x - 5
print(df.evaluate({"x": 2.0}))  # 15.0
```

### ✅ Constraint System
Natural syntax for inequalities and equalities.

```python
from optyx import Variable, Problem

x = Variable("x")
y = Variable("y")

prob = (
    Problem()
    .minimize(x**2 + y**2)
    .subject_to(x + y >= 1)        # Inequality: x + y ≥ 1
    .subject_to(x <= 5)            # Inequality: x ≤ 5
    .subject_to((x - y).constraint_eq(0))  # Equality: x = y
)
```

### ✅ Problem Definition
Fluent API for building optimization problems.

```python
prob = (
    Problem(name="portfolio")
    .minimize(risk)
    .subject_to(total_weight == 1)
    .subject_to(returns >= target)
)

# Or step by step
prob = Problem()
prob.minimize(objective)
prob.subject_to(constraint1)
prob.subject_to(constraint2)
```

### ✅ Solver Integration
Solve with SciPy backends. Get structured results.

```python
solution = prob.solve(method="SLSQP")  # or "trust-constr"

print(solution.status)          # SolverStatus.OPTIMAL
print(solution.objective_value) # 0.5
print(solution["x"])            # Access optimal value by variable name
print(solution.iterations)      # 12
print(solution.solve_time)      # 0.003 seconds
```

---

## Quick Examples

### Constrained Quadratic
```python
from optyx import Variable, Problem

x = Variable("x", lb=0)
y = Variable("y", lb=0)

solution = (
    Problem()
    .minimize(x**2 + y**2)
    .subject_to(x + y >= 1)
    .solve()
)
# x* = 0.5, y* = 0.5, objective = 0.5
```

### Maximization
```python
solution = (
    Problem()
    .maximize(x + 2*y)
    .subject_to(x + y <= 4)
    .subject_to(x <= 2)
    .subject_to(y <= 3)
    .solve()
)
# x* = 1.0, y* = 3.0, objective = 7.0
```

### Rosenbrock (Unconstrained NLP)
```python
x = Variable("x")
y = Variable("y")

rosenbrock = (1 - x)**2 + 100*(y - x**2)**2

solution = Problem().minimize(rosenbrock).solve()
# x* ≈ 1.0, y* ≈ 1.0, objective ≈ 0
```

---

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `Variable(name, lb, ub)` | Decision variable with optional bounds |
| `Constant(value)` | Fixed numeric value in expressions |
| `Problem(name)` | Optimization problem container |
| `Constraint` | Inequality or equality constraint |
| `Solution` | Solver result with optimal values and metadata |

### Functions

| Function | Description |
|----------|-------------|
| `sin`, `cos`, `tan` | Trigonometric functions |
| `exp`, `log` | Exponential and natural logarithm |
| `sqrt`, `abs_` | Square root and absolute value |
| `sinh`, `cosh`, `tanh` | Hyperbolic functions |

### Autodiff

```python
from optyx.core.autodiff import gradient, jacobian, hessian

gradient(expr, var)           # First derivative
jacobian(exprs, vars)         # Matrix of first derivatives
hessian(expr, vars)           # Matrix of second derivatives
```

---

## Project Status

| Phase | Description | Status | Tests |
|-------|-------------|--------|-------|
| Phase 1 | Core Expressions | ✅ Complete | 46 |
| Phase 2 | Autodiff | ✅ Complete | 40 |
| Phase 3 | Constraints & Solvers | ✅ Complete | 75 |
| Phase 4 | Demos & Polish | 🔲 In Progress | - |

**Total: 161 tests passing**

See [ROADMAP.md](optyx-project/ROADMAP.md) for future plans (v2.0+).

---

## Examples

Run the demo scripts:

```bash
# Expression system and autodiff
uv run python examples/expressions_and_autodiff.py

# Constraints and solvers
uv run python examples/constraints_and_solvers.py
```

---

## Contributing

Contributions welcome! Please see [optyx-project/docs_plan.md](optyx-project/docs_plan.md) for documentation standards.

```bash
# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=optyx
```

---

## License

MIT