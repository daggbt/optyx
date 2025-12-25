# optyx

**Symbolic optimization for people who hate writing gradients.**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://daggbt.github.io/optyx/)

Write optimization problems in natural Python. Optyx handles the gradients, constraints, and solver plumbing for you.

📚 **[Documentation](https://daggbt.github.io/optyx/)** · 🚀 **[Quickstart](https://daggbt.github.io/optyx/getting-started/quickstart.html)** · 📖 **[Tutorials](https://daggbt.github.io/optyx/tutorials/basic-optimization.html)** · 💡 **[Examples](https://daggbt.github.io/optyx/examples/portfolio.html)** · 📘 **[API Reference](https://daggbt.github.io/optyx/api/expressions.html)**

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

### For Users

```bash
pip install optyx
```

### Requirements

- **Python 3.12+**
- **NumPy** ≥ 2.0
- **SciPy** ≥ 1.6.0 (≥ 1.11 recommended for best LP performance)

### For Developers

```bash
git clone https://github.com/daggbt/optyx.git
cd optyx
uv sync  # or: pip install -e .
```

---

## Why Optyx?

| Feature | Optyx | SciPy | CVXPY | Pyomo |
|---------|-------|-------|-------|-------|
| **Natural Python syntax** | ✅ `x + y >= 1` | ❌ Manual arrays | ⚠️ DSL-like | ⚠️ Verbose |
| **Automatic gradients** | ✅ Built-in | ❌ Manual/finite diff | ✅ For convex | ❌ Manual |
| **No solver install** | ✅ Uses SciPy | ✅ | ❌ Needs solvers | ❌ Needs solvers |
| **Inspect expressions** | ✅ Symbolic trees | ❌ | ⚠️ Limited | ⚠️ Limited |
| **Fast re-optimization** | ✅ Fast | ✅ | ✅ | ⚠️ Slower |
| **Learning curve** | Low | Medium | Medium | High |

**Optyx is ideal when you want:**
- Clean, readable optimization code without boilerplate
- Automatic differentiation without thinking about it *(gradients, Jacobians, and Hessians computed symbolically)*
- Quick prototyping with SciPy solvers (no installation headaches)
- Debuggable symbolic expressions you can inspect

**Consider alternatives when you need:**
- Convex optimization guarantees → CVXPY
- Mixed-integer programming → PuLP, Pyomo, or Gurobi
- Large-scale industrial optimization → Pyomo + commercial solvers

---

## Use Cases

Optyx excels at **nonlinear programming (NLP)** problems where you want automatic gradients and clean syntax:

### Resource Allocation & Scheduling
- Production scheduling with capacity constraints
- Multi-period planning with NPV optimization
- Equipment assignment and dispatch optimization

### Infrastructure Asset Management
- Road maintenance budget optimization
- Non-linear deterioration and satisfaction models
- Multi-period rehabilitation planning

### Portfolio & Risk Optimization
- Mean-variance portfolio optimization
- Risk-constrained return maximization
- Efficient frontier computation
- Scenario analysis and rebalancing

### Engineering Design
- Parameter optimization with physical constraints
- Multi-objective trade-off analysis
- Sensitivity and what-if analysis

### Operations Research
- Supply chain optimization
- Blending and mixing problems
- Facility location and routing

*See the [examples/](examples/) folder for complete working demos in mining operations, infrastructure asset management, and quantitative finance.*

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

**Supported functions:**

| Category | Functions |
|----------|-----------|
| Trigonometric | `sin`, `cos`, `tan` |
| Inverse Trig | `asin`, `acos`, `atan` |
| Exponential | `exp`, `log`, `log2`, `log10`, `sqrt` |
| Hyperbolic | `sinh`, `cosh`, `tanh` |
| Inverse Hyperbolic | `asinh`, `acosh`, `atanh` |
| Other | `abs_` |

### ✅ Automatic Differentiation
Symbolic gradients via chain rule. No manual derivatives.

```python
from optyx import Variable
from optyx.core.autodiff import gradient

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
    .subject_to((x - y).eq(0))     # Equality: x = y
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
Solve with SciPy backends. Optyx automatically selects the best solver for your problem.

```python
solution = prob.solve()  # Auto-selects: linprog for LP, SLSQP/BFGS for NLP

print(solution.status)          # SolverStatus.OPTIMAL
print(solution.objective_value) # 0.5
print(solution["x"])            # Access optimal value by variable name
print(solution.iterations)      # 12
print(solution.solve_time)      # 0.003 seconds
```

**Automatic solver selection** picks the best method based on problem structure:
- Linear problems → HiGHS LP solver (fastest)
- Unconstrained → BFGS or Nelder-Mead
- With constraints → SLSQP or trust-constr

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
| `Variable(name, lb, ub, domain)` | Decision variable with optional bounds and domain |
| `Constant(value)` | Fixed numeric value in expressions |
| `Problem(name)` | Optimization problem container |
| `Constraint` | Inequality or equality constraint |
| `Solution` | Solver result with optimal values and metadata |

> **Note:** The `domain` parameter accepts `"continuous"` (default), `"integer"`, or `"binary"`. Integer and binary domains are **relaxed to continuous** by the SciPy solver backend (a warning is emitted). For true mixed-integer programming, use PuLP or Pyomo.

### Functions

| Category | Functions |
|----------|-----------|
| Trigonometric | `sin`, `cos`, `tan` |
| Inverse Trig | `asin`, `acos`, `atan` |
| Exponential | `exp`, `log`, `log2`, `log10`, `sqrt` |
| Hyperbolic | `sinh`, `cosh`, `tanh` |
| Inverse Hyperbolic | `asinh`, `acosh`, `atanh` |
| Other | `abs_` |

### Autodiff

```python
from optyx.core.autodiff import gradient, jacobian, hessian

gradient(expr, var)           # First derivative
jacobian(exprs, vars)         # Matrix of first derivatives
hessian(expr, vars)           # Matrix of second derivatives
```

---

## Examples

Run the demo scripts:

```bash
# Expression system and autodiff
uv run python examples/expressions_and_autodiff.py

# Constraints and solvers
uv run python examples/constraints_and_solvers.py

# Mining production scheduling (NPV optimization)
uv run python examples/mine_scheduling.py

# Fleet dispatch optimization (truck-shovel assignment)
uv run python examples/fleet_dispatch.py

# Portfolio optimization (mean-variance, efficient frontier)
uv run python examples/portfolio_optimization.py

# Road maintenance budget optimization
uv run python examples/road_maintenance.py
```

### Domain-Specific Demos

#### 🏗️ Mining Production Scheduling
Multi-period pit production planning with grade blending and capacity constraints.
```python
# 9 blocks, 4 periods, NPV maximization
# Precedence constraints, grade blending (1.0-2.0% Cu)
solution = prob.solve()  # NPV: $50,222k
```

#### 🚚 Fleet Dispatch Optimization
Real-time truck-shovel assignment for maximum throughput.
```python
# 12 trucks, 4 shovels, 8,055 t/h optimal
# Re-optimization in 3ms after equipment breakdown
```

#### 📈 Portfolio Optimization
Mean-variance optimization with efficient frontier.
```python
# 6 commodities, 12.47% return, 20% volatility
# Automatic rebalancing on price shocks
```

#### 🛣️ Road Maintenance Optimization
Budget allocation with non-linear deterioration and satisfaction models.
```python
# 5 roads, $10M budget, nested sigmoid/exponential objective
# Auto-diff computes gradients through complex chain rule
solution = prob.solve()  # I-95: $5M, Main St: $0 (optimal)
```

---

## Benchmarks

Optyx includes a comprehensive benchmark suite comparing performance against SciPy. All benchmarks use numpy vectorization (`@`, `np.sum`) for optimal performance.

### Performance Summary

| Metric | Result | Notes |
|--------|--------|-------|
| **LP Overhead** | ~0.94-1.15x | Near-parity with raw SciPy linprog |
| **NLP Overhead** | ~1.4-2.2x | Autodiff cost increases with problem size |
| **Cache Speedup** | 2x-900x | Dramatic improvement on repeated solves |
| **Rosenbrock NLP** | 0.83x | Exact gradients help convergence |

### Scaling Analysis

<p align="center">
  <img src="benchmarks/results/lp_scaling_comparison.png" width="90%" alt="LP Scaling: Optyx vs SciPy"/>
</p>

**LP Performance**: Optyx achieves near-parity with raw SciPy `linprog`. The overhead ratio stays between 0.94x-1.15x across all problem sizes (10-500 variables).

<p align="center">
  <img src="benchmarks/results/nlp_quadratic_scaling.png" width="90%" alt="NLP Scaling: Optyx vs SciPy"/>
</p>

**NLP Performance**: Autodiff adds ~1.4-2.2x overhead, but provides exact gradients (no finite difference errors) and cleaner code.

### Cache Benefit

<p align="center">
  <img src="benchmarks/results/lp_cache_benefit.png" width="90%" alt="Cache Benefit Analysis"/>
</p>

**Repeated solves are dramatically faster**: The first solve compiles the problem; subsequent solves reuse the cached structure. Speedups range from 2x (small problems) to **900x** (large problems).

### Overhead by Problem Type

<p align="center">
  <img src="benchmarks/results/overhead_breakdown.png" width="70%" alt="Overhead Breakdown"/>
</p>

- **LP problems**: Near parity (~1.1x overhead)
- **NLP problems**: ~1.4-2.2x overhead (autodiff cost)
- **Rosenbrock**: 0.83x - exact gradients help complex optimization landscapes

### Running Benchmarks

```bash
# Run all benchmarks (tests only)
uv run pytest benchmarks/ -v

# Generate performance plots
uv run python benchmarks/run_benchmarks.py

# Run specific category
uv run pytest benchmarks/performance/ -v
```

See the [Benchmarks Documentation](https://daggbt.github.io/optyx/benchmarks.html) for detailed methodology and results.

---

## What's Next

**Optyx is actively evolving.** Here's where we're heading:

- ✅ **Smarter modeling** — Automatic problem classification and solver selection *(completed in v1.1)*
- ✅ **Production-ready** — Caching for fast repeated solves *(completed in v1.1)*
- **Larger problems** — Support for vector and matrix variables to handle optimization with thousands of decision variables
- **Faster execution** — JIT-compiled backends for significant performance improvements on complex models
- **More solvers** — Integration with industry-standard solvers like IPOPT for large-scale nonlinear optimization
- **Better debugging** — Infeasibility diagnostics, constraint violation reports, and model inspection tools

---

## Contributing

Contributions welcome! Please see the documentation site for contribution and documentation standards: https://daggbt.github.io/optyx/contributing.html

```bash
# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=optyx
```

---

## License

MIT