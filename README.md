# Optyx

**Optimization that reads like Python.**

[![PyPI](https://img.shields.io/pypi/v/optyx.svg)](https://pypi.org/project/optyx/)
[![Python 3.12+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/optyx-dev/optyx/actions/workflows/ci.yml/badge.svg)](https://github.com/optyx-dev/optyx/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://optyx-dev.github.io/optyx/)

📚 **[Documentation](https://optyx-dev.github.io/optyx/)** · 🚀 **[Quickstart](https://optyx-dev.github.io/optyx/getting-started/quickstart.html)** · 💡 **[Examples](https://optyx-dev.github.io/optyx/examples/portfolio.html)**

<table>
<tr>
<th>With Optyx</th>
<th>With SciPy</th>
</tr>
<tr>
<td>

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
# x=0.5, y=0.5
```

</td>
<td>

```python
from scipy.optimize import minimize
import numpy as np

def objective(v):
    return v[0]**2 + v[1]**2

def gradient(v):  # manual!
    return np.array([2*v[0], 2*v[1]])

result = minimize(
    objective, x0=[1, 1], jac=gradient,
    method='SLSQP',
    bounds=[(0, None), (0, None)],
    constraints={'type': 'ineq',
                 'fun': lambda v: v[0]+v[1]-1}
)
```

</td>
</tr>
</table>

Your optimization code should read like your math. With Optyx, `x + y >= 1` is exactly that—not a lambda buried in a constraint dictionary.

---

## Why Optyx?

Python has excellent optimization libraries. SciPy provides algorithms. CVXPY handles convex problems. Pyomo scales to industrial applications.

**Optyx takes a different path: radical simplicity.**

- **Write problems as you think them** — `x**2 + y**2` not `lambda v: v[0]**2 + v[1]**2`
- **Never compute gradients by hand** — symbolic autodiff handles derivatives
- **Skip solver configuration** — sensible defaults, automatic solver selection

### Being Honest

Optyx is young and opinionated. It's **not** a replacement for specialized tools:

| Need | Use Instead |
|------|-------------|
| Large-scale MILP with custom branching | Pyomo, OR-Tools, Gurobi |
| Convex guarantees | CVXPY |
| Maximum performance | Raw solver APIs |

Optyx does support MILP (via HiGHS), sparse LPs with 100k+ variables, and solver callbacks—but if you need industrial-grade MIP with cutting planes, a dedicated solver is the right choice.

---

## Installation

```bash
pip install optyx
```

Requires Python 3.12+, NumPy ≥2.0, SciPy ≥1.7.

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
# x=0.5, y=0.5, objective=0.5
```

### Portfolio Optimization

```python
from optyx import Variable, Problem

# Asset weights
tech = Variable("tech", lb=0, ub=1)
energy = Variable("energy", lb=0, ub=1)
finance = Variable("finance", lb=0, ub=1)

# Expected returns and risk (simplified)
returns = 0.12*tech + 0.08*energy + 0.10*finance
risk = tech**2 + energy**2 + finance**2  # variance proxy

solution = (
    Problem()
    .minimize(risk)
    .subject_to(returns >= 0.09)              # minimum return
    .subject_to((tech + energy + finance).eq(1))  # fully invested
    .solve()
)
```

### Autodiff Just Works

```python
from optyx import Variable
from optyx.core.autodiff import gradient

x = Variable("x")
f = x**3 + 2*x**2 - 5*x + 3

df = gradient(f, x)  # Symbolic: 3x² + 4x - 5
print(df.evaluate({"x": 2.0}))  # 15.0
```

### Mixed-Integer Programming

```python
from optyx import BinaryVariable, VectorVariable, Problem
import numpy as np

# Binary knapsack: select items to maximize value within weight limit
n = 5
x = VectorVariable("x", n, domain="binary")
values = np.array([10, 20, 15, 25, 30])
weights = np.array([5, 10, 8, 12, 15])

solution = (
    Problem()
    .maximize(x.dot(values))
    .subject_to(x.dot(weights) <= 30)
    .solve()
)
# Automatically routes to HiGHS MILP solver
```

---

## Features at a Glance

| Feature | Description |
|---------|-------------|
| **Natural syntax** | `x + y >= 1` instead of constraint dictionaries |
| **Automatic gradients** | Symbolic differentiation—no manual derivatives |
| **Smart solver selection** | HiGHS for LP/MILP, SLSQP/BFGS for NLP |
| **Mixed-integer programming** | `BinaryVariable`, `IntegerVariable`, automatic MILP routing |
| **Vector & matrix variables** | `VectorVariable`, `MatrixVariable`, `VariableDict` for scalable models |
| **Sparse LP support** | `subject_to(A @ x <= b)` with `as_matrix(..., storage="auto"|"dense"|"sparse")` — 100k+ variables |
| **Solver callbacks** | Monitor progress, enforce time limits, early termination |
| **LP format export** | `Problem.write("model.lp")` for interop with other solvers |
| **Solution serialization** | `to_json()` / `from_json()` for logging and auditing |
| **Fast re-solve** | Cached compilation + warm starts, up to 900x speedup |
| **Debuggable** | Inspect expression trees, understand your model |

See the [documentation](https://optyx-dev.github.io/optyx/) for the full API reference, tutorials, and real-world examples.

---

## What's Next

Optyx is actively evolving:

- **MIQP / MINLP support** — Quadratic and nonlinear MIP via native HiGHS or Gurobi
- **MPS format I/O** — Import and export MPS files for solver interop
- **More solvers** — IPOPT integration for large-scale NLP
- **Better debugging** — Infeasibility diagnostics and model inspection

See the [roadmap](https://optyx-dev.github.io/optyx/contributing.html) for details.

---

## Contributing

```bash
git clone https://github.com/optyx-dev/optyx.git
cd optyx
uv sync
uv run pytest
```

Contributions welcome! See our [contributing guide](https://optyx-dev.github.io/optyx/contributing.html).

---

## License

MIT
