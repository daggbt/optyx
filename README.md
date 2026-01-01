# Optyx

**Optimization that reads like Python.**

[![PyPI](https://img.shields.io/pypi/v/optyx.svg)](https://pypi.org/project/optyx/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://github.com/daggbt/optyx/actions/workflows/ci.yml/badge.svg)](https://github.com/daggbt/optyx/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-online-blue.svg)](https://daggbt.github.io/optyx/)

📚 **[Documentation](https://daggbt.github.io/optyx/)** · 🚀 **[Quickstart](https://daggbt.github.io/optyx/getting-started/quickstart.html)** · 💡 **[Examples](https://daggbt.github.io/optyx/examples/portfolio.html)**

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
| MILP at scale | Pyomo, OR-Tools, Gurobi |
| Convex guarantees | CVXPY |
| Maximum performance | Raw solver APIs |

But if you want readable optimization code that just works for most problems, Optyx might be for you.

---

## Installation

```bash
pip install optyx
```

Requires Python 3.12+, NumPy ≥2.0, SciPy ≥1.6.

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

---

## Features at a Glance

| Feature | Description |
|---------|-------------|
| **Natural syntax** | `x + y >= 1` instead of constraint dictionaries |
| **Automatic gradients** | Symbolic differentiation—no manual derivatives |
| **Smart solver selection** | HiGHS for LP, SLSQP/BFGS for NLP |
| **Fast re-solve** | Cached compilation, up to 900x speedup |
| **Debuggable** | Inspect expression trees, understand your model |

See the [documentation](https://daggbt.github.io/optyx/) for the full API reference, tutorials, and real-world examples.

---

## What's Next

Optyx is actively evolving:

- **Vector/Matrix variables** — Handle thousands of decision variables cleanly
- **JIT compilation** — Faster execution for complex models  
- **More solvers** — IPOPT integration for large-scale NLP
- **Better debugging** — Infeasibility diagnostics and model inspection

See the [roadmap](https://daggbt.github.io/optyx/contributing.html) for details.

---

## Contributing

```bash
git clone https://github.com/daggbt/optyx.git
cd optyx
uv sync
uv run pytest
```

Contributions welcome! See our [contributing guide](https://daggbt.github.io/optyx/contributing.html).

---

## License

MIT
