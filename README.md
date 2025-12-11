# optyx

Symbolic optimization for people who hate writing gradients.

```python
from optyx import Variable, sin, exp, log
from optyx.core.autodiff import gradient

x = Variable("x", lb=0, ub=10)
y = Variable("y", lb=0)

# Natural Python syntax for expressions
f = 2*x**2 + 3*y**2 + sin(x*y) + exp(-x) * log(y + 1)

# Automatic symbolic differentiation
df_dx = gradient(f, x)
df_dy = gradient(f, y)

# Evaluate at a point
print(f.evaluate({"x": 1.5, "y": 2.5}))  # 22.957968
print(df_dx.evaluate({"x": 1.5, "y": 2.5}))  # 3.669072
```

## Install

```bash
pip install git+https://github.com/daggbt/optyx.git
```

## Features

- **Expression Trees**: Build mathematical expressions with `+`, `-`, `*`, `/`, `**`
- **Transcendental Functions**: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `tanh`, `sinh`, `cosh`
- **Automatic Differentiation**: Symbolic gradients via chain rule
- **Jacobian & Hessian**: Compute first and second-order derivatives for systems
- **Compiled Evaluation**: 2-3x faster evaluation via NumPy code generation
- **Gradient Verification**: Validate symbolic gradients against numerical differentiation

## Status

Work in progress. Core expression system and autodiff complete. Constraint system and solvers coming next.