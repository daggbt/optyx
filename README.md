# optix

Symbolic optimization for people who hate writing gradients.

```python
from optix import Variable, Problem

x = Variable("x", lb=0)
y = Variable("y", lb=0)

prob = Problem()
prob.minimize(x**2 + y**2)
prob.subject_to(x + y >= 1)

solution = prob.solve()
print(solution.x)  # {'x': 0.5, 'y': 0.5}
```

## Install

```bash
pip install optix
```

## Status

Work in progress.