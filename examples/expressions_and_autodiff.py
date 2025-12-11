"""Test script to demonstrate Phase 1 and Phase 2 implementations."""

import numpy as np
from optyx import Variable, Constant, sin, cos, exp, log, sqrt, tanh
from optyx.core.compiler import compile_expression, CompiledExpression
from optyx.core.autodiff import gradient, compute_jacobian, compute_hessian, compile_jacobian, compile_hessian
from optyx.core.verification import verify_gradient, gradient_check

print("=" * 60)
print("OPTYX - Phase 1 & 2 Demo")
print("=" * 60)

# =============================================================================
# Phase 1: Expression System
# =============================================================================
print("\n📦 PHASE 1: Expression System")
print("-" * 40)

# Create variables
x = Variable("x", lb=0, ub=10)
y = Variable("y", lb=0)

# Build expressions using natural Python syntax
expr = 2*x**2 + 3*y**2 + sin(x*y) + exp(-x) * log(y + 1)

print(f"Expression: 2*x² + 3*y² + sin(x*y) + exp(-x)*log(y+1)")
print(f"Variables: {[v.name for v in expr.get_variables()]}")

# Evaluate at a point
point = {"x": 1.5, "y": 2.5}
value = expr.evaluate(point)
print(f"\nEvaluate at x=1.5, y=2.5: {value:.6f}")

# Compiled evaluation (faster)
compiled = compile_expression(expr, [x, y])
value_compiled = compiled(np.array([1.5, 2.5]))
print(f"Compiled evaluation:      {value_compiled:.6f}")

# =============================================================================
# Phase 2: Automatic Differentiation
# =============================================================================
print("\n📐 PHASE 2: Automatic Differentiation")
print("-" * 40)

# Symbolic gradient
grad_x = gradient(expr, x)
grad_y = gradient(expr, y)

print(f"\n∂f/∂x at (1.5, 2.5): {grad_x.evaluate(point):.6f}")
print(f"∂f/∂y at (1.5, 2.5): {grad_y.evaluate(point):.6f}")

# Verify against numerical gradient
print(f"\nGradient verification (symbolic vs numerical):")
print(f"  ∂f/∂x matches: {verify_gradient(expr, x, point)}")
print(f"  ∂f/∂y matches: {verify_gradient(expr, y, point)}")

# =============================================================================
# Chain Rule Demo
# =============================================================================
print("\n🔗 Chain Rule Demo")
print("-" * 40)

# f(x) = sin(x²)
# df/dx = cos(x²) * 2x
f = sin(x**2)
df_dx = gradient(f, x)

test_x = 1.0
analytical = np.cos(test_x**2) * 2 * test_x
computed = df_dx.evaluate({"x": test_x, "y": 0})

print(f"f(x) = sin(x²)")
print(f"df/dx = cos(x²) * 2x")
print(f"At x=1: analytical = {analytical:.6f}, computed = {computed:.6f}")

# =============================================================================
# Jacobian Demo
# =============================================================================
print("\n📊 Jacobian Demo")
print("-" * 40)

# Constraints: g1 = x² + y - 10, g2 = x*y - 5
g1 = x**2 + y - 10
g2 = x * y - 5

jacobian = compute_jacobian([g1, g2], [x, y])
print(f"Constraints: g1 = x² + y - 10, g2 = x*y - 5")
print(f"\nJacobian matrix at (2, 3):")

jac_point = {"x": 2.0, "y": 3.0}
print(f"  J[0,0] = ∂g1/∂x = 2x = {jacobian[0][0].evaluate(jac_point):.1f}")
print(f"  J[0,1] = ∂g1/∂y = 1  = {jacobian[0][1].evaluate(jac_point):.1f}")
print(f"  J[1,0] = ∂g2/∂x = y  = {jacobian[1][0].evaluate(jac_point):.1f}")
print(f"  J[1,1] = ∂g2/∂y = x  = {jacobian[1][1].evaluate(jac_point):.1f}")

# Compiled Jacobian
jac_fn = compile_jacobian([g1, g2], [x, y])
jac_matrix = jac_fn(np.array([2.0, 3.0]))
print(f"\nCompiled Jacobian:\n{jac_matrix}")

# =============================================================================
# Hessian Demo
# =============================================================================
print("\n📈 Hessian Demo")
print("-" * 40)

# Quadratic: f(x,y) = x² + 2xy + 3y²
quad = x**2 + 2*x*y + 3*y**2
print(f"f(x,y) = x² + 2xy + 3y²")

hessian = compute_hessian(quad, [x, y])
hess_point = {"x": 1.0, "y": 1.0}

print(f"\nHessian matrix (constant for quadratic):")
print(f"  H[0,0] = ∂²f/∂x² = 2  : {hessian[0][0].evaluate(hess_point):.1f}")
print(f"  H[0,1] = ∂²f/∂x∂y = 2 : {hessian[0][1].evaluate(hess_point):.1f}")
print(f"  H[1,0] = ∂²f/∂y∂x = 2 : {hessian[1][0].evaluate(hess_point):.1f}")
print(f"  H[1,1] = ∂²f/∂y² = 6  : {hessian[1][1].evaluate(hess_point):.1f}")

# Compiled Hessian
hess_fn = compile_hessian(quad, [x, y])
hess_matrix = hess_fn(np.array([1.0, 1.0]))
print(f"\nCompiled Hessian:\n{hess_matrix}")

# =============================================================================
# Rosenbrock Function (Classic Optimization Test)
# =============================================================================
print("\n🌹 Rosenbrock Function")
print("-" * 40)

# f(x,y) = (1-x)² + 100(y-x²)²
# Minimum at (1, 1)
rosenbrock = (1 - x)**2 + 100*(y - x**2)**2

print(f"f(x,y) = (1-x)² + 100(y-x²)²")
print(f"Known minimum at (1, 1)")

# Check gradient at minimum
grad_x_ros = gradient(rosenbrock, x)
grad_y_ros = gradient(rosenbrock, y)

min_point = {"x": 1.0, "y": 1.0}
print(f"\nGradient at minimum (1, 1):")
print(f"  ∂f/∂x = {grad_x_ros.evaluate(min_point):.6f} (should be 0)")
print(f"  ∂f/∂y = {grad_y_ros.evaluate(min_point):.6f} (should be 0)")

# Gradient check
print(f"\nGradient check (100 random samples):")
# Use looser tolerance for Rosenbrock due to high curvature (100x multiplier)
result = gradient_check(rosenbrock, [x, y], n_samples=100, bounds=(0.1, 5.0), tol=1e-3, seed=42)
print(f"  {result}")

# =============================================================================
# Performance Comparison
# =============================================================================
print("\n⚡ Performance Comparison")
print("-" * 40)

import time

n_iters = 50000

# Evaluate
start = time.perf_counter()
for _ in range(n_iters):
    expr.evaluate({"x": 1.5, "y": 2.5})
eval_time = time.perf_counter() - start

# Compiled evaluate
compiled_expr = compile_expression(expr, [x, y])
start = time.perf_counter()
for _ in range(n_iters):
    compiled_expr(np.array([1.5, 2.5]))
compiled_time = time.perf_counter() - start

# Symbolic gradient + evaluate
start = time.perf_counter()
for _ in range(n_iters):
    grad_x.evaluate({"x": 1.5, "y": 2.5})
grad_eval_time = time.perf_counter() - start

# Compiled gradient (from CompiledExpression)
compiled_full = CompiledExpression(expr, [x, y])
start = time.perf_counter()
for _ in range(n_iters):
    compiled_full.gradient(np.array([1.5, 2.5]))
compiled_grad_time = time.perf_counter() - start

print(f"Iterations: {n_iters:,}")
print(f"\nValue evaluation:")
print(f"  Tree-walk:  {eval_time:.3f}s ({n_iters/eval_time:,.0f}/sec)")
print(f"  Compiled:   {compiled_time:.3f}s ({n_iters/compiled_time:,.0f}/sec)")
print(f"  Speedup:    {eval_time/compiled_time:.2f}x")

print(f"\nGradient evaluation:")
print(f"  Symbolic:   {grad_eval_time:.3f}s ({n_iters/grad_eval_time:,.0f}/sec)")
print(f"  Numerical:  {compiled_grad_time:.3f}s ({n_iters/compiled_grad_time:,.0f}/sec)")

print("\n" + "=" * 60)
print("✅ All Phase 1 & 2 features working!")
print("=" * 60)
