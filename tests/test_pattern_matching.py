import numpy as np
from optyx.core.expressions import Variable, Constant
from optyx.core.vectors import (
    VectorVariable,
    VectorSum,
    DotProduct,
    L1Norm,
    L2Norm,
    LinearCombination,
)
from optyx.core.matrices import QuadraticForm
from optyx.core.autodiff import (
    detect_vector_gradient_pattern,
    VectorExpressionPattern,
)


def test_detect_patterns():
    x = Variable("x")
    c = Constant(2.0)

    # Base cases
    assert detect_vector_gradient_pattern(x) == VectorExpressionPattern.COMPONENT
    assert detect_vector_gradient_pattern(c) == VectorExpressionPattern.CONSTANT

    # Scaled component
    scaled = c * x
    assert (
        detect_vector_gradient_pattern(scaled)
        == VectorExpressionPattern.SCALED_COMPONENT
    )

    # Vector expressions
    n = 10
    v = VectorVariable("v", n)
    coeffs = np.ones(n)

    # Sum
    s = VectorSum(v)
    assert detect_vector_gradient_pattern(s) == VectorExpressionPattern.SUM

    # Scaled Sum
    ss = c * s
    assert detect_vector_gradient_pattern(ss) == VectorExpressionPattern.SCALED_SUM

    # Linear Combination (c @ v)
    lc = LinearCombination(coeffs, v)
    assert detect_vector_gradient_pattern(lc) == VectorExpressionPattern.DOT_PRODUCT

    # Dot Product
    dp = DotProduct(v, v)  # Should be DOT_PRODUCT for now
    assert detect_vector_gradient_pattern(dp) == VectorExpressionPattern.DOT_PRODUCT

    # Norms
    l1 = L1Norm(v)
    assert detect_vector_gradient_pattern(l1) == VectorExpressionPattern.L1_NORM

    l2 = L2Norm(v)
    assert detect_vector_gradient_pattern(l2) == VectorExpressionPattern.L2_NORM

    # Quadratic Form
    Q = np.eye(n)
    qf = QuadraticForm(v, Q)
    assert detect_vector_gradient_pattern(qf) == VectorExpressionPattern.QUADRATIC_FORM
