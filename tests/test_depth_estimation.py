"""Tests for depth estimation and recursion utilities."""

import sys

import pytest

from optyx import VectorVariable, increased_recursion_limit
from optyx.core.autodiff import _estimate_tree_depth, gradient


class TestDepthEstimation:
    """Tests for _estimate_tree_depth function."""

    def test_left_skewed_tree_default(self) -> None:
        """Left-spine heuristic is accurate for left-skewed trees."""
        x = VectorVariable("x", 100)
        # Build left-skewed tree: ((x[0] + x[1]) + x[2]) + ...
        expr = x[0]
        for i in range(1, 100):
            expr = expr + x[i]

        depth = _estimate_tree_depth(expr)
        assert depth == 99  # n-1 additions

    def test_left_skewed_tree_full_traversal(self) -> None:
        """Full traversal matches left-spine for left-skewed trees."""
        x = VectorVariable("x", 50)
        expr = x[0]
        for i in range(1, 50):
            expr = expr + x[i]

        depth_fast = _estimate_tree_depth(expr)
        depth_full = _estimate_tree_depth(expr, full_traversal=True)
        assert depth_fast == depth_full == 49

    def test_right_skewed_tree_heuristic_underestimates(self) -> None:
        """Left-spine heuristic underestimates right-skewed trees."""
        x = VectorVariable("x", 50)
        # Build right-skewed tree: x[0] + (x[1] + (x[2] + ...))
        expr = x[49]
        for i in range(48, -1, -1):
            expr = x[i] + expr  # Note: x[i] is on LEFT, expr is on RIGHT

        depth_fast = _estimate_tree_depth(expr)
        # Left-spine heuristic follows left children, which are all leaves
        assert depth_fast <= 2  # Should be very low

    def test_right_skewed_tree_full_traversal_accurate(self) -> None:
        """Full traversal gives correct depth for right-skewed trees."""
        x = VectorVariable("x", 50)
        expr = x[49]
        for i in range(48, -1, -1):
            expr = x[i] + expr

        depth_full = _estimate_tree_depth(expr, full_traversal=True)
        assert depth_full == 49

    def test_balanced_tree(self) -> None:
        """Full traversal works for balanced trees."""
        x = VectorVariable("x", 8)
        # Build balanced tree manually
        # Level 0: x[0]+x[1], x[2]+x[3], x[4]+x[5], x[6]+x[7]
        level1 = [x[i] + x[i + 1] for i in range(0, 8, 2)]
        # Level 1: (x[0]+x[1])+(x[2]+x[3]), (x[4]+x[5])+(x[6]+x[7])
        level2 = [level1[i] + level1[i + 1] for i in range(0, 4, 2)]
        # Level 2: root
        root = level2[0] + level2[1]

        depth_full = _estimate_tree_depth(root, full_traversal=True)
        assert depth_full == 3  # log2(8) = 3

    def test_max_check_limits_search(self) -> None:
        """max_check parameter limits depth search."""
        x = VectorVariable("x", 200)
        expr = x[0]
        for i in range(1, 200):
            expr = expr + x[i]

        # With limit of 50, should return 50 even though tree is deeper
        depth = _estimate_tree_depth(expr, max_check=50)
        assert depth == 50

    def test_max_check_limits_full_traversal(self) -> None:
        """max_check also limits full traversal."""
        x = VectorVariable("x", 200)
        expr = x[0]
        for i in range(1, 200):
            expr = expr + x[i]

        depth = _estimate_tree_depth(expr, max_check=50, full_traversal=True)
        assert depth == 50

    def test_single_variable_depth_zero(self) -> None:
        """Single variable has depth 0."""
        x = VectorVariable("x", 10)
        depth = _estimate_tree_depth(x[0])
        assert depth == 0

    def test_unary_ops_increase_depth(self) -> None:
        """Unary operations increase depth."""
        x = VectorVariable("x", 10)
        expr = -(-(-x[0]))  # Three negations

        depth = _estimate_tree_depth(expr)
        assert depth == 3


class TestIncreasedRecursionLimit:
    """Tests for increased_recursion_limit context manager."""

    def test_limit_is_increased_inside_context(self) -> None:
        """Recursion limit is higher inside context."""
        original = sys.getrecursionlimit()

        with increased_recursion_limit(5000):
            inside = sys.getrecursionlimit()

        after = sys.getrecursionlimit()

        assert inside == 5000
        assert after == original

    def test_limit_restored_on_exception(self) -> None:
        """Limit is restored even if exception occurs."""
        original = sys.getrecursionlimit()

        with pytest.raises(ValueError):
            with increased_recursion_limit(5000):
                assert sys.getrecursionlimit() == 5000
                raise ValueError("Test exception")

        assert sys.getrecursionlimit() == original

    def test_default_limit_is_5000(self) -> None:
        """Default limit is 5000."""
        with increased_recursion_limit():
            assert sys.getrecursionlimit() == 5000

    def test_custom_limit(self) -> None:
        """Custom limit is respected."""
        with increased_recursion_limit(3000):
            assert sys.getrecursionlimit() == 3000


class TestDeepTreeGradient:
    """Integration tests for gradient on deep trees."""

    def test_iterative_gradient_on_deep_tree(self) -> None:
        """Gradient works on deep trees via automatic iterative fallback."""
        x = VectorVariable("x", 500)
        expr = x[0] ** 2
        for i in range(1, 500):
            expr = expr + x[i] ** 2

        # Should automatically use iterative algorithm
        grad = gradient(expr, x[0])
        # d/dx[0](x[0]² + ...) = 2*x[0]
        result = grad.evaluate({"x[0]": 3.0})
        assert result == pytest.approx(6.0)
