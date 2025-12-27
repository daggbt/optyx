"""Tests for VectorVariable."""

import pytest

from optyx.core.vectors import VectorVariable
from optyx.core.expressions import Variable


class TestVectorVariableCreation:
    """Tests for VectorVariable creation."""

    def test_basic_creation(self):
        """VectorVariable creates the specified number of variables."""
        x = VectorVariable("x", 5)
        assert len(x) == 5
        assert x.size == 5
        assert x.name == "x"

    def test_large_vector(self):
        """VectorVariable can handle large sizes."""
        x = VectorVariable("x", 100)
        assert len(x) == 100

    def test_element_naming(self):
        """Elements are named with bracket notation."""
        x = VectorVariable("x", 5)
        assert x[0].name == "x[0]"
        assert x[1].name == "x[1]"
        assert x[4].name == "x[4]"

    def test_zero_size_raises(self):
        """Size must be positive."""
        with pytest.raises(ValueError, match="positive"):
            VectorVariable("x", 0)

    def test_negative_size_raises(self):
        """Negative size raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            VectorVariable("x", -5)


class TestVectorVariableBounds:
    """Tests for bounds propagation."""

    def test_lower_bound_propagates(self):
        """Lower bound applies to all elements."""
        x = VectorVariable("x", 5, lb=0)
        for v in x:
            assert v.lb == 0

    def test_upper_bound_propagates(self):
        """Upper bound applies to all elements."""
        x = VectorVariable("x", 5, ub=10)
        for v in x:
            assert v.ub == 10

    def test_both_bounds_propagate(self):
        """Both bounds apply to all elements."""
        x = VectorVariable("x", 5, lb=-1, ub=1)
        for v in x:
            assert v.lb == -1
            assert v.ub == 1

    def test_no_bounds_by_default(self):
        """No bounds by default."""
        x = VectorVariable("x", 5)
        assert x[0].lb is None
        assert x[0].ub is None


class TestVectorVariableDomain:
    """Tests for domain propagation."""

    def test_continuous_by_default(self):
        """Domain is continuous by default."""
        x = VectorVariable("x", 5)
        assert x.domain == "continuous"
        for v in x:
            assert v.domain == "continuous"

    def test_integer_domain(self):
        """Integer domain propagates to all elements."""
        x = VectorVariable("x", 5, domain="integer")
        for v in x:
            assert v.domain == "integer"

    def test_binary_domain(self):
        """Binary domain propagates and sets bounds."""
        x = VectorVariable("x", 5, domain="binary")
        for v in x:
            assert v.domain == "binary"
            assert v.lb == 0.0
            assert v.ub == 1.0


class TestVectorVariableIndexing:
    """Tests for indexing operations."""

    def test_positive_index(self):
        """Positive index returns correct element."""
        x = VectorVariable("x", 10)
        assert x[0].name == "x[0]"
        assert x[5].name == "x[5]"
        assert x[9].name == "x[9]"

    def test_negative_index(self):
        """Negative index works like Python lists."""
        x = VectorVariable("x", 10)
        assert x[-1].name == "x[9]"
        assert x[-2].name == "x[8]"
        assert x[-10].name == "x[0]"

    def test_index_returns_variable(self):
        """Indexing returns a Variable instance."""
        x = VectorVariable("x", 5)
        assert isinstance(x[0], Variable)

    def test_index_out_of_range(self):
        """Out of range index raises IndexError."""
        x = VectorVariable("x", 5)
        with pytest.raises(IndexError):
            x[5]
        with pytest.raises(IndexError):
            x[10]
        with pytest.raises(IndexError):
            x[-6]


class TestVectorVariableSlicing:
    """Tests for slicing operations."""

    def test_basic_slice(self):
        """Basic slicing returns new VectorVariable."""
        x = VectorVariable("x", 10)
        y = x[2:5]
        assert isinstance(y, VectorVariable)
        assert len(y) == 3

    def test_slice_preserves_variables(self):
        """Sliced VectorVariable references same Variable objects."""
        x = VectorVariable("x", 10)
        y = x[2:5]
        assert y[0].name == "x[2]"
        assert y[1].name == "x[3]"
        assert y[2].name == "x[4]"

    def test_slice_from_start(self):
        """Slice from start works."""
        x = VectorVariable("x", 10)
        y = x[:3]
        assert len(y) == 3
        assert y[0].name == "x[0]"

    def test_slice_to_end(self):
        """Slice to end works."""
        x = VectorVariable("x", 10)
        y = x[7:]
        assert len(y) == 3
        assert y[0].name == "x[7]"

    def test_slice_with_step(self):
        """Slice with step works."""
        x = VectorVariable("x", 10)
        y = x[::2]  # Every other element
        assert len(y) == 5
        assert y[0].name == "x[0]"
        assert y[1].name == "x[2]"

    def test_negative_slice(self):
        """Negative indices in slice work."""
        x = VectorVariable("x", 10)
        y = x[-3:]
        assert len(y) == 3
        assert y[0].name == "x[7]"

    def test_empty_slice_raises(self):
        """Empty slice raises IndexError."""
        x = VectorVariable("x", 10)
        with pytest.raises(IndexError, match="empty"):
            x[5:5]


class TestVectorVariableIteration:
    """Tests for iteration."""

    def test_iter(self):
        """Can iterate over VectorVariable."""
        x = VectorVariable("x", 5)
        names = [v.name for v in x]
        assert names == ["x[0]", "x[1]", "x[2]", "x[3]", "x[4]"]

    def test_list_conversion(self):
        """Can convert to list."""
        x = VectorVariable("x", 3)
        vars_list = list(x)
        assert len(vars_list) == 3
        assert all(isinstance(v, Variable) for v in vars_list)


class TestVectorVariableGetVariables:
    """Tests for get_variables method."""

    def test_get_variables(self):
        """get_variables returns list of all variables."""
        x = VectorVariable("x", 5)
        vars_list = x.get_variables()
        assert len(vars_list) == 5
        assert all(isinstance(v, Variable) for v in vars_list)
        assert [v.name for v in vars_list] == ["x[0]", "x[1]", "x[2]", "x[3]", "x[4]"]


class TestVectorVariableRepr:
    """Tests for string representation."""

    def test_basic_repr(self):
        """Basic repr shows name and size."""
        x = VectorVariable("x", 5)
        assert "x" in repr(x)
        assert "5" in repr(x)

    def test_repr_with_bounds(self):
        """Repr shows bounds when set."""
        x = VectorVariable("x", 5, lb=0, ub=10)
        r = repr(x)
        assert "lb=0" in r
        assert "ub=10" in r

    def test_repr_with_domain(self):
        """Repr shows non-continuous domain."""
        x = VectorVariable("x", 5, domain="binary")
        assert "binary" in repr(x)
