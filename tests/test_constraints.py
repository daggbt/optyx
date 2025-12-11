"""Tests for the constraint system."""

import pytest
import numpy as np

from optyx import Variable, Constant
from optyx.constraints import Constraint, _make_constraint


class TestConstraintCreation:
    """Tests for creating constraints."""
    
    def test_le_constraint_from_expression(self):
        x = Variable("x")
        c = x <= 5
        assert isinstance(c, Constraint)
        assert c.sense == "<="
    
    def test_ge_constraint_from_expression(self):
        x = Variable("x")
        c = x >= 0
        assert isinstance(c, Constraint)
        assert c.sense == ">="
    
    def test_eq_constraint_via_method(self):
        x = Variable("x")
        c = x.constraint_eq(5)
        assert isinstance(c, Constraint)
        assert c.sense == "=="
    
    def test_constraint_with_expression(self):
        x = Variable("x")
        y = Variable("y")
        c = x + y <= 10
        assert isinstance(c, Constraint)
        assert c.sense == "<="
    
    def test_constraint_normalization(self):
        """Constraint should be normalized to expr sense 0 form."""
        x = Variable("x")
        c = x <= 5  # Should become (x - 5) <= 0
        # At x=5, expr should be 0
        assert c.evaluate({"x": 5.0}) == pytest.approx(0.0)
        # At x=6, expr should be 1 (violated)
        assert c.evaluate({"x": 6.0}) == pytest.approx(1.0)
    
    def test_invalid_sense_raises(self):
        x = Variable("x")
        with pytest.raises(ValueError):
            Constraint(expr=x, sense="<")


class TestConstraintEvaluation:
    """Tests for evaluating constraints."""
    
    def test_le_constraint_satisfied(self):
        x = Variable("x")
        c = x <= 5
        assert c.is_satisfied({"x": 4.0})
        assert c.is_satisfied({"x": 5.0})
        assert not c.is_satisfied({"x": 6.0})
    
    def test_ge_constraint_satisfied(self):
        x = Variable("x")
        c = x >= 0
        assert c.is_satisfied({"x": 1.0})
        assert c.is_satisfied({"x": 0.0})
        assert not c.is_satisfied({"x": -1.0})
    
    def test_eq_constraint_satisfied(self):
        x = Variable("x")
        c = x.constraint_eq(5)
        assert c.is_satisfied({"x": 5.0})
        assert not c.is_satisfied({"x": 5.1})
    
    def test_le_violation(self):
        x = Variable("x")
        c = x <= 5
        assert c.violation({"x": 4.0}) == 0.0
        assert c.violation({"x": 5.0}) == 0.0
        assert c.violation({"x": 7.0}) == pytest.approx(2.0)
    
    def test_ge_violation(self):
        x = Variable("x")
        c = x >= 0
        assert c.violation({"x": 1.0}) == 0.0
        assert c.violation({"x": 0.0}) == 0.0
        assert c.violation({"x": -3.0}) == pytest.approx(3.0)
    
    def test_eq_violation(self):
        x = Variable("x")
        c = x.constraint_eq(5)
        assert c.violation({"x": 5.0}) == 0.0
        assert c.violation({"x": 7.0}) == pytest.approx(2.0)
        assert c.violation({"x": 3.0}) == pytest.approx(2.0)


class TestConstraintVariables:
    """Tests for variable extraction from constraints."""
    
    def test_single_variable(self):
        x = Variable("x")
        c = x <= 5
        assert c.get_variables() == {x}
    
    def test_multiple_variables(self):
        x = Variable("x")
        y = Variable("y")
        c = x + y <= 10
        assert c.get_variables() == {x, y}
    
    def test_complex_expression(self):
        x = Variable("x")
        y = Variable("y")
        z = Variable("z")
        c = x**2 + 2*y - z >= 0
        assert c.get_variables() == {x, y, z}


class TestConstraintRepr:
    """Tests for constraint representation."""
    
    def test_constraint_repr(self):
        x = Variable("x")
        c = x <= 5
        repr_str = repr(c)
        assert "Constraint" in repr_str
        assert "<=" in repr_str
