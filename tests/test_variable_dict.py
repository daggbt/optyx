"""Tests for VariableDict — dict-indexed variable collections."""

import numpy as np
import pytest

from optyx import (
    Variable,
    VariableDict,
    Problem,
)


class TestVariableDictCreation:
    """Test VariableDict construction and basic properties."""

    def test_basic_creation(self):
        foods = ["hamburger", "chicken", "pizza", "salad"]
        buy = VariableDict("buy", foods, lb=0)
        assert len(buy) == 4
        assert buy.name == "buy"

    def test_keys_preserved_order(self):
        keys = ["z", "a", "m", "b"]
        vd = VariableDict("x", keys)
        assert vd.keys() == ["z", "a", "m", "b"]

    def test_variable_naming(self):
        vd = VariableDict("buy", ["ham", "egg"])
        assert vd["ham"].name == "buy[ham]"
        assert vd["egg"].name == "buy[egg]"

    def test_scalar_bounds(self):
        vd = VariableDict("x", ["a", "b", "c"], lb=0, ub=10)
        for key in ["a", "b", "c"]:
            assert vd[key].lb == 0
            assert vd[key].ub == 10

    def test_per_key_bounds(self):
        lb = {"a": 0, "b": 1, "c": 2}
        ub = {"a": 10, "b": 20, "c": 30}
        vd = VariableDict("x", ["a", "b", "c"], lb=lb, ub=ub)
        assert vd["a"].lb == 0
        assert vd["a"].ub == 10
        assert vd["b"].lb == 1
        assert vd["b"].ub == 20
        assert vd["c"].lb == 2
        assert vd["c"].ub == 30

    def test_no_bounds(self):
        vd = VariableDict("x", ["a", "b"])
        assert vd["a"].lb is None
        assert vd["a"].ub is None

    def test_domain_continuous(self):
        vd = VariableDict("x", ["a", "b"])
        assert vd["a"].domain == "continuous"

    def test_domain_binary(self):
        vd = VariableDict("x", ["a", "b"], domain="binary")
        assert vd["a"].domain == "binary"
        assert vd["a"].lb == 0.0
        assert vd["a"].ub == 1.0

    def test_domain_integer(self):
        vd = VariableDict("x", ["a", "b"], domain="integer", lb=0, ub=5)
        assert vd["a"].domain == "integer"

    def test_empty_keys_raises(self):
        with pytest.raises(ValueError, match="at least one key"):
            VariableDict("x", [])

    def test_repr(self):
        vd = VariableDict("buy", ["ham", "egg"])
        r = repr(vd)
        assert "buy" in r
        assert "ham" in r
        assert "egg" in r


class TestVariableDictAccess:
    """Test indexing, iteration, and dict-like interface."""

    def test_getitem(self):
        vd = VariableDict("x", ["a", "b", "c"])
        assert isinstance(vd["a"], Variable)
        assert vd["a"].name == "x[a]"

    def test_getitem_missing_key(self):
        vd = VariableDict("x", ["a", "b"])
        with pytest.raises(KeyError, match="not found"):
            vd["z"]

    def test_contains(self):
        vd = VariableDict("x", ["a", "b", "c"])
        assert "a" in vd
        assert "z" not in vd

    def test_len(self):
        vd = VariableDict("x", ["a", "b", "c"])
        assert len(vd) == 3

    def test_iter(self):
        keys = ["x", "y", "z"]
        vd = VariableDict("v", keys)
        assert list(vd) == keys

    def test_keys(self):
        vd = VariableDict("v", ["a", "b", "c"])
        assert vd.keys() == ["a", "b", "c"]

    def test_values(self):
        vd = VariableDict("v", ["a", "b"])
        vals = vd.values()
        assert len(vals) == 2
        assert all(isinstance(v, Variable) for v in vals)
        assert vals[0].name == "v[a]"

    def test_items(self):
        vd = VariableDict("v", ["a", "b"])
        items = vd.items()
        assert len(items) == 2
        assert items[0] == ("a", vd["a"])
        assert items[1] == ("b", vd["b"])

    def test_get_variables(self):
        vd = VariableDict("v", ["a", "b", "c"])
        vars_ = vd.get_variables()
        assert len(vars_) == 3
        assert all(isinstance(v, Variable) for v in vars_)


class TestVariableDictExpressions:
    """Test sum() and prod() expression building."""

    def test_sum_all(self):
        vd = VariableDict("x", ["a", "b", "c"], lb=0)
        expr = vd.sum()
        # Evaluate the expression
        result = expr.evaluate({"x[a]": 1, "x[b]": 2, "x[c]": 3})
        assert result == 6.0

    def test_sum_subset(self):
        vd = VariableDict("x", ["a", "b", "c"], lb=0)
        expr = vd.sum(["a", "c"])
        result = expr.evaluate({"x[a]": 1, "x[b]": 999, "x[c]": 3})
        assert result == 4.0

    def test_sum_single_key(self):
        vd = VariableDict("x", ["a", "b", "c"])
        expr = vd.sum(["b"])
        # Should return the variable itself
        assert isinstance(expr, Variable)
        assert expr.name == "x[b]"

    def test_sum_invalid_key(self):
        vd = VariableDict("x", ["a", "b"])
        with pytest.raises(KeyError, match="not found"):
            vd.sum(["a", "z"])

    def test_prod_with_dict(self):
        vd = VariableDict("x", ["a", "b", "c"], lb=0)
        coeffs = {"a": 2.0, "b": 3.0, "c": 4.0}
        expr = vd.prod(coeffs)
        result = expr.evaluate({"x[a]": 1, "x[b]": 2, "x[c]": 3})
        assert result == pytest.approx(2.0 + 6.0 + 12.0)

    def test_prod_with_sequence(self):
        vd = VariableDict("x", ["a", "b", "c"], lb=0)
        coeffs = [2.0, 3.0, 4.0]
        expr = vd.prod(coeffs)
        result = expr.evaluate({"x[a]": 1, "x[b]": 2, "x[c]": 3})
        assert result == pytest.approx(20.0)

    def test_prod_partial_dict(self):
        """Coefficients dict can have fewer keys — missing keys get 0 coeff."""
        vd = VariableDict("x", ["a", "b", "c"], lb=0)
        coeffs = {"a": 5.0, "c": 10.0}  # b is missing
        expr = vd.prod(coeffs)
        result = expr.evaluate({"x[a]": 1, "x[b]": 999, "x[c]": 2})
        assert result == pytest.approx(25.0)

    def test_prod_wrong_length_sequence(self):
        vd = VariableDict("x", ["a", "b", "c"])
        with pytest.raises(ValueError, match="Expected 3"):
            vd.prod([1.0, 2.0])

    def test_prod_all_zero_coeffs(self):
        vd = VariableDict("x", ["a", "b"], lb=0)
        expr = vd.prod({"a": 0, "b": 0})
        result = expr.evaluate({"x[a]": 5, "x[b]": 10})
        assert result == 0.0


class TestVariableDictSolve:
    """Test VariableDict in end-to-end optimization problems."""

    def test_lp_diet_problem(self):
        """Classic diet problem using VariableDict."""
        foods = ["burger", "chicken", "pizza"]
        cost = {"burger": 2.49, "chicken": 2.89, "pizza": 1.99}
        protein = {"burger": 25, "chicken": 31, "pizza": 15}

        buy = VariableDict("buy", foods, lb=0, ub=10)
        prob = Problem(name="diet")

        # Minimize cost
        prob.minimize(buy.prod(cost))

        # Require at least 50g protein
        prob.subject_to(buy.prod(protein) >= 50)

        sol = prob.solve()
        assert sol.is_optimal

        # Check solution extraction
        result = sol[buy]
        assert isinstance(result, dict)
        assert set(result.keys()) == set(foods)
        assert all(v >= -1e-6 for v in result.values())

        # Protein constraint satisfied
        total_protein = sum(result[f] * protein[f] for f in foods)
        assert total_protein >= 50 - 1e-6

    def test_lp_with_sum_constraint(self):
        """Use sum() in constraints."""
        items = ["a", "b", "c"]
        x = VariableDict("x", items, lb=0, ub=5)

        prob = Problem(name="sum_test")
        prob.maximize(x.prod({"a": 3, "b": 2, "c": 1}))
        prob.subject_to(x.sum() <= 10)

        sol = prob.solve()
        assert sol.is_optimal
        result = sol[x]
        assert sum(result.values()) <= 10 + 1e-6

    def test_lp_partial_sum_constraint(self):
        """Use sum(subset) in constraints."""
        items = ["a", "b", "c", "d"]
        x = VariableDict("x", items, lb=0, ub=10)

        prob = Problem(name="partial_sum")
        prob.maximize(x.sum())
        # a + b <= 5
        prob.subject_to(x.sum(["a", "b"]) <= 5)
        # c + d <= 8
        prob.subject_to(x.sum(["c", "d"]) <= 8)

        sol = prob.solve()
        assert sol.is_optimal
        result = sol[x]
        assert result["a"] + result["b"] <= 5 + 1e-6
        assert result["c"] + result["d"] <= 8 + 1e-6
        assert sum(result.values()) == pytest.approx(13.0, abs=1e-4)

    def test_solution_getitem_variable_dict(self):
        """Solution[VariableDict] returns dict."""
        x = VariableDict("x", ["a", "b"], lb=0, ub=1)
        prob = Problem(name="test")
        prob.maximize(x.prod({"a": 1, "b": 2}))
        sol = prob.solve()
        assert sol.is_optimal

        result = sol[x]
        assert isinstance(result, dict)
        assert "a" in result
        assert "b" in result

    def test_solution_getitem_individual_var(self):
        """Solution[vd['key']] returns float for individual variable."""
        x = VariableDict("x", ["a", "b"], lb=0, ub=1)
        prob = Problem(name="test")
        prob.maximize(x.prod({"a": 1, "b": 2}))
        sol = prob.solve()

        val_a = sol[x["a"]]
        assert isinstance(val_a, float)

    def test_milp_with_variable_dict(self):
        """VariableDict with binary domain for MILP."""
        items = ["laptop", "phone", "tablet"]
        value = {"laptop": 1000, "phone": 500, "tablet": 300}
        weight = {"laptop": 3, "phone": 1, "tablet": 2}

        select = VariableDict("select", items, domain="binary")
        prob = Problem(name="knapsack")
        prob.maximize(select.prod(value))
        prob.subject_to(select.prod(weight) <= 4)

        sol = prob.solve()
        assert sol.is_optimal

        result = sol[select]
        # Should pick laptop + phone (value=1500, weight=4)
        # or phone + tablet (value=800, weight=3)
        total_weight = sum(result[i] * weight[i] for i in items)
        assert total_weight <= 4 + 1e-6

    def test_nlp_with_variable_dict(self):
        """VariableDict in a nonlinear problem."""
        dims = ["x", "y"]
        v = VariableDict("v", dims)

        prob = Problem(name="nlp_test")
        # Minimize x^2 + y^2
        prob.minimize(v["x"] ** 2 + v["y"] ** 2)
        # Subject to x + y >= 1
        prob.subject_to(v["x"] + v["y"] >= 1)

        sol = prob.solve(x0=np.array([1.0, 1.0]))
        assert sol.is_optimal
        result = sol[v]
        assert result["x"] == pytest.approx(0.5, abs=1e-4)
        assert result["y"] == pytest.approx(0.5, abs=1e-4)


class TestVariableDictEdgeCases:
    """Edge cases and error handling."""

    def test_single_key(self):
        vd = VariableDict("x", ["only"])
        assert len(vd) == 1
        assert vd["only"].name == "x[only]"

    def test_sum_single_element_dict(self):
        vd = VariableDict("x", ["only"])
        expr = vd.sum()
        assert isinstance(expr, Variable)

    def test_keys_with_spaces(self):
        vd = VariableDict("buy", ["ice cream", "hot dog"])
        assert vd["ice cream"].name == "buy[ice cream]"
        assert vd["hot dog"].name == "buy[hot dog]"

    def test_many_keys(self):
        keys = [f"item_{i}" for i in range(100)]
        vd = VariableDict("x", keys, lb=0)
        assert len(vd) == 100
        assert vd["item_50"].name == "x[item_50]"

    def test_prod_with_numpy_array(self):
        vd = VariableDict("x", ["a", "b", "c"], lb=0)
        coeffs = np.array([1.0, 2.0, 3.0])
        expr = vd.prod(coeffs)
        result = expr.evaluate({"x[a]": 1, "x[b]": 1, "x[c]": 1})
        assert result == pytest.approx(6.0)

    def test_variable_dict_variables_are_independent(self):
        """Each VariableDict creates unique variables."""
        vd1 = VariableDict("x", ["a", "b"])
        vd2 = VariableDict("y", ["a", "b"])
        assert vd1["a"].name != vd2["a"].name

    def test_mixed_in_constraint(self):
        """VariableDict variables can mix with regular Variables."""
        vd = VariableDict("x", ["a", "b"], lb=0, ub=10)
        z = Variable("z", lb=0, ub=10)

        prob = Problem(name="mixed")
        prob.maximize(vd["a"] + vd["b"] + z)
        prob.subject_to(vd["a"] + z <= 5)
        prob.subject_to(vd["b"] <= 3)

        sol = prob.solve()
        assert sol.is_optimal
        assert sol[z] + sol[vd["a"]] <= 5 + 1e-6
