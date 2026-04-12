"""Dictionary-indexed variable collections.

Provides VariableDict for modeling problems where variables are
naturally indexed by strings (product names, cities, etc.) rather
than integers.
"""

from __future__ import annotations

from typing import Iterator, Mapping, Sequence


from optyx.core.expressions import Constant, Expression, Variable


class VariableDict:
    """A dictionary-keyed collection of optimization variables.

    Variables are indexed by string keys (e.g., product names, cities)
    rather than integer indices. Supports weighted sums, partial sums,
    and dict-like iteration.

    Args:
        name: Base name for all variables. Individual variables are
            named ``"{name}[{key}]"``.
        keys: Sequence of string keys.
        lb: Lower bound — scalar (broadcast to all), or dict (per-key).
        ub: Upper bound — scalar (broadcast to all), or dict (per-key).
        domain: Variable domain — ``'continuous'``, ``'integer'``, or ``'binary'``.

    Example:
        >>> foods = ["hamburger", "chicken", "pizza", "salad"]
        >>> buy = VariableDict("buy", foods, lb=0)
        >>> buy["hamburger"]  # Variable("buy[hamburger]")
        >>> buy.sum()         # sum of all buy variables
        >>> buy.prod({"hamburger": 2.49, "chicken": 2.89, ...})  # weighted sum
    """

    __slots__ = ("name", "_keys", "_variables", "_key_to_var")
    __array_ufunc__ = None  # Tell NumPy to defer to Python's operators

    def __init__(
        self,
        name: str,
        keys: Sequence[str],
        lb: float | Mapping[str, float] | None = None,
        ub: float | Mapping[str, float] | None = None,
        domain: str = "continuous",
    ) -> None:
        if not keys:
            raise ValueError("VariableDict requires at least one key.")

        self.name = name
        self._keys = list(keys)
        self._variables: dict[str, Variable] = {}
        self._key_to_var: dict[str, Variable] = {}

        for key in self._keys:
            var_lb = lb[key] if isinstance(lb, Mapping) else lb
            var_ub = ub[key] if isinstance(ub, Mapping) else ub
            var = Variable(f"{name}[{key}]", lb=var_lb, ub=var_ub, domain=domain)
            self._variables[key] = var
            self._key_to_var[key] = var

    def __getitem__(self, key: str) -> Variable:
        """Get the variable for a given key.

        Args:
            key: The string key.

        Returns:
            The Variable associated with this key.

        Raises:
            KeyError: If key not found.
        """
        try:
            return self._variables[key]
        except KeyError:
            raise KeyError(
                f"Key {key!r} not found in VariableDict {self.name!r}. "
                f"Available keys: {self._keys}"
            ) from None

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in this VariableDict."""
        return key in self._variables

    def __len__(self) -> int:
        """Number of variables in this VariableDict."""
        return len(self._keys)

    def __iter__(self) -> Iterator[str]:
        """Iterate over keys."""
        return iter(self._keys)

    def __repr__(self) -> str:
        return f"VariableDict({self.name!r}, keys={self._keys})"

    def keys(self) -> list[str]:
        """Return the keys of this VariableDict."""
        return list(self._keys)

    def values(self) -> list[Variable]:
        """Return the variables of this VariableDict."""
        return [self._variables[k] for k in self._keys]

    def items(self) -> list[tuple[str, Variable]]:
        """Return (key, variable) pairs."""
        return [(k, self._variables[k]) for k in self._keys]

    def sum(self, keys: Sequence[str] | None = None) -> Expression:
        """Sum of variables, optionally over a subset of keys.

        Args:
            keys: If provided, sum only these keys. Otherwise sum all.

        Returns:
            Expression representing the sum.

        Example:
            >>> buy.sum()                          # sum of all
            >>> buy.sum(["hamburger", "chicken"])   # partial sum
        """
        subset = self._resolve_keys(keys)
        if len(subset) == 1:
            return self._variables[subset[0]]

        result: Expression = self._variables[subset[0]]
        for key in subset[1:]:
            result = result + self._variables[key]
        return result

    def prod(self, coefficients: Mapping[str, float] | Sequence[float]) -> Expression:
        """Weighted sum (inner product) of variables with coefficients.

        Args:
            coefficients: Either a dict mapping keys to coefficients,
                or a sequence of coefficients in key order.

        Returns:
            Expression representing the weighted sum.

        Example:
            >>> cost = {"hamburger": 2.49, "chicken": 2.89}
            >>> buy.prod(cost)  # 2.49*buy[hamburger] + 2.89*buy[chicken] + ...
        """
        if isinstance(coefficients, Mapping):
            coeff_map = coefficients
        else:
            coeff_list = list(coefficients)
            if len(coeff_list) != len(self._keys):
                raise ValueError(
                    f"Expected {len(self._keys)} coefficients, got {len(coeff_list)}."
                )
            coeff_map = dict(zip(self._keys, coeff_list))

        terms: list[Expression] = []
        for key in self._keys:
            if key in coeff_map:
                c = coeff_map[key]
                if c != 0:
                    terms.append(Constant(c) * self._variables[key])

        if not terms:
            return Constant(0.0)
        result = terms[0]
        for term in terms[1:]:
            result = result + term
        return result

    def get_variables(self) -> list[Variable]:
        """Return all Variable objects in key order."""
        return [self._variables[k] for k in self._keys]

    def _resolve_keys(self, keys: Sequence[str] | None) -> list[str]:
        """Resolve and validate a key subset."""
        if keys is None:
            return self._keys
        resolved = list(keys)
        for key in resolved:
            if key not in self._variables:
                raise KeyError(
                    f"Key {key!r} not found in VariableDict {self.name!r}. "
                    f"Available keys: {self._keys}"
                )
        return resolved
