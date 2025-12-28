"""Matrix variables for optimization problems.

This module provides MatrixVariable for representing 2D matrices of decision variables,
enabling natural syntax like `A = MatrixVariable("A", 3, 4)` with 2D indexing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator, overload

import numpy as np

from optyx.core.expressions import Variable
from optyx.core.vectors import VectorVariable, DomainType

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MatrixVariable:
    """A 2D matrix of optimization variables.

    MatrixVariable creates and manages a 2D collection of scalar Variable instances,
    providing natural 2D indexing, row/column slicing, and transpose views.

    Args:
        name: Base name for the matrix. Elements are named "{name}[i,j]".
        rows: Number of rows.
        cols: Number of columns.
        lb: Lower bound applied to all elements (None for unbounded).
        ub: Upper bound applied to all elements (None for unbounded).
        domain: Variable type for all elements - 'continuous', 'integer', or 'binary'.
        symmetric: If True, enforces A[i,j] == A[j,i] (must be square).

    Example:
        >>> A = MatrixVariable("A", 3, 4)
        >>> A[0, 0]  # Variable named "A[0,0]"
        >>> A[1, :]  # VectorVariable with 4 elements (row 1)
        >>> A[:, 2]  # VectorVariable with 3 elements (column 2)
        >>> A.shape  # (3, 4)
    """

    __slots__ = (
        "name",
        "rows",
        "cols",
        "lb",
        "ub",
        "domain",
        "symmetric",
        "_variables",
        "_is_transpose",
    )

    # Declare types for slots (helps type checkers)
    name: str
    rows: int
    cols: int
    lb: float | None
    ub: float | None
    domain: DomainType
    symmetric: bool
    _variables: list[list[Variable]]
    _is_transpose: bool

    # Tell NumPy to defer to our operators
    __array_ufunc__ = None

    def __init__(
        self,
        name: str,
        rows: int,
        cols: int,
        lb: float | None = None,
        ub: float | None = None,
        domain: DomainType = "continuous",
        symmetric: bool = False,
    ) -> None:
        if rows <= 0:
            raise ValueError(f"Rows must be positive, got {rows}")
        if cols <= 0:
            raise ValueError(f"Cols must be positive, got {cols}")
        if symmetric and rows != cols:
            raise ValueError(f"Symmetric matrix must be square, got {rows}x{cols}")

        self.name = name
        self.rows = rows
        self.cols = cols
        self.lb = lb
        self.ub = ub
        self.domain = domain
        self.symmetric = symmetric
        self._is_transpose = False

        # Create 2D array of variables
        self._variables: list[list[Variable]] = []
        for i in range(rows):
            row: list[Variable] = []
            for j in range(cols):
                if symmetric and j < i:
                    # For symmetric matrix, reuse variable from upper triangle
                    row.append(self._variables[j][i])
                else:
                    row.append(
                        Variable(f"{name}[{i},{j}]", lb=lb, ub=ub, domain=domain)
                    )
            self._variables.append(row)

    @property
    def shape(self) -> tuple[int, int]:
        """Return the shape of the matrix as (rows, cols)."""
        return (self.rows, self.cols)

    @property
    def T(self) -> MatrixVariable:
        """Return a transpose view of the matrix.

        The transpose shares the same underlying variables but swaps
        row and column indexing.

        Example:
            >>> A = MatrixVariable("A", 3, 4)
            >>> A.T.shape  # (4, 3)
            >>> A.T[0, 1] is A[1, 0]  # True
        """
        return MatrixVariable._transpose_view(self)

    @classmethod
    def _transpose_view(cls, original: MatrixVariable) -> MatrixVariable:
        """Create a transpose view of a matrix (internal)."""
        instance = object.__new__(cls)
        instance.name = f"{original.name}.T"
        instance.rows = original.cols
        instance.cols = original.rows
        instance.lb = original.lb
        instance.ub = original.ub
        instance.domain = original.domain
        instance.symmetric = original.symmetric
        instance._is_transpose = not original._is_transpose
        # Transpose the variable array
        instance._variables = [
            [original._variables[j][i] for j in range(original.rows)]
            for i in range(original.cols)
        ]
        return instance

    @overload
    def __getitem__(self, key: tuple[int, int]) -> Variable: ...

    @overload
    def __getitem__(self, key: tuple[int, slice]) -> VectorVariable: ...

    @overload
    def __getitem__(self, key: tuple[slice, int]) -> VectorVariable: ...

    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> MatrixVariable: ...

    def __getitem__(
        self, key: tuple[int | slice, int | slice]
    ) -> Variable | VectorVariable | MatrixVariable:
        """Index or slice the matrix.

        Args:
            key: Tuple of (row, col) indices or slices.

        Returns:
            - Single Variable for A[i, j]
            - VectorVariable for A[i, :] (row) or A[:, j] (column)
            - MatrixVariable for A[i1:i2, j1:j2] (submatrix)

        Example:
            >>> A = MatrixVariable("A", 3, 4)
            >>> A[0, 0]    # Variable("A[0,0]")
            >>> A[1, :]    # VectorVariable (row 1)
            >>> A[:, 2]    # VectorVariable (column 2)
            >>> A[0:2, 1:3]  # 2x2 submatrix
        """
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("Matrix indices must be a tuple of (row, col)")

        row_key, col_key = key

        # Handle negative indices
        if isinstance(row_key, int):
            if row_key < 0:
                row_key = self.rows + row_key
            if row_key < 0 or row_key >= self.rows:
                raise IndexError(
                    f"Row index {row_key} out of range for matrix with {self.rows} rows"
                )

        if isinstance(col_key, int):
            if col_key < 0:
                col_key = self.cols + col_key
            if col_key < 0 or col_key >= self.cols:
                raise IndexError(
                    f"Column index {col_key} out of range for matrix with {self.cols} cols"
                )

        # Case 1: A[i, j] -> Variable
        if isinstance(row_key, int) and isinstance(col_key, int):
            return self._variables[row_key][col_key]

        # Case 2: A[i, :] -> VectorVariable (row)
        if isinstance(row_key, int) and isinstance(col_key, slice):
            row_vars = self._variables[row_key][col_key]
            if len(row_vars) == 0:
                raise IndexError("Slice results in empty row")
            return VectorVariable._from_variables(
                name=f"{self.name}[{row_key},:]",
                variables=row_vars,
                lb=self.lb,
                ub=self.ub,
                domain=self.domain,
            )

        # Case 3: A[:, j] -> VectorVariable (column)
        if isinstance(row_key, slice) and isinstance(col_key, int):
            col_vars = [row[col_key] for row in self._variables[row_key]]
            if len(col_vars) == 0:
                raise IndexError("Slice results in empty column")
            return VectorVariable._from_variables(
                name=f"{self.name}[:,{col_key}]",
                variables=col_vars,
                lb=self.lb,
                ub=self.ub,
                domain=self.domain,
            )

        # Case 4: A[i1:i2, j1:j2] -> MatrixVariable (submatrix)
        if isinstance(row_key, slice) and isinstance(col_key, slice):
            sliced_rows = self._variables[row_key]
            if len(sliced_rows) == 0:
                raise IndexError("Slice results in empty matrix")
            sliced_vars = [row[col_key] for row in sliced_rows]
            if len(sliced_vars[0]) == 0:
                raise IndexError("Slice results in empty matrix")
            return MatrixVariable._from_variables(
                name=f"{self.name}[{row_key.start or 0}:{row_key.stop or self.rows},{col_key.start or 0}:{col_key.stop or self.cols}]",
                variables=sliced_vars,
                lb=self.lb,
                ub=self.ub,
                domain=self.domain,
            )

        raise TypeError(
            f"Invalid index types: ({type(row_key).__name__}, {type(col_key).__name__})"
        )

    @classmethod
    def _from_variables(
        cls,
        name: str,
        variables: list[list[Variable]],
        lb: float | None = None,
        ub: float | None = None,
        domain: DomainType = "continuous",
    ) -> MatrixVariable:
        """Create a MatrixVariable from existing Variable instances.

        This is an internal constructor used for slicing.
        """
        instance = object.__new__(cls)
        instance.name = name
        instance.rows = len(variables)
        instance.cols = len(variables[0]) if variables else 0
        instance.lb = lb
        instance.ub = ub
        instance.domain = domain
        instance.symmetric = False
        instance._is_transpose = False
        instance._variables = [list(row) for row in variables]  # Deep copy
        return instance

    def __iter__(self) -> Iterator[VectorVariable]:
        """Iterate over rows of the matrix."""
        for i in range(self.rows):
            yield self[i, :]

    def __len__(self) -> int:
        """Return the number of rows."""
        return self.rows

    def get_variables(self) -> list[Variable]:
        """Return all variables in this matrix (row-major order).

        For symmetric matrices, each unique variable appears only once.

        Returns:
            List of Variable instances.
        """
        if self.symmetric:
            # For symmetric, only return upper triangle + diagonal
            result: list[Variable] = []
            for i in range(self.rows):
                for j in range(i, self.cols):
                    result.append(self._variables[i][j])
            return result
        else:
            return [var for row in self._variables for var in row]

    def to_numpy(self, solution: dict[str, float]) -> NDArray[np.floating]:
        """Extract matrix values from solution as numpy array.

        Args:
            solution: Dictionary mapping variable names to values.

        Returns:
            2D numpy array with the solution values.

        Example:
            >>> A = MatrixVariable("A", 2, 2)
            >>> solution = {"A[0,0]": 1, "A[0,1]": 2, "A[1,0]": 3, "A[1,1]": 4}
            >>> A.to_numpy(solution)
            array([[1., 2.],
                   [3., 4.]])
        """
        result = np.zeros((self.rows, self.cols))
        for i in range(self.rows):
            for j in range(self.cols):
                result[i, j] = solution[self._variables[i][j].name]
        return result

    def __repr__(self) -> str:
        bounds = ""
        if self.lb is not None or self.ub is not None:
            bounds = f", lb={self.lb}, ub={self.ub}"
        domain_str = "" if self.domain == "continuous" else f", domain='{self.domain}'"
        sym_str = ", symmetric=True" if self.symmetric else ""
        return f"MatrixVariable('{self.name}', {self.rows}, {self.cols}{bounds}{domain_str}{sym_str})"
