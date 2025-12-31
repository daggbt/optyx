"""Matrix variables for optimization problems.

This module provides MatrixVariable for representing 2D matrices of decision variables,
enabling natural syntax like `A = MatrixVariable("A", 3, 4)` with 2D indexing.

It also provides matrix operations like matrix-vector multiplication, quadratic forms,
trace, and diagonal extraction.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Iterator, overload

import numpy as np

from optyx.core.expressions import Expression, Variable, BinaryOp, Constant
from optyx.core.vectors import (
    VectorVariable,
    VectorExpression,
    DomainType,
    LinearCombination,
)
from optyx.core.errors import DimensionMismatchError

if TYPE_CHECKING:
    from numpy.typing import NDArray, ArrayLike


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
    def size(self) -> int:
        """Return the total number of elements."""
        return self.rows * self.cols

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

    def rows_iter(self) -> Iterator[VectorVariable]:
        """Iterate over rows of the matrix.

        Returns:
            Iterator of VectorVariable, one for each row.

        Example:
            >>> A = MatrixVariable("A", 3, 4)
            >>> for i, row in enumerate(A.rows_iter()):
            ...     print(f"Row {i}: {len(row)} elements")
        """
        for i in range(self.rows):
            yield self[i, :]

    def cols_iter(self) -> Iterator[VectorVariable]:
        """Iterate over columns of the matrix.

        Returns:
            Iterator of VectorVariable, one for each column.

        Example:
            >>> A = MatrixVariable("A", 3, 4)
            >>> for j, col in enumerate(A.cols_iter()):
            ...     print(f"Col {j}: {len(col)} elements")
        """
        for j in range(self.cols):
            yield self[:, j]

    def diagonal(self) -> VectorVariable:
        """Extract the main diagonal of a square matrix.

        Returns:
            VectorVariable containing the diagonal elements.

        Raises:
            ValueError: If the matrix is not square.

        Example:
            >>> A = MatrixVariable("A", 3, 3)
            >>> d = A.diagonal()
            >>> len(d)  # 3
            >>> d[0].name  # 'A[0,0]'
        """
        if self.rows != self.cols:
            raise ValueError(
                f"diagonal() requires a square matrix, got {self.rows}x{self.cols}"
            )

        diag_vars = [self._variables[i][i] for i in range(self.rows)]

        return VectorVariable._from_variables(
            name=f"diag({self.name})",
            variables=diag_vars,
            lb=self.lb,
            ub=self.ub,
            domain=self.domain,
        )

    def trace(self) -> Expression:
        """Compute the trace (sum of diagonal elements) of a square matrix.

        Returns:
            Expression representing the sum of diagonal elements.

        Raises:
            ValueError: If the matrix is not square.

        Example:
            >>> A = MatrixVariable("A", 3, 3)
            >>> tr = A.trace()
            >>> # tr = A[0,0] + A[1,1] + A[2,2]
        """
        from optyx.core.expressions import BinaryOp

        if self.rows != self.cols:
            raise ValueError(
                f"trace() requires a square matrix, got {self.rows}x{self.cols}"
            )

        # Sum diagonal elements
        result: Expression = self._variables[0][0]
        for i in range(1, self.rows):
            result = BinaryOp(result, self._variables[i][i], "+")
        return result

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


# =============================================================================
# Matrix-Vector Multiplication
# =============================================================================


class MatrixVectorProduct(VectorExpression):
    """Matrix-vector product: A @ x where A is a constant matrix.

    This creates a VectorExpression where each element is a linear combination
    of the vector elements weighted by the corresponding matrix row.

    Args:
        matrix: 2D NumPy array (constant coefficients).
        vector: VectorVariable or VectorExpression to multiply.

    Example:
        >>> import numpy as np
        >>> A = np.array([[1, 2], [3, 4]])
        >>> x = VectorVariable("x", 2)
        >>> product = MatrixVectorProduct(A, x)
        >>> product.evaluate({"x[0]": 1, "x[1]": 2})
        [5.0, 11.0]  # [1*1+2*2, 3*1+4*2]
    """

    __slots__ = ("matrix", "vector", "_expressions", "size")

    def __init__(
        self,
        matrix: np.ndarray,
        vector: VectorVariable | VectorExpression,
    ) -> None:
        matrix = np.asarray(matrix)
        if matrix.ndim != 2:
            raise ValueError(f"Matrix must be 2D, got {matrix.ndim}D")

        vec_size = vector.size if hasattr(vector, "size") else len(vector)
        if matrix.shape[1] != vec_size:
            raise DimensionMismatchError(
                operation="matrix-vector product",
                left_shape=matrix.shape,
                right_shape=vec_size,
                suggestion="Matrix columns must match vector size.",
            )

        self.matrix = matrix
        self.vector = vector
        self.size = matrix.shape[0]

        # Create a LinearCombination for each row
        self._expressions: list[Expression] = [
            LinearCombination(matrix[i, :], vector) for i in range(self.size)
        ]

    def evaluate(self, values: Mapping[str, ArrayLike | float]) -> list[float]:
        """Evaluate the matrix-vector product."""
        return [expr.evaluate(values) for expr in self._expressions]  # type: ignore[misc]

    def get_variables(self) -> set[Variable]:
        """Return all variables this expression depends on."""
        if isinstance(self.vector, VectorVariable):
            return set(self.vector._variables)
        return self.vector.get_variables()

    def __repr__(self) -> str:
        vec_name = (
            self.vector.name if isinstance(self.vector, VectorVariable) else "expr"
        )
        return f"MatrixVectorProduct({self.matrix.shape}, {vec_name})"


def matmul(
    matrix: np.ndarray, vector: VectorVariable | VectorExpression
) -> MatrixVectorProduct:
    """Matrix-vector multiplication: A @ x.

    Args:
        matrix: 2D NumPy array of constant coefficients.
        vector: VectorVariable or VectorExpression.

    Returns:
        MatrixVectorProduct (a VectorExpression).

    Example:
        >>> import numpy as np
        >>> A = np.array([[1, 2], [3, 4]])
        >>> x = VectorVariable("x", 2)
        >>> y = matmul(A, x)  # or just A @ x
    """
    return MatrixVectorProduct(matrix, vector)


# =============================================================================
# Quadratic Form
# =============================================================================


class QuadraticForm(Expression):
    """Quadratic form: x' @ Q @ x where Q is a constant matrix.

    This represents the scalar expression xᵀQx, commonly used for:
    - Portfolio variance: w' @ Σ @ w
    - Regularization terms: x' @ I @ x = ||x||²
    - Quadratic objectives in optimization

    Args:
        vector: VectorVariable or VectorExpression (the x).
        matrix: 2D NumPy array (the Q matrix, should be square).

    Example:
        >>> import numpy as np
        >>> Q = np.array([[1, 0.5], [0.5, 2]])
        >>> x = VectorVariable("x", 2)
        >>> qf = QuadraticForm(x, Q)
        >>> qf.evaluate({"x[0]": 1, "x[1]": 1})
        4.0  # 1*1 + 2*0.5*1*1 + 2*1 = 1 + 1 + 2 = 4
    """

    __slots__ = ("vector", "matrix")

    def __init__(
        self,
        vector: VectorVariable | VectorExpression,
        matrix: np.ndarray,
    ) -> None:
        matrix = np.asarray(matrix)
        if matrix.ndim != 2:
            raise ValueError(f"Matrix must be 2D, got {matrix.ndim}D")
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(
                f"Matrix must be square, got {matrix.shape[0]}x{matrix.shape[1]}"
            )

        vec_size = vector.size if hasattr(vector, "size") else len(vector)
        if matrix.shape[0] != vec_size:
            raise DimensionMismatchError(
                operation="quadratic form",
                left_shape=matrix.shape,
                right_shape=vec_size,
                suggestion="Matrix dimensions must match vector size.",
            )

        self.vector = vector
        self.matrix = matrix

    def evaluate(self, values: Mapping[str, ArrayLike | float]) -> float:
        """Evaluate the quadratic form xᵀQx."""
        # Get vector values
        if isinstance(self.vector, VectorVariable):
            x = np.array([v.evaluate(values) for v in self.vector._variables])
        else:
            x = np.array([expr.evaluate(values) for expr in self.vector._expressions])

        # Compute x' @ Q @ x
        return float(x @ self.matrix @ x)

    def get_variables(self) -> set[Variable]:
        """Return all variables this expression depends on."""
        if isinstance(self.vector, VectorVariable):
            return set(self.vector._variables)
        return self.vector.get_variables()

    def __repr__(self) -> str:
        vec_name = (
            self.vector.name if isinstance(self.vector, VectorVariable) else "expr"
        )
        return f"QuadraticForm({vec_name}, {self.matrix.shape})"


def quadratic_form(
    vector: VectorVariable | VectorExpression, matrix: np.ndarray
) -> QuadraticForm:
    """Create a quadratic form expression: x' @ Q @ x.

    Args:
        vector: VectorVariable or VectorExpression.
        matrix: 2D NumPy array (square matrix Q).

    Returns:
        QuadraticForm expression.

    Example:
        >>> import numpy as np
        >>> from optyx import VectorVariable
        >>> from optyx.core.matrices import quadratic_form
        >>>
        >>> # Portfolio variance
        >>> cov_matrix = np.array([[0.04, 0.01], [0.01, 0.09]])
        >>> weights = VectorVariable("w", 2, lb=0, ub=1)
        >>> variance = quadratic_form(weights, cov_matrix)
    """
    return QuadraticForm(vector, matrix)


# =============================================================================
# Trace Function
# =============================================================================


def trace(matrix: MatrixVariable | np.ndarray) -> Expression:
    """Compute the trace of a matrix (sum of diagonal elements).

    Args:
        matrix: MatrixVariable or NumPy array.

    Returns:
        Expression representing the sum of diagonal elements.

    Raises:
        ValueError: If matrix is not square.

    Example:
        >>> A = MatrixVariable("A", 3, 3)
        >>> tr = trace(A)
        >>> # tr = A[0,0] + A[1,1] + A[2,2]
    """
    if isinstance(matrix, np.ndarray):
        # For constant matrix, return scalar
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("trace requires a square matrix")
        return Constant(float(np.trace(matrix)))

    if matrix.rows != matrix.cols:
        raise ValueError(
            f"trace requires a square matrix, got {matrix.rows}x{matrix.cols}"
        )

    # Sum diagonal elements
    result: Expression = matrix[0, 0]
    for i in range(1, matrix.rows):
        result = BinaryOp(result, matrix[i, i], "+")
    return result


# =============================================================================
# Diagonal Functions
# =============================================================================


def diag(
    matrix_or_vector: MatrixVariable | VectorVariable | np.ndarray,
) -> VectorVariable | VectorExpression | np.ndarray:
    """Extract diagonal from a matrix, or create a diagonal matrix from a vector.

    This function has dual behavior like numpy.diag:
    - If input is a matrix: extract diagonal as a vector
    - If input is a vector: this raises an error (use diag_matrix instead)

    Args:
        matrix_or_vector: MatrixVariable or NumPy array.

    Returns:
        - For MatrixVariable: VectorVariable containing diagonal elements
        - For 2D NumPy array: 1D NumPy array of diagonal

    Raises:
        ValueError: If matrix is not square or if given a vector.

    Example:
        >>> A = MatrixVariable("A", 3, 3)
        >>> d = diag(A)  # VectorVariable with A[0,0], A[1,1], A[2,2]
    """
    if isinstance(matrix_or_vector, np.ndarray):
        return np.diag(matrix_or_vector)

    if isinstance(matrix_or_vector, VectorVariable):
        raise TypeError(
            "diag() on VectorVariable is ambiguous. Use diag_matrix() to create "
            "a diagonal matrix from a vector."
        )

    matrix = matrix_or_vector
    if matrix.rows != matrix.cols:
        raise ValueError(
            f"diag requires a square matrix, got {matrix.rows}x{matrix.cols}"
        )

    # Extract diagonal variables
    diag_vars = [matrix[i, i] for i in range(matrix.rows)]

    return VectorVariable._from_variables(
        name=f"diag({matrix.name})",
        variables=diag_vars,
        lb=matrix.lb,
        ub=matrix.ub,
        domain=matrix.domain,
    )


def diag_matrix(
    vector: VectorVariable | np.ndarray,
    lb: float | None = None,
    ub: float | None = None,
) -> MatrixVariable | np.ndarray:
    """Create a diagonal matrix from a vector.

    Args:
        vector: VectorVariable or 1D NumPy array.
        lb: Lower bound for off-diagonal elements (only for VectorVariable).
        ub: Upper bound for off-diagonal elements (only for VectorVariable).

    Returns:
        - For VectorVariable: MatrixVariable with vector on diagonal, zeros elsewhere
        - For 1D NumPy array: 2D NumPy array with diagonal

    Note:
        For VectorVariable, this creates a new MatrixVariable where the diagonal
        elements are the same Variable objects as the input vector. Off-diagonal
        elements are fixed at 0 (via bounds lb=0, ub=0).

    Example:
        >>> x = VectorVariable("x", 3)
        >>> D = diag_matrix(x)  # 3x3 matrix with x on diagonal
    """
    if isinstance(vector, np.ndarray):
        return np.diag(vector)

    n = len(vector)

    # Create a new matrix with the vector on the diagonal
    # We need to create a special matrix where diagonal = vector, off-diagonal = 0
    # For simplicity, we'll create a MatrixVariable from existing variables

    # Build the variable grid
    variables: list[list[Variable]] = []
    for i in range(n):
        row: list[Variable] = []
        for j in range(n):
            if i == j:
                # Diagonal: use the vector's variable
                row.append(vector._variables[i])
            else:
                # Off-diagonal: create a fixed-zero variable
                row.append(
                    Variable(
                        f"_diag_{vector.name}[{i},{j}]",
                        lb=0.0,
                        ub=0.0,
                        domain=vector.domain,
                    )
                )
        variables.append(row)

    return MatrixVariable._from_variables(
        name=f"diag({vector.name})",
        variables=variables,
        lb=lb,
        ub=ub,
        domain=vector.domain,
    )


# =============================================================================
# Frobenius Norm
# =============================================================================


class FrobeniusNorm(Expression):
    """Frobenius norm of a matrix: ||A||_F = sqrt(sum of squared elements).

    Args:
        matrix: MatrixVariable to compute norm of.

    Example:
        >>> A = MatrixVariable("A", 2, 2)
        >>> fn = FrobeniusNorm(A)
        >>> fn.evaluate({"A[0,0]": 1, "A[0,1]": 2, "A[1,0]": 3, "A[1,1]": 4})
        5.477...  # sqrt(1+4+9+16) = sqrt(30)
    """

    __slots__ = ("matrix",)

    def __init__(self, matrix: MatrixVariable) -> None:
        self.matrix = matrix

    def evaluate(self, values: Mapping[str, ArrayLike | float]) -> float:
        """Evaluate the Frobenius norm."""
        sum_sq = 0.0
        for i in range(self.matrix.rows):
            for j in range(self.matrix.cols):
                val = self.matrix[i, j].evaluate(values)
                sum_sq += val * val
        return float(np.sqrt(sum_sq))

    def get_variables(self) -> set[Variable]:
        """Return all variables this expression depends on."""
        return set(self.matrix.get_variables())

    def __repr__(self) -> str:
        return f"FrobeniusNorm({self.matrix.name})"


def frobenius_norm(matrix: MatrixVariable) -> FrobeniusNorm:
    """Compute the Frobenius norm of a matrix.

    Args:
        matrix: MatrixVariable to compute norm of.

    Returns:
        FrobeniusNorm expression.

    Example:
        >>> A = MatrixVariable("A", 2, 2)
        >>> norm = frobenius_norm(A)
    """
    return FrobeniusNorm(matrix)
