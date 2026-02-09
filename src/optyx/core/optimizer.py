"""Expression optimization passes."""

from optyx.core.expressions import Expression, BinaryOp, NarySum, NaryProduct


def flatten_expression(expr: Expression) -> Expression:
    """Flatten an expression tree by coalescing associative operations.

    Converts nested addition chains into NarySum nodes and nested multiplication
    chains into NaryProduct nodes. This reduces tree depth from O(N) to O(1)
    for loop-constructed sums/products.

    Uses iterative traversal to handle deep trees without RecursionError.

    Args:
        expr: The expression to optimize.

    Returns:
        A new optimized expression (or the original if no changes needed).
    """
    # 1. Handle Associative Chains (Iteratively)
    if isinstance(expr, BinaryOp) and expr.op in ("+", "*"):
        terms = _gather_associative_terms(expr, expr.op)

        # Optimize the gathered terms recursively
        # (The depth of *different* operators is usually shallow, so recursion is safe here)
        optimized_terms = tuple(flatten_expression(t) for t in terms)

        # Reconstruct
        if len(optimized_terms) == 2:
            return BinaryOp(optimized_terms[0], optimized_terms[1], expr.op)  # type: ignore

        if expr.op == "+":
            return NarySum(optimized_terms)
        elif expr.op == "*":
            return NaryProduct(optimized_terms)

    # 2. Recursively optimize other BinaryOps (e.g. -, /, **)
    if isinstance(expr, BinaryOp):
        left = flatten_expression(expr.left)
        right = flatten_expression(expr.right)
        if left is not expr.left or right is not expr.right:
            return BinaryOp(left, right, expr.op)
        return expr

    # 3. Future: Recurse into UnaryOps, etc.

    return expr


def _gather_associative_terms(expr: Expression, op: str) -> list[Expression]:
    """Iteratively gather inputs for an associative chain.

    Uses an explicit stack to avoid RecursionError on deep trees.
    Preserves Left-to-Right operand order.
    """
    terms: list[Expression] = []

    # Stack stores nodes to visit.
    # To yield L then R, we must push R then L.
    stack = [expr]

    while stack:
        node = stack.pop()

        if isinstance(node, BinaryOp) and node.op == op:
            stack.append(node.right)
            stack.append(node.left)
        elif isinstance(node, NarySum) and op == "+":
            # NarySum stores terms (t1, t2, t3).
            # To pop t1, t2, t3, push them in reverse: t3, t2, t1.
            for term in reversed(node.terms):
                stack.append(term)
        elif isinstance(node, NaryProduct) and op == "*":
            for factor in reversed(node.factors):
                stack.append(factor)
        else:
            terms.append(node)

    return terms
