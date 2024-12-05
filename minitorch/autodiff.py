from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # basically derivative of f wrt to dimension i at point vals
    x_vec_high = list(vals)
    x_vec_low = list(vals)
    x_vec_high[arg] += epsilon
    x_vec_low[arg] -= epsilon
    f_x_vec_high = f(*x_vec_high)
    f_x_vec_low = f(*x_vec_low)
    f_prime = (f_x_vec_high - f_x_vec_low) / (2 * epsilon)
    return f_prime


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulate the derivative `x` to the current variable."""
        ...

    @property
    def unique_id(self) -> int:
        """Return the unique identifier for the variable."""
        ...

    def is_leaf(self) -> bool:
        """Check if the variable is a leaf node."""
        ...

    def is_constant(self) -> bool:
        """Check if the variable is constant."""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Return the parent variables of the current variable."""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """Apply the chain rule to propagate the derivative `d_output` backward."""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    visited = set()
    sorted_list: List[Variable] = []

    def _visit(var: Variable) -> None:
        if var.unique_id in visited or var.is_constant():
            return
        if not var.is_leaf():
            for parent in var.parents:
                if not parent.is_constant():
                    _visit(parent)
        visited.add(var.unique_id)
        sorted_list.insert(0, var)

    _visit(variable)
    return sorted_list


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
    ----
    variable: The right-most variable
    deriv: The derivative of the variable that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.

    """
    topo_order = topological_sort(variable)
    derivatives = {variable.unique_id: deriv}
    # Traverse the nodes
    for var in topo_order:
        deriv = derivatives[var.unique_id]
        if var.is_leaf():
            var.accumulate_derivative(derivatives[var.unique_id])
        else:
            parent_variables = var.chain_rule(derivatives[var.unique_id])
            for parent, parent_deriv in parent_variables:
                # Accumulate derivatives for parent nodes
                if parent.is_constant():
                    continue
                derivatives.setdefault(parent.unique_id, 0.0)
                derivatives[parent.unique_id] += parent_deriv
    return


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Return the saved tensors for backpropagation."""
        return self.saved_values
