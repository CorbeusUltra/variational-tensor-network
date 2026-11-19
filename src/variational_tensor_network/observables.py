"""Evaluate local observables from bond reduced density matrices."""

from torch import Tensor

from .models import heisenberg_term, spin_x, spin_z


def expectation_from_rdm(rho: Tensor, operator: Tensor) -> Tensor:
    """Measure a two-site matrix operator using rho[s, t, S, T]."""
    size = rho.shape[0] * rho.shape[1]
    rho_matrix = rho.reshape(size, size)
    if operator.shape != rho_matrix.shape:
        raise ValueError(f"Expected operator.shape == {tuple(rho_matrix.shape)}, got {tuple(operator.shape)}.")
    return (rho_matrix * operator.mT).sum()



def magnetization_z(i: int, n_spins:int , rho: Tensor) -> float:
    S_z = spin_z(i, n_spins)
    return expectation_from_rdm(rho, S_z).item()

def magnetization_x(i: int, n_spins:int , rho: Tensor) -> float:
    S_x = spin_x(i, n_spins)
    return expectation_from_rdm(rho, S_x).item()

def spin_spin_correlation(i: int, j: int, n_spins: int, rho: Tensor) -> float :
    S_dot_S = heisenberg_term(i, j, n_spins)
    return expectation_from_rdm(rho, S_dot_S).item()
