"""Evaluate local observables from bond reduced density matrices."""

from torch import Tensor, kron

from .models import heisenberg_term, spin_x, spin_z
from .rdms import BondRDMs


def expectation_from_rdm(rho: Tensor, operator: Tensor) -> Tensor:
    """Measure a two-site matrix operator using rho[s, t, S, T]."""
    size = rho.shape[0] * rho.shape[1]
    rho_matrix = rho.reshape(size, size)
    if operator.shape != rho_matrix.shape:
        raise ValueError(f"Expected operator.shape == {tuple(rho_matrix.shape)}, got {tuple(operator.shape)}.")
    return (rho_matrix * operator.mT).sum()



def mean_magnetization_z(rdms: BondRDMs, spins_per_peps_site: int) -> float:
    """Measure the mean longitudinal magnetization per physical spin."""
    spins_per_bond = 2 * spins_per_peps_site
    S_z = sum(spin_z(i, spins_per_bond) for i in range(1, spins_per_bond + 1))
    S_z_per_spin = S_z / spins_per_bond
    return sum(expectation_from_rdm(rho, S_z_per_spin).item() for rho in rdms.values()) / len(rdms)


def mean_spin_z_spin_z_correlation(rdms: BondRDMs, spins_per_peps_site: int) -> float:
    """Measure the mean longitudinal correlation between spins on neighboring PEPS sites."""
    S_z_per_spin = sum(spin_z(i, spins_per_peps_site) for i in range(1, spins_per_peps_site + 1)) / spins_per_peps_site
    operator = kron(S_z_per_spin, S_z_per_spin)
    return sum(expectation_from_rdm(rho, operator).item() for rho in rdms.values()) / len(rdms)


def magnetization_z(i: int, n_spins:int , rho: Tensor) -> float:
    S_z = spin_z(i, n_spins)
    return expectation_from_rdm(rho, S_z).item()

def magnetization_x(i: int, n_spins:int , rho: Tensor) -> float:
    S_x = spin_x(i, n_spins)
    return expectation_from_rdm(rho, S_x).item()

def spin_spin_correlation(i: int, j: int, n_spins: int, rho: Tensor) -> float :
    SxS = heisenberg_term(i, j, n_spins)
    return expectation_from_rdm(rho, SxS).item()
