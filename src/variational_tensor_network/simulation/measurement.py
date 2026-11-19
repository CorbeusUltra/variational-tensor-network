from collections.abc import Callable
from itertools import combinations
from math import isfinite
from typing import Any

from torch import Tensor

from variational_tensor_network.objects import Direction
from variational_tensor_network.observables import (
    magnetization_x,
    magnetization_z,
    spin_spin_correlation,
)
from variational_tensor_network.rdms import BondRDMs
from variational_tensor_network.tilings import Site

type SpinCorrelations = dict[tuple[Site, Direction], dict[tuple[int, int], float]]


def _measure_magnetization(rdms: BondRDMs, spins_per_peps_site: int, measure_spin: Callable[[int, int, Tensor], float]) -> tuple[dict[Site, tuple[float, ...]], float]:
    if spins_per_peps_site < 1:
        raise ValueError("spins_per_peps_site must be positive.")
    local: dict[Site, tuple[float, ...]] = {}
    spins_per_bond = 2 * spins_per_peps_site
    for (site, direction), rho in rdms.items():
        if direction == "r":
            local[site] = tuple(measure_spin(i, spins_per_bond, rho) for i in range(1, spins_per_peps_site + 1))
    if not local:
        raise ValueError("At least one right-bond RDM is required.")
    total = sum(sum(values) for values in local.values())
    saturation = len(local) * spins_per_peps_site / 2
    return local, total / saturation


def measure_magnetization_x(rdms: BondRDMs, spins_per_peps_site: int) -> tuple[dict[Site, tuple[float, ...]], float]:
    return _measure_magnetization(rdms, spins_per_peps_site, magnetization_x)


def measure_magnetization_z(rdms: BondRDMs, spins_per_peps_site: int) -> tuple[dict[Site, tuple[float, ...]], float]:
    return _measure_magnetization(rdms, spins_per_peps_site, magnetization_z)



def measure_spin_spin_correlations(rdms: BondRDMs, spins_per_peps_site: int) -> SpinCorrelations:
    if spins_per_peps_site < 1:
        raise ValueError("spins_per_peps_site must be positive.")
    n_spins = 2 * spins_per_peps_site
    correlations = {}
    for bond, rho in rdms.items():
        pairs = {}
        for i, j in combinations(range(1, n_spins + 1), 2):
            pairs[(i, j)] = spin_spin_correlation(i, j, n_spins, rho)
        correlations[bond] = pairs
    return correlations



def validate_observables(observables: dict[str, Any]) -> None:
    groups = {
        "energy_per_spin": [observables["energy_per_spin"]],
        "m_z_over_m_sat": [observables["m_z_over_m_sat"]],
        "local_magnetization_z": [value for values in observables["local_magnetization_z"].values() for value in values],
        "spin_spin_correlations": [value for pairs in observables["spin_spin_correlations"].values() for value in pairs.values()],
    }
    for name, values in groups.items():
        if not all(isfinite(value) for value in values):
            raise FloatingPointError(f"Non-finite {name}.")
