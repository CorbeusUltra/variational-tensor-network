from collections.abc import Callable

from torch import Tensor

from variational_tensor_network.observables import (
    magnetization_x,
    magnetization_z,
    # spin_spin_correlation,
)
from variational_tensor_network.rdms import BondRDMs
from variational_tensor_network.tilings import Site


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


def print_ctmrg_observables(local_S_z: dict[Site, tuple[float, ...]], m_z_over_m_sat: float, energy_per_spin: float) -> None:
    for site, values_S_z in local_S_z.items():
        print(f"\n  PEPS site : {site}")
        for i, spin_z in enumerate(values_S_z, start=1):
            print(f"    spin {i} | <S_{i}^z> : {spin_z:+.12f}")
    print(f"\n  m_z/m_sat: {m_z_over_m_sat:.12f}\n")
    print(f"\n  Energy per spin: {energy_per_spin:.12f}\n")
