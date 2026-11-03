"""Run simple-update optimization and CTMRG energy measurements."""

from dataclasses import replace  # noqa: I001
from time import perf_counter

from torch import Tensor

from .ctmrg import run_ctmrg
from .initialize import initial_a_sites, initial_bond_weights, initial_env_sites, uniform_boundary
from .objects import Config
from .observables import simple_update_rdms, ctmrg_rdms, expectation_from_rdm
from .peps import absorb_sqrt_weights, double_layer, get_site_weights
from .reporting import format_duration
from .simple_update import simple_update
from .tilings import TILINGS


def run_simulation(tiling_name: str, config: Config, H: Tensor, chi_values: list[int], schedule: tuple[tuple[float, float], ...], *, verbosity: int = 1) -> tuple[list[tuple[float, float]], float, list[float]]:
    """Optimize one state and measure energy per PEPS site at several CTMRG dimensions."""
    if verbosity >= 1:
        print("Simple update")
    sites_coords, unitcell_coord = TILINGS[tiling_name]
    a_sites = initial_a_sites(config, sites_coords)
    bond_weights = initial_bond_weights(a_sites)
    start_su = perf_counter()
    history = simple_update(a_sites, bond_weights, unitcell_coord, H, schedule=schedule, verbosity=verbosity)
    rdms_su = simple_update_rdms(a_sites, bond_weights, unitcell_coord)
    energy_per_peps_site_su = sum(expectation_from_rdm(rho, H).item() for rho in rdms_su.values()) / len(a_sites)
    if verbosity >= 1:
        print(f"\nSimple update total time: {format_duration(perf_counter() - start_su)}\nenergy per PEPS site simple update : {energy_per_peps_site_su:.12f}\n", flush=True)
    a_tilde_sites = {}
    for site, a in a_sites.items():
        weights = get_site_weights(site, bond_weights, unitcell_coord)
        a_tilde_sites[site] = absorb_sqrt_weights(a, weights)
    A_sites = {site: double_layer(a_tilde) for site, a_tilde in a_tilde_sites.items()}
    energies_per_peps_site_ctmrg = []
    for chi_index, chi in enumerate(chi_values, start=1):
        start_chi = perf_counter()
        if verbosity >= 1:
            print(f"\nCTMRG {chi_index}/{len(chi_values)}", flush=True)
        config_chi = replace(config, chi=chi)
        delta = uniform_boundary(config_chi)
        env_sites = initial_env_sites(A_sites, delta, config_chi)
        env_sites, _, _ = run_ctmrg(env_sites, a_tilde_sites, unitcell_coord, config_chi, H, conv_tol=1e-8, verbosity=verbosity)
        rdms_ctmrg = ctmrg_rdms(a_tilde_sites, env_sites, unitcell_coord)
        energy_per_peps_site_ctmrg = sum(expectation_from_rdm(rho, H).item() for rho in rdms_ctmrg.values()) / len(a_sites)
        energies_per_peps_site_ctmrg.append(energy_per_peps_site_ctmrg)
        if verbosity >= 1:
            print(f"  Energy per PEPS site ctmrg={energy_per_peps_site_ctmrg:.12f}, total time={format_duration(perf_counter() - start_chi)}\n", flush=True)
    return history, energy_per_peps_site_su, energies_per_peps_site_ctmrg


def run_D_scan(D_values: list[int], tiling_name: str, config: Config, H: Tensor, chi_values_by_D: dict[int, list[int]], schedule: tuple[tuple[float, float], ...], *, verbosity: int = 1) -> dict[int, dict]:
    """Run simulations at several bond dimensions for one tiling."""
    results_by_D = {}
    for D in D_values:
        start_D = perf_counter()
        chi_values = chi_values_by_D[D]
        config = replace(config, D=D)
        if verbosity >= 1:
            print(f"\n\nD={D}\n", flush=True)
        history, energy_per_peps_site_su, energies_per_peps_site_ctmrg = run_simulation(tiling_name, config, H, chi_values, schedule, verbosity=verbosity)
        results_by_D[D] = {"chi_values": chi_values, "history": history, "energy_per_peps_site_su": energy_per_peps_site_su, "energies_per_peps_site_ctmrg": energies_per_peps_site_ctmrg}
        if verbosity >= 1:
            print(f"\n{tiling_name} | D={D} | total time: {format_duration(perf_counter() - start_D)}\n", flush=True)
    return results_by_D
