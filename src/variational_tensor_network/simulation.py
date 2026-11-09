"""Run simple-update optimization and CTMRG energy measurements."""

from dataclasses import replace  # noqa: I001

from torch import Tensor

from .ctmrg import ctmrg
from .initialize import initial_a_sites, initial_bond_weights, initial_env_sites, initial_boundary
from .objects import Config
from .peps import absorb_sqrt_weights, double_layer, get_site_weights
from .reporting import print_run_summary
from .simple_update import simple_update
from .tilings import TILINGS


def run_simulation(tiling_name: str, config: Config, H: Tensor, chi_values: list[int], schedule: tuple[tuple[float, float], ...], *, verbosity: int = 1) -> tuple[dict, dict]:
    """Optimize one state and measure energy per PEPS site at several CTMRG dimensions."""
    sites_coords, unitcell_coord = TILINGS[tiling_name]
    a_sites = initial_a_sites(config, sites_coords)
    bond_weights = initial_bond_weights(a_sites)
    a_sites, bond_weights, su_info = simple_update(a_sites, bond_weights, unitcell_coord, H, schedule=schedule, verbosity=verbosity)
    print_run_summary("Simple update", su_info, verbosity)
    a_tilde_sites = {}
    for site, a in a_sites.items():
        weights = get_site_weights(site, bond_weights, unitcell_coord)
        a_tilde_sites[site] = absorb_sqrt_weights(a, weights)
    A_sites = {site: double_layer(a_tilde) for site, a_tilde in a_tilde_sites.items()}
    ctmrg_info_by_chi = {}
    for chi_index, chi in enumerate(chi_values, start=1):
        label = f"CTMRG {chi_index:>{len(str(len(chi_values)))}}/{len(chi_values)}"
        config_chi = replace(config, chi=chi)
        delta = initial_boundary(config_chi)
        env_sites = initial_env_sites(A_sites, delta, config_chi)
        env_sites, ctmrg_info = ctmrg(env_sites, a_tilde_sites, unitcell_coord, config_chi, H, conv_tol=1e-8, verbosity=verbosity, label=label)
        ctmrg_info_by_chi[chi] = ctmrg_info
        print_run_summary("CTMRG", ctmrg_info, verbosity)
    return su_info, ctmrg_info_by_chi



# Legacy code

# def run_D_scan(D_values: list[int], tiling_name: str, config: Config, H: Tensor, chi_values_by_D: dict[int, list[int]], schedule: tuple[tuple[float, float], ...], *, verbosity: int = 1) -> dict[int, dict]:
#     """Run simulations at several bond dimensions for one tiling."""
#     results_by_D = {}
#     for D in D_values:
#         start_D = perf_counter()
#         chi_values = chi_values_by_D[D]
#         config = replace(config, D=D)
#         if verbosity >= 1:
#             print(f"\n\nD={D}\n", flush=True)
#         su_info, ctmrg_info_by_chi = run_simulation(tiling_name, config, H, chi_values, schedule, verbosity=verbosity)
#         results_by_D[D] = {
#             "chi_values": chi_values,
#             "su_info": su_info,
#             "ctmrg_info_by_chi": ctmrg_info_by_chi,
#             "tau_error_history": su_info["tau_error_history"],
#             "energy_per_peps_site_su": su_info["energy_per_peps_site"],
#             "energies_per_peps_site_ctmrg": [
#                 ctmrg_info_by_chi[chi]["energy_per_peps_site"]
#                 for chi in chi_values
#             ],
#         }
#         if verbosity >= 1:
#             print(f"\n{tiling_name} | D={D} | total time: {format_duration(perf_counter() - start_D)}\n", flush=True)
#     return results_by_D
