"""Full CTMRG algorithm."""

from time import perf_counter
from warnings import warn

import torch
from torch import Tensor, stack

from .ctmrg_moves import all_moves_sites
from .objects import Config, EnvSites
from .observables import ctmrg_rdms, expectation_from_rdm
from .peps import TensorSites, double_layer
from .reporting import format_duration, format_scientific, print_progress
from .tilings import UnitCellCoord


def run_ctmrg(env_sites: EnvSites, a_tilde_sites: TensorSites, unitcell_coord: UnitCellCoord, config: Config, H: Tensor, *, stop_on_convergence: bool = True, max_cycles: int = 2000, conv_tol: float = 1e-6, verbosity: int = 1) -> tuple[EnvSites, list[Tensor], list[Tensor]]:
    if max_cycles < 2:
        raise ValueError("max_cycles must be at least 2.")
    start = last_update = perf_counter()
    if verbosity >= 1:
        print(f"\nchi={config.chi}, tol={format_scientific(conv_tol)}\n", flush=True)
    print_progress(f"  cycle=1/{max_cycles} | first cycle and energy measurement...", verbosity)
    A_sites = {site: double_layer(a_tilde) for site, a_tilde in a_tilde_sites.items()}
    env_sites, _ = all_moves_sites(A_sites, env_sites, unitcell_coord, config)
    rdms = ctmrg_rdms(a_tilde_sites, env_sites, unitcell_coord)
    energies = stack([expectation_from_rdm(rho, H) for rho in rdms.values()]).reshape(len(a_tilde_sites), 2)
    stable_cycles = 0
    errors: list[Tensor] = []
    thresholds: list[Tensor] = []
    for cycle in range(2, max_cycles + 1):
        old_energies = energies
        env_sites, _ = all_moves_sites(A_sites, env_sites, unitcell_coord, config)
        rdms = ctmrg_rdms(a_tilde_sites, env_sites, unitcell_coord)
        energies = stack([expectation_from_rdm(rho, H) for rho in rdms.values()]).reshape(len(a_tilde_sites), 2)
        error = (energies - old_energies).abs().amax(dim=1)
        threshold = torch.full_like(error, conv_tol)
        errors.append(error)
        thresholds.append(threshold)
        converged = (error <= threshold).all()
        stable_cycles = stable_cycles + 1 if converged else 0
        maximum_error = error.max().item()
        now = perf_counter()
        if cycle == 2 or now - last_update >= 1.0:
            print_progress(f"  cycle={cycle}/{max_cycles} | error={maximum_error:.2e} | stable={min(stable_cycles, 3)}/3 | time={format_duration(now - start)}", verbosity)
            last_update = now
        if stop_on_convergence and stable_cycles >= 3:
            print_progress(f"  Converged | cycle={cycle}/{max_cycles} | error={maximum_error:.2e} | stable={min(stable_cycles, 3)}/3 | time={format_duration(now - start)}", verbosity, final=True)
            return env_sites, errors, thresholds
    status = "Not converged" if stop_on_convergence else "Fixed cycles completed"
    print_progress(f"  {status} | cycle={cycle}/{max_cycles} | error={maximum_error:.2e} | stable={min(stable_cycles, 3)}/3 | time={format_duration(perf_counter() - start)}", verbosity, final=True)
    if stop_on_convergence:
        recent_errors = "\n".join(f"{error.max().item():.3e}" for error in errors[-10:])
        unconverged_sites = (~(errors[-1] <= thresholds[-1])).sum().item()
        threshold_min = thresholds[-1].min().item()
        threshold_max = thresholds[-1].max().item()
        warn(f"CTMRG did not converge after {max_cycles} cycles "
             f"(D={config.D}, chi={config.chi}).\n"
             f"Unconverged sites: {unconverged_sites}/{len(env_sites)}.\n"
             f"Consecutive cycles below tolerance: {stable_cycles}/3.\n"
             f"Last maximum bond-energy changes, oldest to newest: \n{recent_errors}\n"
             f"Final per-site thresholds: {threshold_min:.2e} to {threshold_max:.2e}.\n"
             "Returning the last environments without convergence.\n",
             RuntimeWarning)
    return env_sites, errors, thresholds
