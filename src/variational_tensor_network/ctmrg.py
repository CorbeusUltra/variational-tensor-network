"""Full CTMRG algorithm."""

from dataclasses import replace
from time import perf_counter
from warnings import warn

import torch
from torch import Tensor, stack

from .ctmrg_moves import all_moves_sites
from .objects import Config, EnvSites
from .observables import ctmrg_rdms, expectation_from_rdm
from .peps import TensorSites, double_layer
from .reporting import format_duration, print_progress
from .tilings import UnitCellCoord


def ctmrg(env_sites: EnvSites, a_tilde_sites: TensorSites, unitcell_coord: UnitCellCoord, config: Config, H: Tensor, *, stop_on_convergence: bool = True, max_cycles: int = 2000, conv_tol: float = 1e-6, required_stable_cycles: int = 3, verbosity: int = 1, label: str = "CTMRG") -> tuple[EnvSites, dict]:
    """Compute final environments without modifying the input environments or bulk tensors."""
    if max_cycles < 2:
        raise ValueError("max_cycles must be at least 2.")
    if required_stable_cycles < 1:
        raise ValueError("required_stable_cycles must be a strictly positive integer.")
    env_sites = {site: replace(env) for site, env in env_sites.items()}
    cycle_width = len(str(max_cycles))
    start = last_update = perf_counter()
    update_interval = 0.1 if verbosity >= 3 else 1.0
    if verbosity >= 1:
        print(f"\n{label} | chi={config.chi} | tol={conv_tol:.2e}", flush=True)
    print_progress(f"  cycle={1:>{cycle_width}}/{max_cycles} | first cycle and energy measurement...", verbosity)
    A_sites = {site: double_layer(a_tilde) for site, a_tilde in a_tilde_sites.items()}
    env_sites = all_moves_sites(A_sites, env_sites, unitcell_coord, config)
    rdms = ctmrg_rdms(a_tilde_sites, env_sites, unitcell_coord)
    energies = stack([expectation_from_rdm(rho, H) for rho in rdms.values()]).reshape(len(a_tilde_sites), 2)
    stable_cycles = 0
    info = {
        "D": config.D,
        "chi": config.chi,
        "sites": tuple(a_tilde_sites),
        "tolerance": conv_tol,
        "max_cycles": max_cycles,
        "stop_on_convergence": stop_on_convergence,
        "required_stable_cycles": required_stable_cycles,
        "error_history": [],
        "threshold_history": [],
        "energy_history": [energies.sum().item() / len(a_tilde_sites)],
        "stable_cycles_history": [0],
    }
    errors = info["error_history"]
    thresholds = info["threshold_history"]
    for cycle in range(2, max_cycles + 1):
        old_energies = energies
        env_sites = all_moves_sites(A_sites, env_sites, unitcell_coord, config)
        rdms = ctmrg_rdms(a_tilde_sites, env_sites, unitcell_coord)
        energies = stack([expectation_from_rdm(rho, H) for rho in rdms.values()]).reshape(len(a_tilde_sites), 2)
        energy_per_site = energies.sum().item() / len(a_tilde_sites)
        info["energy_history"].append(energy_per_site)
        error = (energies - old_energies).abs().amax(dim=1)
        threshold = torch.full_like(error, conv_tol)
        errors.append(error.clone())
        thresholds.append(threshold.clone())
        converged = (error <= threshold).all()
        stable_cycles = stable_cycles + 1 if converged else 0
        info["stable_cycles_history"].append(stable_cycles)
        maximum_error = error.max().item()
        now = perf_counter()
        progress = f"  cycle={cycle:>{cycle_width}}/{max_cycles} | error={maximum_error:9.2e} | energy={energy_per_site:.12f} | stable={min(stable_cycles, required_stable_cycles)}/{required_stable_cycles} | time={format_duration(now - start)}"
        if verbosity >= 4 or cycle == 2 or now - last_update >= update_interval:
            print_progress(progress, verbosity)
            last_update = now
        if stop_on_convergence and stable_cycles >= required_stable_cycles:
            break
    print_progress(progress, verbosity, final=True)
    if stop_on_convergence and stable_cycles < required_stable_cycles:
        recent_errors = "\n".join(f"{error.max().item():.3e}" for error in errors[-10:])
        unconverged_sites = (~(errors[-1] <= thresholds[-1])).sum().item()
        threshold_min = thresholds[-1].min().item()
        threshold_max = thresholds[-1].max().item()
        warn(f"CTMRG did not converge after {max_cycles} cycles "
             f"(D={config.D}, chi={config.chi}).\n"
             f"Unconverged sites: {unconverged_sites}/{len(env_sites)}.\n"
             f"Consecutive cycles below tolerance: {stable_cycles}/{required_stable_cycles}.\n"
             f"Last errors, oldest to newest: \n{recent_errors}\n"
             f"Final per-site thresholds: {threshold_min:.2e} to {threshold_max:.2e}.\n"
             "Returning the last environments without convergence.\n",
             RuntimeWarning)
    if not stop_on_convergence:
        status = "fixed_iterations"
    elif stable_cycles >= required_stable_cycles:
        status = "converged"
    else:
        status = "not_converged"
    info.update({
        "status": status,
        "duration": perf_counter() - start,
        "total_cycles": cycle,
        "final_error": maximum_error,
        "stable_cycles": stable_cycles,
        "energy_per_peps_site": energy_per_site,
        "bond_energies": dict(zip(rdms, energies.detach().cpu().reshape(-1).tolist())),
    })
    return env_sites, info
