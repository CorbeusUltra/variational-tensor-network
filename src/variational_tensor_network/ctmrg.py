"""Full CTMRG algorithm."""

# from math import isfinite
from warnings import warn

import torch
from torch import Tensor

from .ctmrg_moves import all_moves, all_moves_sites
from .objects import Config, Environment, EnvSites

# from .observables import ctmrg_rdms, expectation_from_rdm
from .peps import TensorSites  # , double_layer
from .tilings import UnitCellCoord


def run_ctmrg(env: Environment, A: Tensor, config: Config, *, stop_on_convergence: bool = True, max_iter: int = 1000, conv_atol: float = 1e-10, conv_rtol: float = 1e-10) -> tuple[Environment, list[float], list[float]]:
    if max_iter < 2:
        raise ValueError("max_iter must be at least 2.")
    env, xi_new = all_moves(A, env, config)
    errors: list[float] = []
    thresholds: list[float] = []
    for _ in range(max_iter - 1):
        xi_old = xi_new
        env, xi_new = all_moves(A, env, config)
        error = (xi_new - xi_old).abs().max().item()
        scale = max(xi_old.abs().max().item(), xi_new.abs().max().item())
        threshold = conv_atol + conv_rtol * scale
        errors.append(error)
        thresholds.append(threshold)
        if stop_on_convergence and error <= threshold:
            return env, errors, thresholds
    if stop_on_convergence:
        recent_errors = "\n".join(f"{error:.3e}" for error in errors[-10:])
        warn(f"CTMRG did not converge after {max_iter} cycles "
             f"(D={config.D}, chi={config.chi}).\n"
             f"Last errors, oldest to newest: \n{recent_errors}\n"
             f"Final threshold: {thresholds[-1]:.2e}.\n"
             "Returning the last environment without convergence.\n",
             RuntimeWarning)
    return env, errors, thresholds


def run_ctmrg_sites(env_sites: EnvSites, A_sites: TensorSites, unitcell_coord: UnitCellCoord, config: Config, *, stop_on_convergence: bool = True, max_iter: int = 1000, conv_atol: float = 1e-6, conv_rtol: float = 1e-6) -> tuple[EnvSites, list[Tensor], list[Tensor]]:
    if max_iter < 2:
        raise ValueError("max_iter must be at least 2.")
    env_sites, xi_new = all_moves_sites(A_sites, env_sites, unitcell_coord, config)
    errors: list[Tensor] = []
    thresholds: list[Tensor] = []
    for _ in range(max_iter - 1):
        xi_old = xi_new
        env_sites, xi_new = all_moves_sites(A_sites, env_sites, unitcell_coord, config)
        error = (xi_new - xi_old).abs().amax(dim=0)
        scale = torch.maximum(xi_old.abs().amax(dim=0), xi_new.abs().amax(dim=0))
        threshold = conv_atol + conv_rtol * scale
        errors.append(error.detach().cpu())
        thresholds.append(threshold.detach().cpu())
        if stop_on_convergence and (error <= threshold).all().item():
            return env_sites, errors, thresholds
    if stop_on_convergence:
        recent_errors = "\n".join(f"{error.max().item():.3e}" for error in errors[-10:])
        unconverged_sites = (~(errors[-1] <= thresholds[-1])).sum().item()
        threshold_min = thresholds[-1].min().item()
        threshold_max = thresholds[-1].max().item()
        warn(f"CTMRG did not converge after {max_iter} cycles "
             f"(D={config.D}, chi={config.chi}).\n"
             f"Unconverged sites: {unconverged_sites}/{len(env_sites)}.\n"
             f"Last maximum errors, oldest to newest: \n{recent_errors}\n"
             f"Final per-site thresholds: {threshold_min:.2e} to {threshold_max:.2e}.\n"
             "Returning the last environments without convergence.\n",
             RuntimeWarning)
    return env_sites, errors, thresholds
