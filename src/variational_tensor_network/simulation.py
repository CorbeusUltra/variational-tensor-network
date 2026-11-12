"""Provide functions to optimize peps state, and measure them."""

import json
from collections.abc import Iterator
from hashlib import sha256
from itertools import combinations
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from variational_tensor_network._utils.path import find_project_root
from variational_tensor_network._utils.reporting import (
    print_boxed,
    print_run_summary,
)
from variational_tensor_network.ctmrg import ctmrg
from variational_tensor_network.initialize import (
    initial_a_sites,
    initial_bond_weights,
    initial_boundary,
    initial_env_sites,
)
from variational_tensor_network.models import MODELS
from variational_tensor_network.objects import Config
from variational_tensor_network.observables import (
    magnetization_x,
    magnetization_z,
    spin_spin_correlation,
)
from variational_tensor_network.peps import (
    absorb_sqrt_weights,
    double_layer,
    get_site_weights,
)
from variational_tensor_network.plots.ctmrg import (
    plot_ctmrg_convergence,
    plot_energy_vs_chi,
)
from variational_tensor_network.plots.simple_update import (
    plot_simple_update,
    plot_simple_update_by_stage,
)
from variational_tensor_network.simple_update import simple_update
from variational_tensor_network.tilings import TILINGS

root = find_project_root(__file__)


def run_simple_update(model_name: str, parameters_by_model: dict[str, dict[str, float]], tiling_name: str, D: int, seed: int, schedule: tuple[tuple[float, float], ...] = ((.05, 1e-4), (.01, 1e-6), (.005, 1e-8)), *, max_sweeps: int = 10000, verbosity: int = 1, show_fig: bool, save_fig: bool) -> tuple[Path, dict]:
    model = MODELS[model_name]
    parameters = parameters_by_model[model_name]
    parameters = {name: float(value) for name, value in parameters.items()}
    parameters_text = ", ".join(f"{name}={value:g}" for name, value in parameters.items())
    H = model["bond"](**parameters)
    d = model["d"]
    config = Config(d=d, D=D)
    sites_coords, unitcell_coord = TILINGS[tiling_name]
    torch.manual_seed(seed)
    heading = f"{model['label']} ({parameters_text}) | {tiling_name} | (D={D}, d={d})"
    if verbosity >= 1:
        print_boxed(heading)
    a_sites = initial_a_sites(config, sites_coords)
    bond_weights = initial_bond_weights(a_sites)
    a_sites, bond_weights, info_su = simple_update(a_sites, bond_weights, unitcell_coord, H, schedule=schedule, max_sweeps=max_sweeps, verbosity=verbosity)
    print_run_summary("Simple update", info_su, verbosity=verbosity)
    if verbosity >= 1:
        spins_per_peps_site = MODELS[model_name]["spins_per_peps_site"]
        energy_per_spin = info_su["energy_per_peps_site"] / spins_per_peps_site
        print(f"  Energy per spin: {energy_per_spin:.12f}\n", flush=True)
    metadata = {
        "schema": 1,
        "model": model_name,
        "parameters": parameters,
        "tiling": tiling_name,
        "D": config.D,
        "d": config.d,
        "dtype": str(next(iter(a_sites.values())).dtype),
        "initialization": "random",
        "seed": seed,
        "schedule": schedule,
        "max_sweeps": max_sweeps,
    }
    key = json.dumps(metadata, sort_keys=True).encode("utf-8")
    tag = sha256(key).hexdigest()
    output_dir = root / "data" / "simple_update"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"{tag}.pt"
    state = {
        "metadata": metadata,
        "a_sites": a_sites,
        "bond_weights": bond_weights,
        "info_su": info_su,
    }
    with NamedTemporaryFile("wb", dir=output_dir, delete=False) as stream:
        torch.save(state, stream)
    Path(stream.name).replace(state_path)
    if verbosity >= 1:
        print(f"\nSimple update saved under '{state_path}'.\n", flush=True)
    if show_fig or save_fig:
        parameters_tag = "_".join(f"{key}={value}" for key, value in metadata["parameters"].items())
        plot_tag = f"{metadata['model']}_{metadata['tiling']}_D={metadata['D']}_{parameters_tag}_seed={metadata['seed']}"
        for prefix, plot in (("su", plot_simple_update), ("su_by_stage", plot_simple_update_by_stage)):
            output_path = root / "outputs" / "figures" / "analysis" / f"{prefix}_{plot_tag}" if save_fig else None
            plot(info_su, show=show_fig, output_path=output_path)
    return state_path, state



def matching_states(directory: Path, query: dict) -> Iterator[tuple[Path, dict]]:
    """Yield saved states matching every requested metadata field."""
    for path in sorted(directory.glob("*.pt")):
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        metadata = state["metadata"]
        if all(key in metadata and metadata[key] == value for key, value in query.items()):
            yield path, state


def run_ctmrg(model_name: str, parameters_by_model: dict[str, dict[str, float]], tiling_name: str, D: int, seed: int, *, max_chi_iterations: int = 10, conv_tol_chi: float = 1e-6, max_ctmrg_cycles: int = 2000, conv_tol_ctmrg: float = 1e-7, verbosity: int = 1, show_fig: bool = False, save_fig: bool = False, convergence_last_chi: bool = False) -> dict:
    parameters = parameters_by_model[model_name]
    state_query = {
        "model": model_name,
        "parameters": parameters,
        "tiling": tiling_name,
        "D": D,
        "seed": seed,
    }
    state_dir = root / "data" / "simple_update"
    candidates = ((path, state) for path, state in matching_states(state_dir, state_query) if state["info_su"]["status"] == "converged" and isfinite(state["info_su"]["energy_per_peps_site"]))
    selected = min(candidates, key=lambda item: item[1]["info_su"]["energy_per_peps_site"], default=None)
    if selected is None:
        if verbosity >= 1:
            print("\nNo converged saved state found. Running Simple Update.", flush=True)
        selected = run_simple_update(model_name, parameters_by_model, tiling_name, D, seed, verbosity=verbosity, show_fig=False, save_fig=False)
    elif verbosity >= 1:
        model = MODELS[model_name]
        parameters_text = ", ".join(f"{name}={value:g}" for name, value in parameters.items())
        heading = f"{model['label']} ({parameters_text}) | {tiling_name} | (D={D}, d={model["d"]})"
        print_boxed(heading)
    state_path, state = selected
    info_su = state["info_su"]
    if not(info_su["status"] == "converged" and isfinite(info_su["energy_per_peps_site"])):
            raise RuntimeError(f"Simple Update did not produce a converged finite-energy state. Saved result: {state_path}")
    metadata = state["metadata"]
    H = MODELS[metadata["model"]]["bond"](**metadata["parameters"])
    spins_per_peps_site = MODELS[model_name]["spins_per_peps_site"]
    energy_per_spin = state['info_su']['energy_per_peps_site'] / spins_per_peps_site
    if verbosity >= 1:
        print(f"\nSelected state: {state_path}")
        print(f"SU schedule: {metadata['schedule']}")
        print(f"SU energy per spin: {energy_per_spin:.12f}\n", flush=True)
    _, unitcell_coord = TILINGS[metadata["tiling"]]
    a_tilde_sites = {}
    for site, a in state["a_sites"].items():
        weights = get_site_weights(site, state["bond_weights"], unitcell_coord)
        a_tilde_sites[site] = absorb_sqrt_weights(a, weights)
    A_sites = {site: double_layer(a_tilde) for site, a_tilde in a_tilde_sites.items()}
    chi = D**2 + 16
    config = Config(D=metadata["D"], d=metadata["d"], chi=chi)
    env_sites = initial_env_sites(A_sites, initial_boundary(config), config)
    _, info_ctmrg = ctmrg(env_sites, a_tilde_sites, unitcell_coord, config, H, max_cycles=max_ctmrg_cycles, conv_tol=conv_tol_ctmrg, verbosity=verbosity, label=f"CTMRG first iteration (max={max_chi_iterations})")
    print_run_summary("CTMRG", info_ctmrg, verbosity)
    energy_per_spin = info_ctmrg["energy_per_peps_site"] / spins_per_peps_site
    info = {"history_ctmrg" : {chi : info_ctmrg}}
    for index in range(2, max_chi_iterations + 1):
        chi += 2*D
        old_energy_per_spin = energy_per_spin
        config = Config(D=metadata["D"], d=metadata["d"], chi=chi)
        env_sites = initial_env_sites(A_sites, initial_boundary(config), config)
        _, info_ctmrg = ctmrg(env_sites, a_tilde_sites, unitcell_coord, config, H, max_cycles=max_ctmrg_cycles, conv_tol=conv_tol_ctmrg, verbosity=verbosity, label=f"CTMRG {index} (max={max_chi_iterations})")
        info["history_ctmrg"][chi] = info_ctmrg
        print_run_summary("CTMRG", info_ctmrg, verbosity)
        energy_per_spin = info_ctmrg["energy_per_peps_site"] / spins_per_peps_site
        rdms = info_ctmrg["bond_rdms"]
        n_spins_per_bond = 2 * spins_per_peps_site
        m_z = 0
        converged = False
        for bond, rho in rdms.items():
            site, direction = bond
            if direction == "r":
                if verbosity >= 1:
                    print(f"\n  PEPS site : {site} ")
                for i in range(1, spins_per_peps_site + 1):
                    S_z = magnetization_z(i,n_spins_per_bond, rho)
                    if verbosity >= 1:
                        S_x = magnetization_x(i,n_spins_per_bond, rho)
                        print(f"    spin {i} | <S_{i}^z> : {S_z:+.12f} | <S_{i}^x> : {S_x:+.12f}")
                    m_z += S_z
            if verbosity >= 1:
                print(f"\n    bond direction : {direction}")
            for i, j in combinations(range(1, n_spins_per_bond + 1), 2):
                correlation = spin_spin_correlation(i, j, n_spins_per_bond, rho)
                if verbosity >= 1:
                    print(f"      spin couple ({i}, {j}) | <S_{i}·S_{j}> = {correlation:+.12f}")
        n_sites = len(A_sites)
        m_z_over_m_sat = m_z / (n_sites * spins_per_peps_site / 2)
        if verbosity >= 1:
            print(f"\n  m_z/m_sat: {m_z_over_m_sat:.12f}\n")
            print(f"\n  Energy per spin: {energy_per_spin:.12f}\n")
        error = abs(energy_per_spin - old_energy_per_spin)
        if error < conv_tol_chi:
            info["status"] = "converged"
            info["m_z_over_m_sat"] = m_z_over_m_sat
            converged = True
            break
    if converged is False:
        info["status"] = "not_converged"
        info["m_z_over_m_sat"] = m_z_over_m_sat
    converged_chi = [chi for chi in info["history_ctmrg"] if info["history_ctmrg"][chi]["status"] == converged]
    if not (show_fig or save_fig):
        return info
    parameters_tag = "_".join(f"{key}={value}" for key, value in metadata["parameters"].items())
    plot_tag = f"{metadata['model']}_{metadata['tiling']}_D={metadata['D']}_{parameters_tag}_seed={metadata['seed']}"
    if len(converged_chi) >= 4:
        energies = [info["history_ctmrg"][chi]["energy_per_peps_site"] / spins_per_peps_site for chi in converged_chi]
        name = f"ctmrg_energy_vs_chi_{plot_tag}"
        output_path = root / "outputs" / "figures" / "analysis" / name if save_fig else None
        plot_energy_vs_chi(converged_chi, energies, metadata["D"], show=show_fig, output_path=output_path)
    if convergence_last_chi:
        info_chi = info["history_ctmrg"][chi]
        errors = [error.max().item() for error in info_chi["error_history"]]
        name = f"ctmrg_convergence_{plot_tag}_chi={chi}_conv_tol_ctmrg={conv_tol_ctmrg:.0e}"
        output_path = root / "outputs" / "figures" / "analysis" / name if save_fig else None
        plot_ctmrg_convergence(errors, show=show_fig, output_path=output_path)
    return info
