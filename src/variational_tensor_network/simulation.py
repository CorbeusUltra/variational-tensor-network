"""Provide functions to optimize peps state, and measure them."""

import json
from collections.abc import Iterator
from hashlib import sha256
from itertools import combinations
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

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
from variational_tensor_network.plots import (
    plot_ctmrg_convergence,
    plot_energy_vs_chi,
    plot_simple_update,
    plot_simple_update_by_stage,
)
from variational_tensor_network.plots._utils import find_project_root
from variational_tensor_network.reporting import print_boxed, print_run_summary
from variational_tensor_network.simple_update import simple_update
from variational_tensor_network.tilings import TILINGS


def run_simple_update(model_name: str, parameters_by_model: dict[str, dict[str, float]], tiling_name: str, D: int, seed: int, schedule: tuple[tuple[float, float], ...] = ((.1, 1e-4), (.01, 1e-6), (.001, 1e-8)), *, max_sweeps: int = 10000, verbosity: int = 1, show_fig: bool, save_fig: bool) -> tuple[Path, dict]:
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
    a_sites, bond_weights, info = simple_update(a_sites, bond_weights, unitcell_coord, H, schedule=schedule, max_sweeps=max_sweeps, verbosity=verbosity)
    print_run_summary("Simple update", info, verbosity=verbosity)
    if verbosity >= 1:
        spins_per_peps_site = MODELS[model_name]["spins_per_peps_site"]
        energy_per_spin = info["energy_per_peps_site"] / spins_per_peps_site
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
    root = find_project_root(__file__)
    output_dir = root / "data" / "simple_update"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"{tag}.pt"
    state = {
        "metadata": metadata,
        "a_sites": a_sites,
        "bond_weights": bond_weights,
        "su_info": info,
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
            output_path = root / "outputs" / "analysis" / f"{prefix}_{plot_tag}" if save_fig else None
            plot(info, show=show_fig, output_path=output_path)
    return state_path, state



def matching_states(directory: Path, query: dict) -> Iterator[tuple[Path, dict]]:
    """Yield saved states matching every requested metadata field."""
    for path in sorted(directory.glob("*.pt")):
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        metadata = state["metadata"]
        if all(key in metadata and metadata[key] == value for key, value in query.items()):
            yield path, state


def run_ctmrg(model_name: str, parameters_by_model: dict[str, dict[str, float]], tiling_name: str, D: int, seed: int, chi_values: list[int], *, max_cycles: int = 2000, conv_tol: float = 1e-8, verbosity: int = 1, convergence_chi: int | None = None, show_fig: bool = False, save_fig: bool = False) -> None:
    parameters = parameters_by_model[model_name]
    state_query = {
        "model": model_name,
        "parameters": parameters,
        "tiling": tiling_name,
        "D": D,
        "seed": seed,
    }
    root = find_project_root(__file__)
    state_dir = root / "data" / "simple_update"
    candidates = ((path, state) for path, state in matching_states(state_dir, state_query) if state["su_info"]["status"] == "converged" and isfinite(state["su_info"]["energy_per_peps_site"]))
    selected = min(candidates, key=lambda item: item[1]["su_info"]["energy_per_peps_site"], default=None)
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
    su_info = state["su_info"]
    if not(su_info["status"] == "converged" and isfinite(su_info["energy_per_peps_site"])):
            raise RuntimeError(f"Simple Update did not produce a converged finite-energy state. Saved result: {state_path}")
    metadata = state["metadata"]
    H = MODELS[metadata["model"]]["bond"](**metadata["parameters"])
    spins_per_peps_site = MODELS[model_name]["spins_per_peps_site"]
    energy_per_spin = state['su_info']['energy_per_peps_site'] / spins_per_peps_site
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
    info_by_chi = {}
    for index, chi in enumerate(chi_values, start=1):
        config = Config(D=metadata["D"], d=metadata["d"], chi=chi)
        env_sites = initial_env_sites(A_sites, initial_boundary(config), config)
        label = f"CTMRG {index}/{len(chi_values)}"
        _, info = ctmrg(env_sites, a_tilde_sites, unitcell_coord, config, H, max_cycles=max_cycles, conv_tol=conv_tol, verbosity=verbosity, label=label)
        info_by_chi[chi] = info
        print_run_summary("CTMRG", info, verbosity)
        if verbosity >= 1:
            energy_per_spin = info["energy_per_peps_site"] / spins_per_peps_site
            rdms = info["bond_rdms"]
            n_spins_bonds = 2 * spins_per_peps_site
            for bond, rho in rdms.items():
                site, direction = bond
                if direction == "r":
                    print(f"\n  PEPS site : {site} ")
                    for i in range(1, n_spins_bonds + 1):
                        m_z = magnetization_z(i,n_spins_bonds, rho)
                        m_x = magnetization_x(i,n_spins_bonds, rho)
                        print(f"    spin {i} | <S_{i}^z> : {m_z:+.12f} | <S_{i}^x> : {m_x:+.12f}")
                print(f"\n    bond direction : {direction}")
                for i, j in combinations(range(1, n_spins_bonds + 1), 2):
                    correlation = spin_spin_correlation(i, j, n_spins_bonds, rho)
                    print(f"      spin couple ({i}, {j}) | <S_{i}·S_{j}> = {correlation:+.12f}")
            print(f"\n  Energy per spin: {energy_per_spin:.12f}\n")
    if not (show_fig or save_fig):
        return
    converged_chi = [chi for chi in chi_values if info_by_chi[chi]["status"] == "converged"]
    parameters_tag = "_".join(f"{key}={value}" for key, value in metadata["parameters"].items())
    plot_tag = f"{metadata['model']}_{metadata['tiling']}_D={metadata['D']}_{parameters_tag}_seed={metadata['seed']}"
    if len(converged_chi) >= 4:
        energies = [info_by_chi[chi]["energy_per_peps_site"] / spins_per_peps_site for chi in converged_chi]
        name = f"ctmrg_energy_vs_chi_{plot_tag}"
        output_path = root / "outputs" / "analysis" / name if save_fig else None
        plot_energy_vs_chi(converged_chi, energies, metadata["D"], show=show_fig, output_path=output_path)
    if convergence_chi is not None:
        info = info_by_chi[convergence_chi]
        errors = [error.max().item() for error in info["error_history"]]
        name = f"ctmrg_convergence_{plot_tag}_chi={convergence_chi}_tol={conv_tol:.0e}"
        output_path = root / "outputs" / "analysis" / name if save_fig else None
        plot_ctmrg_convergence(errors, show=show_fig, output_path=output_path)
