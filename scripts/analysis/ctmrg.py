"""Measure a saved PEPS state at one CTMRG bond dimension."""


from collections.abc import Iterator # noqa: I001
from math import isfinite
from pathlib import Path

import torch

from variational_tensor_network.ctmrg import ctmrg
from variational_tensor_network.initialize import initial_boundary, initial_env_sites
from variational_tensor_network.models import MODELS
from variational_tensor_network.objects import Config
from variational_tensor_network.peps import (absorb_sqrt_weights, double_layer, get_site_weights)
from variational_tensor_network.plots import plot_ctmrg_convergence, plot_energy_vs_chi
from variational_tensor_network.plots._utils import find_project_root, plot_output_path
from variational_tensor_network.reporting import print_run_summary
from variational_tensor_network.tilings import TILINGS


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
        raise ValueError(f"No converged state matching {state_query} in '{state_dir}'.")
    state_path, state = selected
    metadata = state["metadata"]
    if verbosity >= 1:
        print(f"\nSelected state: {state_path}")
        print(f"SU schedule: {metadata['schedule']}")
        print(f"SU energy per PEPS site: {state['su_info']['energy_per_peps_site']:.12f}\n", flush=True)
    H = MODELS[metadata["model"]]["bond"](**metadata["parameters"])
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
    if not (show_fig or save_fig):
        return
    converged_chi = [chi for chi in chi_values if info_by_chi[chi]["status"] == "converged"]
    if len(converged_chi) >= 4:
        energies = [info_by_chi[chi]["energy_per_peps_site"] for chi in converged_chi]
        name = f"ctmrg_energy_vs_chi_{state_path.stem}"
        output_path = plot_output_path(__file__, f"{name}.pdf") if save_fig else None
        plot_energy_vs_chi(converged_chi, energies, metadata["D"], show=show_fig, output_path=output_path)
    if convergence_chi is not None:
        info = info_by_chi[convergence_chi]
        errors = [error.max().item() for error in info["error_history"]]
        name = f"ctmrg_convergence_{state_path.stem}_chi={convergence_chi}_tol={conv_tol:.0e}"
        output_path = plot_output_path(__file__, f"{name}.pdf") if save_fig else None
        plot_ctmrg_convergence(errors, show=show_fig, output_path=output_path)



def main() -> None:
    model_name = "heisenberg"  # "heisenberg", "ffb", "shastry_sutherland"
    parameters_by_model = {
        "heisenberg":                    {"J": 1.0, "h": 0},
        "ffb":                {"JD": 2.0, "J": 1.0, "h": 0},
        "shastry_sutherland": {"JD": 2.0, "J": 1.0, "h": 0},
    }
    tiling_name = "checkerboard"  # "checkerboard", "abcd", "rectangle_2x3"
    D = 2
    seed = 0

    n_chi = 10
    chi_values = [16 + D * i for i in range(n_chi)] # chi_values_by_D = {D: [D**2 + i for i in range(n_chi)] for D in D_values} ; chi_values_by_D = {D: [D**2 + D * i for i in range(n_chi)] for D in D_values} ; chi_values_by_D = {D: [16 + 2 * D**2 * i for i in range(n_chi)] for D in D_values} ; chi_values_by_D = {D: [16 + D * i for i in range(n_chi)] for D in D_values}
    convergence_chi = chi_values[-1]  # chi_values[-1] | None
    conv_tol = 1e-8
    max_cycles = 2000

    verbosity = 1
    show_fig = True
    save_fig = True

    run_ctmrg(model_name, parameters_by_model, tiling_name, D, seed, chi_values, max_cycles=max_cycles, conv_tol=conv_tol, verbosity=verbosity, convergence_chi=convergence_chi, show_fig=show_fig, save_fig=save_fig)


if __name__ == "__main__":
    main()
