import json
from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch

from variational_tensor_network._utils.path import find_project_root
from variational_tensor_network._utils.reporting import (
    print_boxed,
    print_run_summary,
)
from variational_tensor_network.initialize import (
    initial_a_sites,
    initial_bond_weights,
)
from variational_tensor_network.models import MODELS
from variational_tensor_network.objects import Config
from variational_tensor_network.plots.simple_update import (
    plot_simple_update_diagnostics,
)
from variational_tensor_network.simple_update import simple_update
from variational_tensor_network.tilings import TILINGS

root = find_project_root(__file__)



def matching_states(directory: Path, query: dict) -> Iterator[tuple[Path, dict]]:
    """Yield saved states matching every requested metadata field."""
    for path in sorted(directory.glob("*.pt")):
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        metadata = state["metadata"]
        if all(key in metadata and metadata[key] == value for key, value in query.items()):
            yield path, state


def compute_simple_update(model_name: str, parameters_by_model: dict[str, dict[str, float]], tiling_name: str, D: int, seed: int, schedule: tuple[tuple[float, float], ...] = ( (0.1, 1e-7), (0.01, 1e-8), (0.001, 1e-9), ), *, max_sweeps: int = 2000, verbosity: int = 1) -> dict[str, Any]:
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
    state = {"metadata": metadata, "a_sites": a_sites, "bond_weights": bond_weights, "info_su": info_su}
    return state


def save_simple_update_state(state: dict[str, Any], output_dir: Path) -> Path:
    metadata = state["metadata"]
    identity = {key: value for key, value in metadata.items() if key not in {"initialization", "seed", "schedule", "dtype", "max_sweeps"}}
    key = json.dumps(identity, sort_keys=True).encode("utf-8")
    tag = sha256(key).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"{tag}.pt"
    with NamedTemporaryFile("wb", dir=output_dir, delete=False) as stream:
        torch.save(state, stream)
    Path(stream.name).replace(state_path)
    return state_path


def run_simple_update(model_name: str, parameters_by_model: dict[str, dict[str, float]], tiling_name: str, D: int, seed: int, schedule: tuple[tuple[float, float], ...] = ((.1, 1e-7), (.01, 5e-8), (.001, 1e-8)), *, max_sweeps: int = 3000, verbosity: int = 1, show_fig: bool, save_fig: bool) -> tuple[Path, dict]:
    state = compute_simple_update(model_name, parameters_by_model, tiling_name, D, seed, schedule=schedule, max_sweeps=max_sweeps, verbosity=verbosity)
    state_path = save_simple_update_state(state, root / "data" / "simple_update")
    if verbosity >= 1:
        print(f"\nSimple update saved under '{state_path}'.\n", flush=True)
    if show_fig or save_fig:
        info_su = state["info_su"]
        metadata = state["metadata"]
        plot_simple_update_diagnostics(info_su, metadata, show=show_fig, output_dir=root / "outputs" / "figures" / "analysis" if save_fig else None)
    return state_path, state
