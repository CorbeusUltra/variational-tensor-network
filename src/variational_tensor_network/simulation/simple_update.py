import json
from collections.abc import Iterator
from hashlib import sha256
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal

import torch

from variational_tensor_network._utils.path import find_project_root
from variational_tensor_network._utils.reporting import (
    print_boxed,
    print_run_summary,
)
from variational_tensor_network.defaults import (
    DEFAULT_SU_MAX_STEPS,
    DEFAULT_SU_SCHEDULE,
)
from variational_tensor_network.initialize import (
    InitMethod,
    initial_simple_update_state,
    initial_state_from_lower_D_padding,
    initial_state_from_lower_D_svd,
)
from variational_tensor_network.models import MODELS, model_bond_hamiltonians
from variational_tensor_network.objects import BondOperators, BondWeights, Config
from variational_tensor_network.peps import TensorSites
from variational_tensor_network.plots.simple_update import (
    plot_simple_update_diagnostics,
)
from variational_tensor_network.simple_update import simple_update
from variational_tensor_network.tilings import TILINGS, Site, UnitCellCoord

type SUInitialization = InitMethod | Literal["lower_D_padding", "lower_D_svd", "nearby_h"]

root = find_project_root(__file__)



def matching_states(directory: Path, query: dict) -> Iterator[tuple[Path, dict]]:
    """Yield saved states matching every requested metadata field."""
    for path in sorted(directory.glob("*.pt")):
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        metadata = state["metadata"]
        if metadata.get("sweep_direction") is not None:
            continue
        if all(key in metadata and metadata[key] == value for key, value in query.items()):
            yield path, state


def is_usable_su_state(state: dict[str, Any]) -> bool:
    info = state["info_su"]
    return (info["status"] == "converged" and isfinite(info["energy_per_peps_site"]))


def find_lower_D_source(directory: Path, query: dict[str, Any], D: int) -> tuple[Path, dict] | None:
    for lower_D in range(D - 1, 1, -1):
        candidates = ((path, state) for path, state in matching_states(directory, {**query, "D": lower_D}) if is_usable_su_state(state))
        source = min(candidates, key=lambda item: item[1]["info_su"]["energy_per_peps_site"], default=None)
        if source is not None:
            return source
    return None


def load_nearby_h_source(source_path: Path, query: dict[str, Any], config: Config) -> tuple[Path, dict]:
    state = torch.load(source_path, map_location="cpu", weights_only=True)
    metadata = state["metadata"]
    source_params = metadata["parameters"]
    target_params = query["parameters"]
    source_couplings = {k: v for k, v in source_params.items() if k != "h"}
    target_couplings = {k: v for k, v in target_params.items() if k != "h"}
    if (metadata["model"] != query["model"] or metadata["tiling"] != query["tiling"] or metadata["D"] != config.D or metadata["d"] != config.d or "h" not in source_params or "h" not in target_params or source_couplings != target_couplings or not is_usable_su_state(state)):
        raise ValueError(f"Incompatible or unconverged SU checkpoint: {source_path}")
    return source_path, state



def initialize_simple_update(config: Config, sites_coords: tuple[Site, ...], unitcell_coord: UnitCellCoord, H: BondOperators, query: dict[str, Any], seed: int, *, method: SUInitialization | None, tau_svd: float, verbosity: int, source_path: Path | None = None) -> tuple[TensorSites, BondWeights, dict[str, Any]]:
    directory = root / "data" / "simple_update"
    match method:
        case "random" | "ones":
            source = None
        case "nearby_h":
            if source_path is None:
                raise ValueError("nearby_h requires the previous SU checkpoint.")
            source = load_nearby_h_source(source_path, query, config)
        case None | "lower_D_padding" | "lower_D_svd":
            source = find_lower_D_source(directory, query, config.D)
        case _:
            raise ValueError(f"Unknown simple-update initialization: {method!r}")
    if source is None:
        if method is None:
            if config.D > 2 and verbosity >= 1:
                print(f"No converged lower-D state found for D={config.D}. " "Using random initialization.")
            method = "random"
        if method != "random" and method != "ones":
            raise ValueError(f"No compatible checkpoint for {method!r} at D={config.D}.")
        torch.manual_seed(seed)
        a_sites, bond_weights = initial_simple_update_state(config, sites_coords, method=method)
        return a_sites, bond_weights, {"initialization": method, "source_D": None, "source_filename": None}
    source_path, previous = source
    if method == "nearby_h":
        return previous["a_sites"], previous["bond_weights"], {"initialization": "nearby_h", "source_D": previous["metadata"]["D"], "source_filename": source_path.name, "source_h": previous["metadata"]["parameters"]["h"]}
    source_D = previous["metadata"]["D"]
    if verbosity >= 1:
        print(f"Initializing D={config.D} from D={source_D}: {source_path}")
    a_sites = previous["a_sites"]
    bond_weights = previous["bond_weights"]
    if method is None:
        has_self_bonds = any(unitcell_coord(x + 1, y) == (x, y) or unitcell_coord(x, y + 1) == (x, y) for x, y in sites_coords)
        method = "lower_D_padding" if has_self_bonds else "lower_D_svd"
    if method == "lower_D_svd":
        a_sites, bond_weights = initial_state_from_lower_D_svd(a_sites, bond_weights, config.D, unitcell_coord, H, tau=tau_svd)
    else:
        a_sites, bond_weights = initial_state_from_lower_D_padding(a_sites, bond_weights, config.D)
    return a_sites, bond_weights, {"initialization": method, "source_D": source_D, "source_filename": source_path.name}


def compute_simple_update(model_name: str, parameters_by_model: dict[str, dict[str, float]], tiling_name: str, D: int, seed: int, schedule: tuple[tuple[float, float], ...], *, initialization: SUInitialization | None = None, max_sweeps: int, source_path: Path | None = None, verbosity: int = 1) -> dict[str, Any]:
    model = MODELS[model_name]
    parameters = parameters_by_model[model_name]
    parameters = {name: float(value) for name, value in parameters.items()}
    parameters_text = ", ".join(f"{name}={value:g}" for name, value in parameters.items())
    d = model["d"]
    config = Config(d=d, D=D)
    sites_coords, unitcell_coord = TILINGS[tiling_name]
    H = model_bond_hamiltonians(model_name, parameters, sites_coords)
    heading = f"{model['label']} ({parameters_text}) | {tiling_name} | (D={D}, d={d})"
    if verbosity >= 1:
        print_boxed(heading)
    query = {"model": model_name, "parameters": parameters, "tiling": tiling_name}
    if not schedule:
        raise ValueError("schedule must contain at least one tau stage.")
    a_sites, bond_weights, init_metadata = initialize_simple_update(config, sites_coords, unitcell_coord, H, query, seed, method=initialization, tau_svd=schedule[0][0], verbosity=verbosity, source_path=source_path)
    if verbosity >= 1:
        suffix = " (auto)" if initialization is None else ""
        print(f"Initialization: {init_metadata['initialization']}{suffix}", flush=True)
    a_sites, bond_weights, info_su = simple_update(a_sites, bond_weights, unitcell_coord, H, schedule=schedule, max_sweeps=max_sweeps, verbosity=verbosity)
    print_run_summary("Simple update", info_su, verbosity=verbosity)
    if verbosity >= 1:
        spins_per_peps_site = MODELS[model_name]["spins_per_peps_site"]
        energy_per_spin = info_su["energy_per_peps_site"] / spins_per_peps_site
        print(f"  Energy per spin: {energy_per_spin:.12f}\n", flush=True)
    metadata = {
        "schema": 2,
        "model": model_name,
        "parameters": parameters,
        "tiling": tiling_name,
        "D": config.D,
        "d": config.d,
        "dtype": str(next(iter(a_sites.values())).dtype),
        **init_metadata,
        "seed": seed,
        "schedule": schedule,
        "max_sweeps": max_sweeps,
    }
    state = {"metadata": metadata, "a_sites": a_sites, "bond_weights": bond_weights, "info_su": info_su}
    return state


def save_simple_update_state(state: dict[str, Any], output_dir: Path) -> Path:
    metadata = state["metadata"]
    identity = {key: value for key, value in metadata.items() if key in {"schema", "model", "parameters", "tiling", "D", "d"}}
    direction = metadata.get("sweep_direction")
    if direction is not None:
        identity["sweep_direction"] = direction
    key = json.dumps(identity, sort_keys=True).encode("utf-8")
    tag = sha256(key).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"{tag}.pt"
    with NamedTemporaryFile("wb", dir=output_dir, delete=False) as stream:
        torch.save(state, stream)
    Path(stream.name).replace(state_path)
    return state_path


def run_simple_update(model_name: str, parameters_by_model: dict[str, dict[str, float]], tiling_name: str, D: int, seed: int, schedule: tuple[tuple[float, float], ...] = DEFAULT_SU_SCHEDULE, *, max_sweeps: int = DEFAULT_SU_MAX_STEPS, initialization: SUInitialization | None = None, source_path: Path | None = None, verbosity: int = 1, sweep_direction: Literal["up", "down"] | None = None, show_fig: bool, save_fig: bool) -> tuple[Path, dict]:
    state = compute_simple_update(model_name, parameters_by_model, tiling_name, D, seed, schedule=schedule, initialization=initialization, max_sweeps=max_sweeps, source_path=source_path, verbosity=verbosity)
    state["metadata"]["sweep_direction"] = sweep_direction
    state_path = save_simple_update_state(state, root / "data" / "simple_update")
    if verbosity >= 1:
        print(f"\nSimple update saved under '{state_path}'.\n", flush=True)
        # print(f"Hash : {state_path.name}", flush=True)
    if show_fig or save_fig:
        info_su = state["info_su"]
        metadata = state["metadata"]
        plot_simple_update_diagnostics(info_su, metadata, show=show_fig, output_dir = root / "outputs" / "figures" / "simple_update" if save_fig else None)
    return state_path, state


def select_or_run_su(model_name: str, parameters_by_model: dict[str, dict[str, float]], tiling_name: str, D: int, seed: int, *, verbosity: int = 1) -> tuple[Path, dict[str, Any]]:
    parameters = parameters_by_model[model_name]
    query = {"model": model_name, "parameters": parameters, "tiling": tiling_name, "D": D}
    candidates = ((path, state) for path, state in matching_states(root / "data" / "simple_update", query) if is_usable_su_state(state))
    selected = min(candidates, key=lambda item: item[1]["info_su"]["energy_per_peps_site"], default=None)
    if selected is None:
        if verbosity >= 1:
            print("\nNo converged saved state found. Running Simple Update.", flush=True)
        return run_simple_update(model_name, parameters_by_model, tiling_name, D, seed, verbosity=verbosity, show_fig=False, save_fig=False)
    if verbosity >= 1:
        model = MODELS[model_name]
        parameters_text = ", ".join(f"{name}={value:g}" for name, value in parameters.items())
        print_boxed(f"{model['label']} ({parameters_text}) | {tiling_name} | (D={D}, d={model['d']})")
    return selected
