"""Run simple update and plot its convergence across a tau schedule."""

import json # noqa: I001
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from variational_tensor_network.initialize import initial_a_sites, initial_bond_weights
from variational_tensor_network.models import MODELS
from variational_tensor_network.objects import Config
from variational_tensor_network.plots import (plot_output_path, plot_simple_update, plot_simple_update_by_stage)
from variational_tensor_network.plots._utils import find_project_root
from variational_tensor_network.reporting import print_boxed, print_run_summary
from variational_tensor_network.simple_update import simple_update
from variational_tensor_network.tilings import TILINGS


def run_simple_update(model_name: str, parameters_by_model: dict[str, dict[str, float]], tiling_name: str, D: int, seed: int, schedule: tuple[tuple[float, float], ...], *, max_sweeps: int = 10000, verbosity: int = 1, show_fig: bool, save_fig: bool) -> None:
    model = MODELS[model_name]
    parameters = parameters_by_model[model_name]
    parameters_text = ", ".join(f"{name}={value:g}" for name, value in parameters.items())
    parameters_tag = "_".join(f"{name}={value:g}" for name, value in parameters.items())  # noqa: F841
    H = model["bond"](**parameters)
    config = Config(d=model["d"], D=D)
    sites_coords, unitcell_coord = TILINGS[tiling_name]
    torch.manual_seed(seed)
    heading = f"{model['label']} (d={config.d}) ({parameters_text}) | {tiling_name} | D={D}"
    if verbosity >= 1:
        print_boxed(heading)

    a_sites = initial_a_sites(config, sites_coords)
    bond_weights = initial_bond_weights(a_sites)
    a_sites, bond_weights, info = simple_update(a_sites, bond_weights, unitcell_coord, H, schedule=schedule, max_sweeps=max_sweeps, verbosity=verbosity)
    print_run_summary("Simple update", info, verbosity=verbosity)
    metadata = {
        "schema": 1,
        "model": model_name,
        "parameters": dict(parameters),
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
    tag = sha256(key).hexdigest()[:12]
    state_id = f"{model_name}_{tiling_name}_D={D}_d={config.d}_seed={seed}_{tag}"
    output_dir = find_project_root(__file__) / "data" / "simple_update"
    output_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"{state_id}.pt"
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
        for prefix, plot in (("su", plot_simple_update), ("su_by_stage", plot_simple_update_by_stage)):
            title = f"{prefix}_{state_path.stem}"
            output_path = None
            if save_fig:
                output_path = plot_output_path(__file__, title + ".pdf")
            plot(info, title=title, show=show_fig, output_path=output_path)


def main() -> None:
    verbosity = 1
    model_name = "heisenberg"  # "heisenberg", "ffb", "shastry_sutherland"
    tiling_name = "checkerboard"  # "checkerboard", "abcd", "rectangle_2x3"
    D = 2  # 3, 4
    max_sweeps = 10000  # Maximum sweeps per tau stage
    seed = 0
    schedule = ((.1, 1e-5), (.01, 1e-6), (.001, 1e-7)) # schedule = ((.01, 1e-6), (.001, 1e-7)) ; schedule = ((.001, 1e-7),) ; schedule = ((.1, 1e-5), (.01, 1e-7), (.001, 1e-8))
    parameters_by_model = {
        "heisenberg": {"J": 1.0, "h": 0},
        "ffb": {"JD": 2.0, "J": 1.0, "h": 0},
        "shastry_sutherland": {"JD": 2.0, "J": 1.0, "h": 0},
    }
    show_fig = False
    save_fig = False
    run_simple_update(model_name, parameters_by_model, tiling_name, D, seed, schedule, max_sweeps=max_sweeps, verbosity=verbosity, show_fig=show_fig, save_fig=save_fig)


if __name__ == "__main__":
    main()
