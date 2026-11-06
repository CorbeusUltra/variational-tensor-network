"""Run simple update and plot its convergence across a tau schedule."""

import torch  # noqa: I001
from torch import Tensor

from variational_tensor_network.initialize import initial_a_sites, initial_bond_weights
from variational_tensor_network.models import MODELS
from variational_tensor_network.objects import Config
from variational_tensor_network.plots import plot_output_path, plot_simple_update, plot_simple_update_by_stage
from variational_tensor_network.reporting import print_boxed, print_run_summary
from variational_tensor_network.simple_update import simple_update
from variational_tensor_network.tilings import TILINGS, Site, UnitCellCoord


def collect_convergence_sites(sites_coords: tuple[Site, ...], unitcell_coord: UnitCellCoord, config: Config, H: Tensor, *, schedule: tuple[tuple[float, float], ...], max_sweeps: int = 10000, verbosity: int = 1) -> dict:
    """Initialize independent site tensors and collect simple-update diagnostics."""
    a_sites = initial_a_sites(config, sites_coords)
    bond_weights = initial_bond_weights(a_sites)
    _, _, info = simple_update(a_sites, bond_weights, unitcell_coord, H, schedule=schedule, max_sweeps=max_sweeps, verbosity=verbosity)
    return info


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

    model = MODELS[model_name]
    parameters = parameters_by_model[model_name]
    parameters_text = ", ".join(f"{name}={value:g}" for name, value in parameters.items())
    parameters_tag = "_".join(f"{name}={value:g}" for name, value in parameters.items())
    H = model["bond"](**parameters)
    config = Config(d=model["d"], D=D)
    sites_coords, unitcell_coord = TILINGS[tiling_name]
    torch.manual_seed(seed)
    heading = f"{model['label']} (d={config.d}) ({parameters_text}) | {tiling_name} | D={D}"
    if verbosity >= 1:
        print_boxed(heading)
    info = collect_convergence_sites(sites_coords, unitcell_coord, config, H, schedule=schedule, max_sweeps=max_sweeps, verbosity=verbosity)
    print_run_summary("Simple update", info, verbosity=verbosity)
    title = f"su_{model_name}_{tiling_name}_D={D}_{parameters_tag}"
    output_path = plot_output_path(__file__, title + ".pdf")
    plot_simple_update(info, title=title, show=True, output_path=output_path)
    title = f"su_by_stage_{model_name}_{tiling_name}_D={D}_{parameters_tag}"
    output_path = plot_output_path(__file__, title + ".pdf")
    plot_simple_update_by_stage(info, title=title, show=True, output_path=output_path)


if __name__ == "__main__":
    main()
