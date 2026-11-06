"""Collect and plot CTMRG convergence diagnostics over a fixed number of cycles."""

import torch # noqa: I001
from torch import Tensor

from variational_tensor_network.ctmrg import ctmrg
from variational_tensor_network.initialize import initial_env_sites, initial_single_and_double_layer_sites, uniform_boundary
from variational_tensor_network.models import MODELS
from variational_tensor_network.objects import Config
from variational_tensor_network.plots import plot_ctmrg_convergence_by_site, plot_output_path
from variational_tensor_network.reporting import print_boxed, print_run_summary
from variational_tensor_network.tilings import TILINGS, Site, UnitCellCoord


def collect_convergence_sites(sites_coords: tuple[Site, ...], unitcell_coord: UnitCellCoord, config: Config, H: Tensor, max_cycles: int = 100, *, conv_tol: float = 1e-8, stop_on_convergence: bool = False, verbosity: int = 1) -> dict:
    """Collect per-site CTMRG histories for independently initialized site tensors."""
    a_tilde_sites, A_sites = initial_single_and_double_layer_sites(config, sites_coords)
    delta = uniform_boundary(config)
    env_sites = initial_env_sites(A_sites, delta, config)
    _, info = ctmrg(env_sites, a_tilde_sites, unitcell_coord, config, H, stop_on_convergence=stop_on_convergence, max_cycles=max_cycles, conv_tol=conv_tol, verbosity=verbosity)
    return info



def main() -> None:
    verbosity = 1
    model_name = "heisenberg" # "heisenberg", "ffb", "shastry_sutherland"
    tiling_name = "checkerboard" # "checkerboard", "abcd", "rectangle_2x3"
    D = 2  # 3, 4
    chi = 4  # 8, 16
    max_cycles = 100
    conv_tol = 1e-8
    stop_on_convergence = False
    parameters_by_model = {
        "heisenberg": {"J": 1.0, "h": 0},
        "ffb": {"JD": 2.0, "J": 1.0, "h": 0},
        "shastry_sutherland": {"JD": 2.0, "J": 1.0, "h": 0},
    }

    model = MODELS[model_name]
    parameters = parameters_by_model[model_name]
    parameters_text = ", ".join(f"{name}={value:g}" for name, value in parameters.items())
    parameters_tag = "_".join(f"{name}={value:g}" for name, value in parameters.items())
    H = model["bond"](**parameters_by_model[model_name])
    torch.manual_seed(0)
    config = Config(d=model["d"], D=D, chi=chi)
    sites_coords, unitcell_coord = TILINGS[tiling_name]
    info = collect_convergence_sites(sites_coords, unitcell_coord, config, H, max_cycles=max_cycles, conv_tol=conv_tol, stop_on_convergence=stop_on_convergence, verbosity=verbosity)
    if verbosity >= 1 :
        print_boxed(f"{model['label']} (d={model['d']}) ({parameters_text}) | {tiling_name} | D={D}, chi={chi}")
        print_run_summary("CTMRG", info)
    title = f"ctmrg_{model_name}_{tiling_name}_D={D}_chi={chi}_{parameters_tag}"
    output_path = plot_output_path(__file__, title + ".pdf")
    plot_ctmrg_convergence_by_site(info["error_history"], info["threshold_history"], sites_coords, unitcell_coord, title=title, show=True, output_path=output_path)
    # errors, thresholds = collect_uniform_convergence(config, H)
    # output_path = plot_output_path(__file__, "single_site.pdf")
    # plot_ctmrg_convergence(errors["Single-site"], thresholds, show=True, output_path=output_path)
    # output_path = plot_output_path(__file__, "uniform_comparison.pdf")
    # plot_uniform_ctmrg_comparison(errors, thresholds, show=True, output_path=output_path)


if __name__ == "__main__":
    main()


# Legacy code

# def collect_uniform_convergence(config: Config, H: Tensor, max_cycles: int = 100) -> tuple[dict[str, list[float]], list[float]]:
#     """Collect single-site and multisite histories for the same uniform state."""
#     a_tilde = initial_a(config)
#     A = double_layer(a_tilde)
#     delta = uniform_boundary(config)
#     def single_site(x: int, y: int) -> Site:
#         return (0, 0)
#     setups = {
#         "Single-site": (((0, 0),), single_site),
#         "checkerboard": TILINGS["checkerboard"],
#         "rectangle_2x3": TILINGS["rectangle_2x3"],
#     }
#     errors = {}
#     thresholds = []
#     for name, (sites_coords, unitcell_coord) in setups.items():
#         a_tilde_sites = {site: a_tilde for site in sites_coords}
#         A_sites = {site: A for site in sites_coords}
#         env_sites = initial_env_sites(A_sites, delta, config)
#         env_sites, info = ctmrg(env_sites, a_tilde_sites, unitcell_coord, config, H, stop_on_convergence=False, max_cycles=max_cycles)
#         site_errors = info["error_history"]
#         site_thresholds = info["threshold_history"]
#         errors[name] = torch.stack(site_errors).amax(dim=1).tolist()
#         if not thresholds:
#             thresholds = torch.stack(site_thresholds).amax(dim=1).tolist()
#     return errors, thresholds
