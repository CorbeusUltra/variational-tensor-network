"""Collect and plot CTMRG convergence diagnostics over a fixed number of cycles."""

import torch # noqa: I001
from torch import Tensor

from variational_tensor_network.ctmrg import run_ctmrg
from variational_tensor_network.initialize import initial_a, initial_env_sites, initial_single_and_double_layer_sites, uniform_boundary
from variational_tensor_network.models import heisenberg_bond
from variational_tensor_network.objects import Config
from variational_tensor_network.peps import double_layer
from variational_tensor_network.plots import plot_convergence, plot_convergence_comparison, plot_convergence_sites, plot_output_path
from variational_tensor_network.tilings import TILINGS, Site, UnitCellCoord


def collect_uniform_convergence(config: Config, H: Tensor, max_cycles: int = 100) -> tuple[dict[str, list[float]], list[float]]:
    """Collect single-site and multisite histories for the same uniform state."""
    a_tilde = initial_a(config)
    A = double_layer(a_tilde)
    delta = uniform_boundary(config)
    def single_site(x: int, y: int) -> Site:
        return (0, 0)
    setups = {
        "Single-site": (((0, 0),), single_site),
        "checkerboard": TILINGS["checkerboard"],
        "rectangle_2x3": TILINGS["rectangle_2x3"],
    }
    errors = {}
    thresholds = []
    for name, (sites_coords, unitcell_coord) in setups.items():
        a_tilde_sites = {site: a_tilde for site in sites_coords}
        A_sites = {site: A for site in sites_coords}
        env_sites = initial_env_sites(A_sites, delta, config)
        _, site_errors, site_thresholds = run_ctmrg(env_sites, a_tilde_sites, unitcell_coord, config, H, stop_on_convergence=False, max_cycles=max_cycles)
        errors[name] = torch.stack(site_errors).amax(dim=1).tolist()
        if not thresholds:
            thresholds = torch.stack(site_thresholds).amax(dim=1).tolist()
    return errors, thresholds


def collect_convergence_sites(sites_coords: tuple[Site, ...], unitcell_coord: UnitCellCoord, config: Config, H: Tensor, max_cycles: int = 100) -> tuple[list[Tensor], list[Tensor]]:
    """Collect per-site CTMRG histories for independently initialized site tensors."""
    a_tilde_sites, A_sites = initial_single_and_double_layer_sites(config, sites_coords)
    delta = uniform_boundary(config)
    env_sites = initial_env_sites(A_sites, delta, config)
    _, errors, thresholds = run_ctmrg(env_sites, a_tilde_sites, unitcell_coord, config, H, stop_on_convergence=False, max_cycles=max_cycles)
    return errors, thresholds



def main() -> None:
    config = Config()
    H = heisenberg_bond()
    torch.manual_seed(0)
    errors, thresholds = collect_uniform_convergence(config, H)
    output_path = plot_output_path(__file__, "single_site.pdf")
    plot_convergence(errors["Single-site"], thresholds, output_path=output_path)
    output_path = plot_output_path(__file__, "uniform_comparison.pdf")
    plot_convergence_comparison(errors, thresholds, output_path=output_path)
    tilings_names = ("checkerboard", "rectangle_2x3")
    for name in tilings_names:
        sites_coords, unitcell_coord = TILINGS[name]
        site_errors, site_thresholds = collect_convergence_sites(sites_coords, unitcell_coord, config, H)
        output_path = plot_output_path(__file__, f"{name}.pdf")
        plot_convergence_sites(site_errors, site_thresholds, sites_coords, unitcell_coord, name, show=True, output_path=output_path)


if __name__ == "__main__":
    main()
