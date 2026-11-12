"""Export tiling diagrams with labels identifying shared bond weights."""

from time import perf_counter

import torch

from variational_tensor_network._utils.path import find_project_root
from variational_tensor_network.initialize import initial_a_sites, initial_bond_weights
from variational_tensor_network.objects import Config
from variational_tensor_network.plots.tilings import plot_weights
from variational_tensor_network.tilings import TILINGS


def main() -> None:
    start = perf_counter()
    torch.manual_seed(0)
    config = Config()
    for name, (sites_coords, unitcell_coord) in TILINGS.items():
        a_sites = initial_a_sites(config, sites_coords)
        bond_weights = initial_bond_weights(a_sites)
        root = find_project_root(__file__)
        output_path = root / "outputs" / "figures" / "tilings" / "weighted" / name
        plot_weights(sites_coords, unitcell_coord, bond_weights, title=name, show=True, output_path=output_path)
    print(f"\nExported {len(TILINGS)} diagrams in {perf_counter() - start:.2g} s.\n")


if __name__ == "__main__":
    main()
