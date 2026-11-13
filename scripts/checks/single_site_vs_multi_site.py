"""Check uniform-state agreement between single-site and multisite CTMRG."""

import torch # noqa: I001
from time import perf_counter

from variational_tensor_network import initialize
from variational_tensor_network.ctmrg import all_moves, all_moves_sites
from variational_tensor_network.objects import Config
from variational_tensor_network.peps import double_layer
from variational_tensor_network.tilings import TILINGS, Site, UnitCellCoord


def check_tilings(sites_coords: tuple[Site, ...], unitcell_coord: UnitCellCoord, config: Config, steps: int = 5) -> None:
    """Compare all environment tensors after each CTMRG cycle for a uniform state."""
    torch.manual_seed(0)
    a = initialize.initial_a(config)
    A = double_layer(a)
    delta = initialize.initial_boundary(config)
    A_sites = {site: A for site in sites_coords}
    env_sites = initialize.initial_env_sites(A_sites, delta, config)
    env_reference = initialize.initial_env(delta, A, config)
    for _ in range(steps):
        env_reference = all_moves(A, env_reference, config)
        env_sites = all_moves_sites(A_sites, env_sites, unitcell_coord, config)
        for env in env_sites.values():
            for actual, expected in zip(env, env_reference):
                torch.testing.assert_close(actual, expected, atol=1e-8, rtol=1e-8)
    print(f"OK | uniform reference: {steps} matching CTMRG cycles")


def print_tiling(sites_coord: tuple[Site, ...], unitcell_coord: UnitCellCoord, width: int = 5, height: int = 5) -> None:
    labels = {site: i for i, site in enumerate(sites_coord)}
    for y in range(height):
        print(" ".join(f"[{labels[unitcell_coord(x, y)]}]" for x in range(width)))



def main() -> None:
    start = perf_counter()
    config = Config()
    print(config)
    for name, (sites_coords, unitcell_coord) in TILINGS.items():
        print(f"\n\n[{name}] ({len(sites_coords)} sites)")
        print()
        print_tiling(sites_coords, unitcell_coord)
        print()
        check_tilings(sites_coords, unitcell_coord, config)
    print(f"\nAll {len(TILINGS)} cases passed in {perf_counter() - start:.2f} s.")


if __name__ == "__main__":
    main()
