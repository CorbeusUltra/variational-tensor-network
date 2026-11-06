"""Check periodic bond-weight sharing between neighboring sites."""

from time import perf_counter

import torch

from variational_tensor_network.initialize import initial_a_sites, initial_bond_weights
from variational_tensor_network.objects import BondWeights, Config
from variational_tensor_network.peps import get_site_weights
from variational_tensor_network.tilings import TILINGS, Site, UnitCellCoord


def check_weights(sites_coords: tuple[Site, ...], unitcell_coord: UnitCellCoord, bond_weights: BondWeights) -> None:
    """Check the number of bond weights and their sharing between neighbors."""
    assert len(bond_weights) == 2 * len(sites_coords), "Expected two bond weights per PEPS site."
    for site in sites_coords:
        x, y = site
        weights = get_site_weights(site, bond_weights, unitcell_coord)
        right = get_site_weights((x + 1, y), bond_weights, unitcell_coord)
        down = get_site_weights((x, y + 1), bond_weights, unitcell_coord)

        assert weights.r is right.l, f"Right bond is not shared at {site}."
        assert weights.d is down.u, f"Down bond is not shared at {site}."



def main() -> None:
    start = perf_counter()
    torch.manual_seed(0)
    config = Config()
    for name, (sites_coords, unitcell_coord) in TILINGS.items():
        a_sites = initial_a_sites(config, sites_coords)
        bond_weights = initial_bond_weights(a_sites)
        check_weights(sites_coords, unitcell_coord, bond_weights)
        labels = {id(weight): f"𝜆_{i}" for i, weight in enumerate(bond_weights.values())}
        print(f"\n\n{name}: {len(a_sites)} sites, {len(bond_weights)} bond weights.\n")
        for site in sites_coords:
            x, y = site
            weights = get_site_weights(site, bond_weights, unitcell_coord)
            right = get_site_weights((x + 1, y), bond_weights, unitcell_coord)
            down = get_site_weights((x, y + 1), bond_weights, unitcell_coord)
            shared_r = weights.r is right.l
            shared_d = weights.d is down.u
            names = ", ".join(f"{direction}={labels[id(getattr(weights, direction))]}" for direction in ("r", "u", "l", "d"))
            print(f"  {site}: {names}; shared_r={shared_r}, shared_d={shared_d}")
        print()
    print(f"\nAll {len(TILINGS)} weight-sharing checks passed in {perf_counter() - start:.2g} s.\n")


if __name__ == "__main__":
    main()
