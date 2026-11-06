"""Print site representatives and export tiling diagrams for visual inspection."""

from time import perf_counter

from variational_tensor_network.ctmrg_moves import UnitCellCoord
from variational_tensor_network.plots import plot_output_path, plot_tiling
from variational_tensor_network.tilings import TILINGS, Site


def print_tiling(sites_coord: tuple[Site, ...], unitcell_coord: UnitCellCoord, width: int = 5, height: int = 5) -> None:
    labels = {site: i for i, site in enumerate(sites_coord)}
    for y in range(height):
        print(" ".join(f"[{labels[unitcell_coord(x, y)]}]" for x in range(width)))



def main() -> None:
    start = perf_counter()
    for name, (sites_coords, unitcell_coord) in TILINGS.items():
        print(f"\n\n[{name}] ({len(sites_coords)} sites)")
        print()
        print_tiling(sites_coords, unitcell_coord)
        print()
        output_path = plot_output_path(__file__, f"{name}.pdf")
        plot_tiling(sites_coords, unitcell_coord, size=6, title=name, show=True, output_path=output_path)
    print(f"\nExported {len(TILINGS)} diagrams in {perf_counter() - start:.2f} s.\n")


if __name__ == "__main__":
    main()
