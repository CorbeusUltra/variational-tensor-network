from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Circle
from torch import Tensor

from ..tilings import TILINGS, Site, UnitCellCoord
from ._utils import finish_plot


def plot_ctmrg_convergence_max(errors: list[float], title: str = "", show: bool = False, output_path: Path | None = None, *, tolerance: float | None = None) -> None:
    cycles = range(2, len(errors) + 2)
    fig, ax = plt.subplots()
    ax.semilogy(cycles, errors, "-", color="blue")
    if tolerance is not None:
        ax.axhline(tolerance, linestyle="--", color="black", label="Tolerance")
    ax.set(xlabel="CTMRG cycle", ylabel="Maximum bond-energy change")
    ax.legend()
    ax.grid()
    finish_plot(fig, title=title, show=show, output_path=output_path)


def plot_ctmrg_convergence_by_site(errors: list[Tensor], tolerance: float, sites_coords: tuple[Site, ...], unitcell_coord: UnitCellCoord, title: str = "", show: bool = False, output_path: Path | None = None) -> None:
    errors_array = torch.stack(errors).numpy()
    cycles = range(2, len(errors) + 2)
    fig, ax = plt.subplots()
    for i, site in enumerate(sites_coords):
        ax.semilogy(cycles, errors_array[:, i], color=f"C{i}")
    ax.axhline(tolerance, linestyle="--", color="black", label="Tolerance")
    ax.set(xlabel="number of cycle", ylabel=r"change in energy")
    ax.legend(loc="upper center")
    ax.grid()
    inset = ax.inset_axes([0.68, 0.55, 0.28, 0.38], facecolor="white")
    for y in range(6):
        for x in range(6):
            i = sites_coords.index(unitcell_coord(x, y))
            inset.add_patch(Circle((x, y), radius=0.3, facecolor="white", edgecolor=f"C{i}"))
            inset.text(x, y+.05, str(i), ha="center", va="center", fontsize=7, color=f"C{i}")
    inset.set_xlim(-0.5, 5.5)
    inset.set_ylim(5.5, -0.5)
    inset.set_aspect("equal")
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("0.5")
        spine.set_linewidth(0.8)
    inset.set_title("Tiling", fontsize=9)
    finish_plot(fig, title=title, show=show, output_path=output_path)


def plot_ctmrg_convergence(result: dict[str, Any], *, by_site: bool = False, show: bool = False, output_path: Path | None = None) -> None:
    info = result["info_ctmrg"]
    errors = info["error_history"]
    if not errors:
        print("No CTMRG cycle history.")
        return
    title = rf"CTMRG | $\chi={info['chi']}$ | {info['status']}"
    if by_site:
        tiling = result["metadata"]["su_metadata"]["tiling"]
        _, unitcell_coord = TILINGS[tiling]
        sites_coords = tuple(site for site, direction in info["bond_rdms"] if direction == "r")
        if output_path is not None:
            output_path = output_path.with_name(f"{output_path.name}_by_site")
        plot_ctmrg_convergence_by_site(errors, info["tolerance"], sites_coords, unitcell_coord, title=title, show=show, output_path=output_path)
        return
    plot_ctmrg_convergence_max([error.max().item() for error in errors], title=title, tolerance=info["tolerance"], show=show, output_path=output_path)


def plot_energy_vs_chi(chi_values: list[int], energies: list[float], title: str = "", show: bool = False, output_path: Path | None = None) -> None:
    fig, ax = plt.subplots()
    ax.plot(chi_values, energies, "s-b")
    ax.set(xlabel=r"$\chi$", ylabel="Energy per spin")
    ax.set_xticks(chi_values)
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.set_axisbelow(True)
    ax.grid(alpha=0.3)
    finish_plot(fig, title=title, show=show, output_path=output_path)


# Legacy code

# def plot_energy_by_tiling_vs_chi(energies_by_tiling: dict[str, tuple[list[int], list[float]]], title: str = "", show: bool = False, output_path: Path | None = None) -> None:
#     from itertool import cycle
#     markers = cycle(["^", "s", "p", "h", "D", "v", "o", "+", "x", ""])
#     fig, ax = plt.subplots()
#     for tiling_name, (chi_values, energies) in energies_by_tiling.items():
#         ax.plot(chi_values, energies, "-", marker=next(markers), label = f"{tiling_name}")
#     ax.set(xlabel=r"$\chi$", ylabel="Energy per PEPS site")
#     all_chi = sorted({chi for chi_values, _ in energies_by_tiling.values() for chi in chi_values})
#     ax.set_xticks(all_chi)
#     ax.legend()
#     ax.ticklabel_format(axis="y", style="plain", useOffset=False)
#     ax.set_axisbelow(True)
#     ax.grid(alpha=0.3)
#     finish_plot(fig, title=title, show=show, output_path=output_path)

# def plot_energy_by_D_vs_chi(energies_by_D: dict[int, tuple[list[int], list[float]]], title: str = "", show: bool = False, output_path: Path | None = None) -> None:
#     markers = cycle(["^", "s", "p", "h", "D", "v", "o", "+", "x", ""])
#     fig, ax = plt.subplots()
#     for D, (chi_values, energies) in energies_by_D.items():
#         ax.plot(chi_values, energies, "-", marker=next(markers), label = fr"$D = {D}$")
#     ax.set(xlabel=r"$\chi$", ylabel="Energy per PEPS site")
#     all_chi = sorted({chi for chi_values, _ in energies_by_D.values() for chi in chi_values})
#     ax.set_xticks(all_chi)
#     ax.legend()
#     ax.ticklabel_format(axis="y", style="plain", useOffset=False)
#     ax.set_axisbelow(True)
#     ax.grid(alpha=0.3)
#     finish_plot(fig, title=title, show=show, output_path=output_path)

# def plot_uniform_ctmrg_comparison(errors: dict[str, list[float]], thresholds: list[float], title: str = "", show: bool = False, output_path: Path | None = None) -> None:
#     cycles = range(2, len(thresholds) + 2)
#     fig, ax = plt.subplots()
#     ax.semilogy(cycles, errors["Single-site"], "k-", label="Single-site")
#     for name, style in (("checkerboard", "--"), ("rectangle_2x3", ":")):
#         ax.semilogy(cycles, errors[name], style, label=name)
#     ax.semilogy(cycles, thresholds, "r--", label="Threshold")
#     ax.set(xlabel="number of cycle", ylabel=r"change in energy")
#     ax.legend()
#     ax.grid()
#     finish_plot(fig, title=title, show=show, output_path=output_path)
