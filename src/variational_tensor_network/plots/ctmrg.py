from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Circle
from torch import Tensor

from ..tilings import Site, UnitCellCoord
from ._utils import finish_plot


def plot_ctmrg_convergence(errors: list[float], thresholds: list[float] | None = None, title: str = "", show: bool = False, output_path: Path | None = None) -> None:
    cycles = range(2, len(errors) + 2)
    fig, ax = plt.subplots()
    ax.semilogy(cycles, errors, "x-", color="blue", label=r"$|\text{energy}_\mathrm{new} - \text{energy}_\mathrm{old}|$")
    if thresholds is not None:
        ax.semilogy(cycles, thresholds, "--", color="red", label="threshold")
    ax.set(xlabel="number of cycle", ylabel=r"change in energy")
    ax.legend()
    ax.grid()
    finish_plot(fig, title=title, show=show, output_path=output_path)


def plot_ctmrg_convergence_by_site(errors: list[Tensor], thresholds: list[Tensor], sites_coords: tuple[Site, ...], unitcell_coord: UnitCellCoord, title: str = "", show: bool = False, output_path: Path | None = None) -> None:
    errors_array = torch.stack(errors).numpy()
    thresholds_array = torch.stack(thresholds).numpy()
    cycles = range(2, len(errors) + 2)
    fig, ax = plt.subplots()
    for i, site in enumerate(sites_coords):
        ax.semilogy(cycles, errors_array[:, i], color=f"C{i}")
    ax.semilogy(cycles, thresholds_array[:, 0], "--", color="black", label="Threshold")
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


def plot_energy_vs_chi(chi_values: list[int], energies: list[float], D: int, title: str = "", show: bool = False, output_path: Path | None = None) -> None:
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
