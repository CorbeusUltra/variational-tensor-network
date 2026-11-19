from itertools import cycle
from pathlib import Path

import matplotlib.pyplot as plt

from ._utils import finish_plot

markers = cycle(["^", "s", "p", "h", "D", "v", "o", "+", "x", ""])
colors = cycle(plt.get_cmap("Set1").colors) # okabe_ito


def plot_magnetization(curves_by_D: dict[int, list[tuple[float, float]]], *, title: str = "", show: bool = False, output_path: Path | None = None) -> None:
    """Plot one magnetization curve per D from sorted (h, m_z / m_sat) pairs."""
    fig, ax = plt.subplots()
    for D, points in curves_by_D.items():
        fields, magnetizations = zip(*points)
        ax.plot(fields, magnetizations, linestyle="-", marker=next(markers), color=next(colors), markersize=4, label=f"$D={D}$")
    ax.set(xlabel=r"$h$", ylabel=r"$m_z/m_{\mathrm{sat}}$")
    ax.grid(alpha=0.3)
    ax.legend()
    finish_plot(fig, title=title, show=show, output_path=output_path)


def plot_energy(curves_by_D: dict[int, list[tuple[float, float]]], *, title: str = "", show: bool = False, output_path: Path | None = None) -> None:
    """Plot one magnetization curve per D from sorted (h, E) pairs."""
    fig, ax = plt.subplots()
    for D, points in curves_by_D.items():
        fields, energies = zip(*points)
        ax.plot(fields, energies, linestyle="-", marker=next(markers), color=next(colors), markersize=4, label=f"$D={D}$")
    ax.set(xlabel=r"$h$", ylabel=r"$E/N_{\mathrm{spins}}$")
    ax.grid(alpha=0.3)
    ax.legend()
    finish_plot(fig, title=title, show=show, output_path=output_path)
