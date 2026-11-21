from itertools import cycle, pairwise
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


def plot_magnetization_sweep(curves, *, show=False, output_path=None):
    fig, ax = plt.subplots()
    for direction, label, color, style in (("up", r"$h$ increasing", "C0", "-"), ("down", r"$h$ decreasing", "C1", "-")):
        points = curves[direction]
        h, m = zip(*points)
        ax.plot(h, m, marker="o", markersize=2, color=color, linestyle=style, label=label)
        segments = [(a, b) for a, b in pairwise(points)]
        if segments:
            a, b = segments[len(segments) // (2 if direction == "up" else 3)]
            ax.annotate("", xy=b, xytext=a,
                        arrowprops={"arrowstyle": "->", "color": color})
    ax.set(title=r"$D=2$", xlabel=r"$h$", ylabel=r"$m_z/m_{\mathrm{sat}}$")
    ax.grid(alpha=0.3)
    ax.legend()
    finish_plot(fig, show=show, output_path=output_path)


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
