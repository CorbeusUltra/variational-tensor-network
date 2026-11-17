from pathlib import Path

import matplotlib.pyplot as plt

from ._utils import finish_plot


def plot_magnetization(curves_by_D: dict[int, list[tuple[float, float]]], *, title: str = "", show: bool = False, output_path: Path | None = None) -> None:
    """Plot one magnetization curve per D from sorted (h, m_z / m_sat) pairs."""
    fig, ax = plt.subplots()
    for D, points in curves_by_D.items():
        fields, magnetizations = zip(*points)
        ax.plot(fields, magnetizations, "o-", markersize=4, label=f"$D={D}$")
    ax.set(xlabel=r"$h$", ylabel=r"$m_z/m_{\mathrm{sat}}$")
    ax.grid(alpha=0.3)
    ax.legend()
    finish_plot(fig, title=title, show=show, output_path=output_path)
