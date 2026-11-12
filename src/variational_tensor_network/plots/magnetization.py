from pathlib import Path

import matplotlib.pyplot as plt

from ._utils import finish_plot


def plot_magnetization(h: list[float], m_z_over_m_sat: list[float] | None = None, title: str = "", show: bool = False, output_path: Path | None = None) -> None:
    fig, ax = plt.subplots()
    ax.plot(h, m_z_over_m_sat, "s-", color="red")
    ax.set(xlabel="$h$", ylabel=r"$\frac{m_z}{m_\text{sat}}$")
    ax.grid()
    finish_plot(fig, title=title, show=show, output_path=output_path)
