from pathlib import Path

import matplotlib.pyplot as plt

from ._utils import finish_plot


def plot_simple_update(tau_error_history: list[tuple[float, float]], title: str = "", show: bool = False, output_path: Path | None = None) -> None:
    if not tau_error_history:
        raise ValueError("History must not be empty.")
    taus, errors = zip(*tau_error_history)
    sweeps = range(1, len(tau_error_history) + 1)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6), height_ratios=(3, 1))
    axes[0].semilogy(sweeps, errors)
    axes[0].set_ylabel("Maximum change in bond energies")
    axes[1].step(sweeps, taus, where="post")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="Sweep", ylabel=r"$\tau$")
    for ax in axes:
        ax.grid(alpha=0.3)
    finish_plot(fig, title=title, show=show, output_path=output_path)
