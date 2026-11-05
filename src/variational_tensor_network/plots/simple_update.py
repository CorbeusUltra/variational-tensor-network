from pathlib import Path

import matplotlib.pyplot as plt

from ._utils import finish_plot


def plot_simple_update(info: dict, title: str = "", show: bool = False, output_path: Path | None = None) -> None:
    tau_error_history = info["tau_error_history"]
    if not tau_error_history:
        raise ValueError("History must not be empty.")
    taus, errors = zip(*tau_error_history)
    sweeps = range(1, len(tau_error_history) + 1)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6), height_ratios=(3, 1))
    axes[0].semilogy(sweeps, errors, color = "blue")
    thresholds = [stage["tolerance"] for stage in info["stages"] for _ in range(stage["sweeps"])]
    axes[0].step(sweeps, thresholds, where="post", linestyle="--", color="red", label="Tolerance")
    axes[0].legend()
    axes[0].set_ylabel("Maximum change in bond energies")
    axes[1].step(sweeps, taus, where="post", color="purple")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="Sweep", ylabel=r"$\tau$")
    for ax in axes:
        ax.grid(alpha=0.3)
    finish_plot(fig, title=title, show=show, output_path=output_path)
