from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

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
    axes[0].set_ylabel("Change in energy")
    axes[1].step(sweeps, taus, where="post", color="purple")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="Sweep", ylabel=r"$\tau$")
    for ax in axes:
        ax.grid(alpha=0.3)
    finish_plot(fig, title=title, show=show, output_path=output_path)


def plot_simple_update_by_stage(info: dict, title: str = "", show: bool = False, output_path: Path | None = None) -> None:
    """Plot each executed tau stage on its own axes with local sweep indices."""
    stages = info["stages"]
    history = info["tau_error_history"]
    if not stages or not history:
        raise ValueError("Stage histories must not be empty.")
    if sum(stage["sweeps"] for stage in stages) != len(history):
        raise ValueError("Stage sweep counts must match the error history length.")
    fig, axes = plt.subplots(len(stages), 1, figsize=(10, 2.8 * len(stages)), squeeze=False)
    fig.supxlabel("Sweep within stage")
    formatter = ScalarFormatter(useMathText=True)
    start = 0
    for index, (ax, stage) in enumerate(zip(axes[:, 0], stages), start=1):
        stop = start + stage["sweeps"]
        errors = [error for _, error in history[start:stop]]
        sweeps = range(1, len(errors) + 1)
        ax.semilogy(sweeps, errors, color="blue", label="Change in energy")
        ax.axhline(stage["tolerance"], color="red", linestyle="--", label="Tolerance")
        tau_text = formatter.format_data(stage["tau"])
        tol_text = formatter.format_data(stage["tolerance"]) if stage["tolerance"] != 0 else "0"
        ax.set_title(rf"Stage {index} | {stage['status']} | $\tau={tau_text}$, $\mathrm{{tol}}={tol_text}$")
        ax.grid(alpha=0.3)
        ax.legend()
        start = stop
    finish_plot(fig, title=title, show=show, output_path=output_path)
