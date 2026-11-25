from itertools import pairwise
from math import isfinite

import matplotlib.pyplot as plt

from ._utils import finish_plot

D_MARKERS = ("o", "s", "^", "D", "+", "x", "v", "p", "h")
D_COLORS = plt.get_cmap("Set1").colors
D_STYLES = tuple(zip(D_MARKERS, D_COLORS, strict=True))




def _draw_observable(ax, curves_by_D, *, ylabel):
    for D, points in curves_by_D.items():
        marker, color = D_STYLES[(D - 2) % len(D_STYLES)]
        fields, values = zip(*points)
        ax.plot(fields, values, linestyle="-", marker=marker, color=color, markersize=4, label=f"$D={D}$")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()


def plot_observable(curves_by_D, *, ylabel, title="", show=False, output_path=None):
    """Plot one observable O curve per D from sorted (h, O) pairs."""
    fig, ax = plt.subplots()
    _draw_observable(ax, curves_by_D, ylabel=ylabel)
    ax.set_xlabel(r"$h$")
    finish_plot(fig, title=title, show=show, output_path=output_path)


def plot_energy_and_magnetization(energy_curves, magnetization_curves, *, continuation=False, title="", show=False, output_path=None):
    fig, (energy_ax, magnetization_ax) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    draw_curves = _draw_field_sweep if continuation else _draw_observable
    draw_curves(energy_ax, energy_curves, ylabel=r"$E/N_{\mathrm{spins}}$")
    draw_curves(magnetization_ax, magnetization_curves, ylabel=r"$m_z/m_{\mathrm{sat}}$")
    magnetization_ax.set_xlabel(r"$h$")
    finish_plot(fig, title=title, show=show, output_path=output_path)


def _draw_field_sweep(ax, curves, *, ylabel):
    for direction, label, color, style in (("up", r"$h$ increasing", "C0", "-"), ("down", r"$h$ decreasing", "C1", "-")):
        points = curves[direction]
        if not any(isfinite(y) for _, y in points):
            continue
        h, m = zip(*points)
        ax.plot(h, m, marker="o", markersize=2, color=color, linestyle=style, label=label)
        segments = [(a, b) for a, b in pairwise(points) if isfinite(a[1]) and isfinite(b[1])]
        if segments:
            a, b = segments[len(segments) // (2 if direction == "up" else 3)]
            ax.annotate("", xy=b, xytext=a, arrowprops={"arrowstyle": "->", "color": color})
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()


def plot_field_sweep(curves, *, ylabel: str = "", show=False, output_path=None):
    fig, ax = plt.subplots()
    _draw_field_sweep(ax, curves, ylabel=ylabel)
    ax.set(title=r"$D=2$", xlabel=r"$h$", ylabel=ylabel)
    finish_plot(fig, show=show, output_path=output_path)
