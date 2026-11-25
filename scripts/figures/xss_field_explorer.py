"""Explore the saved XSS field scan with Matplotlib. Reads scan.json files only."""

import argparse
import json
from itertools import combinations
from math import dist, hypot, isfinite
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.widgets import Button, Slider
from shastry_sutherland_bilayer_visualization.geometry import plane_position
from shastry_sutherland_bilayer_visualization.lattice import build_lattice

from variational_tensor_network._utils.path import find_project_root
from variational_tensor_network.tilings import TILINGS

root = find_project_root(__file__)
DEFAULT_DATA_DIR = root / "data" / "campaigns" / "D3-lowest-energy" / "ctmrg"
MODEL = "x_shastry_sutherland"
TILING = "abcd"
COUPLINGS = {"JD": 1.0, "J1": 0.35, "J2": 0.25}
FIELD_MIN, FIELD_MAX = 1.9, 2.675
DIMENSIONS = (3, 4, 5, 6)
ENERGY = "energy_per_spin"
MAGNETIZATION = "m_z_over_m_sat"
BOND_STYLES = {
    "JD": (2.5, "-", 3, r"$J_D$"),
    "J": (1.3, "-", 2, r"$J_1$"),
    "J2": (1.3, ":", 1, r"$J_2$"),
}
MOMENT_THRESHOLD = 1e-9
ARROW_EXPONENT = 0.1  # Matches the current xss_lattice.py rendering.


def load_scans(data_dir):
    """Select the same model, campaign, couplings and field window as the PDF."""
    scans = {dimension: {} for dimension in DIMENSIONS}
    for path in sorted(data_dir.glob("*/scan.json")):
        scan = json.loads(path.read_text(encoding="utf-8"))
        point = scan["point"]
        dimension = point["D"]
        params = point["parameters"]
        field = params.get("h")
        if (
            scan["status"] != "converged"
            or point["model"] != MODEL
            or point["tiling"] != TILING
            or dimension not in scans
            or {key: value for key, value in params.items() if key != "h"} != COUPLINGS
            or field is None
            or not FIELD_MIN <= field <= FIELD_MAX
            or (scan.get("sweep_direction") is not None and dimension != 3)
        ):
            continue
        if any(scan.get(key) is None or not isfinite(scan[key]) for key in (ENERGY, MAGNETIZATION)):
            continue
        if field in scans[dimension]:
            raise ValueError(f"Duplicate converged point D={dimension}, h={field:g}: {path}")
        scans[dimension][field] = scan
    scans = {dimension: dict(sorted(points.items())) for dimension, points in scans.items() if points}
    if not scans:
        raise ValueError(f"No converged points for the reference figure in {data_dir}")
    return scans


def endpoint_index(spin):
    return 0 if spin.endpoint == -1 else 1


def bond_key(bond, representative):
    source, target = bond.spin1.dimer.cell, bond.spin2.dimer.cell
    rep = representative(*source)
    if source == target:
        return (rep, "r"), (1, 2)
    direction = "r" if source[0] != target[0] else "d"
    pair = (1 + endpoint_index(bond.spin1), 3 + endpoint_index(bond.spin2))
    return (rep, direction), pair


def create_spin_panel(ax):
    """Create the 4x4 lattice once and return a callback that updates its artists."""
    _, representative = TILINGS[TILING]
    lattice = build_lattice(L=4)
    positions = {}
    for spin in lattice.spins:
        x, y, _ = plane_position(spin)
        positions[spin] = (x, -y)
    norm = Normalize(vmin=-0.75, vmax=0.25, clip=True)
    cmap = LinearSegmentedColormap.from_list("xss_corr", [(0.0, "#09665c"), (0.75, "#dededa"), (1.0, "#8c510a")], N=257)
    background = ax.get_facecolor()
    bond_artists = []
    for bond in lattice.bonds:
        width, style, zorder, _ = BOND_STYLES[bond.coupling]
        x, y = zip(positions[bond.spin1], positions[bond.spin2])
        line, = ax.plot(x, y, lw=width, ls=style, zorder=zorder, path_effects=[pe.Stroke(linewidth=width + 0.9, foreground=background), pe.Normal()])
        bond_artists.append((line, bond_key(bond, representative)))
    points = ax.scatter(*zip(*positions.values()), s=24, color="0.15", zorder=4)
    max_length = 0.4 * min(dist(a, b) for a, b in combinations(positions.values(), 2))
    arrows = []
    for spin, origin in positions.items():
        arrow = FancyArrowPatch(origin, origin, arrowstyle="-|>", mutation_scale=8, shrinkA=0, shrinkB=0, linewidth=1.1, color="black", zorder=5, visible=False, path_effects=[pe.Stroke(linewidth=2.8, foreground=background), pe.Normal()])
        ax.add_patch(arrow)
        arrows.append((arrow, origin, representative(*spin.dimer.cell), endpoint_index(spin)))
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.margins(0.12)
    ax.autoscale_view()
    ax.set_autoscale_on(False)
    handles = [Line2D([], [], color="0.25", lw=width, ls=style, label=label)
               for width, style, _, label in BOND_STYLES.values()]
    legend = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                       frameon=False, ncol=3)
    warning = ax.text(0.5, 0.5, "Measures unavailable for this point.",
                      transform=ax.transAxes, ha="center", va="center", visible=False)
    colorbar = ax.figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                                 shrink=0.78, fraction=0.05, pad=0.04)
    colorbar.ax.set_title(r"$C_{ij}$", pad=8)
    colorbar.set_ticks([-0.75, -0.5, -0.25, 0, 0.25],
                       labels=["−0.75", "−0.5", "−0.25", "0", "+0.25"])
    def update(scan):
        point = scan["point"]
        chi_label = rf"\qquad \chi={scan['selected_chi']}" if scan.get("selected_chi") is not None else ""
        ax.set_title(rf"$h={point['parameters']['h']:g}\qquad D={point['D']}{chi_label}$", pad=14)
        local = scan.get("local_magnetization")
        bonds = scan.get("spin_spin_correlations")
        has_local_data = bool(local and bonds)
        warning.set_visible(not has_local_data)
        legend.set_visible(has_local_data)
        points.set_visible(has_local_data)
        if not has_local_data:
            for line, _ in bond_artists:
                line.set_visible(False)
            for arrow, *_ in arrows:
                arrow.set_visible(False)
            return
        moments = {tuple(row["site"]): list(zip(row["S_x"], row["S_z"], strict=True)) for row in local}
        correlations = {(tuple(row["site"]), row["direction"]): { tuple(pair["spins"]): pair["value"] for pair in row["pairs"] } for row in bonds}
        for line, (key, pair) in bond_artists:
            line.set_color(cmap(norm(correlations[key][pair])))
            line.set_visible(True)
        for arrow, (x, y), rep, spin in arrows:
            sx, sz = moments[rep][spin]
            magnitude = hypot(sx, sz)
            arrow.set_visible(magnitude >= MOMENT_THRESHOLD)
            if magnitude >= MOMENT_THRESHOLD:
                length = max_length * min(magnitude / 0.5, 1.0) ** ARROW_EXPONENT
                factor = length / magnitude
                arrow.set_positions((x, y), (x + factor * sx, y + factor * sz))
    return update


def create_explorer(scans, initial_dimension=4, initial_field=2.025):
    fig = plt.figure(figsize=(13, 6.7))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.1, 1), left=0.085, right=0.94,
                          bottom=0.205, top=0.86, wspace=0.24, hspace=0.09)
    energy_ax = fig.add_subplot(grid[0, 0])
    magnetization_ax = fig.add_subplot(grid[1, 0], sharex=energy_ax)
    spin_ax = fig.add_subplot(grid[:, 1])
    fig.suptitle(r"$J_1/J_D=0.35\qquad J_2/J_D=0.25\qquad J_D=1$", fontsize=13, y=0.965)
    markers = ("o", "s", "^", "D", "+", "x", "v", "p", "h")
    colors = plt.get_cmap("Set1").colors
    plot_axes = (energy_ax, magnetization_ax)
    keys = (ENERGY, MAGNETIZATION)
    circles = []
    guides = []
    for ax, key, ylabel in zip(plot_axes, keys, (r"$E/N_{\mathrm{spins}}$", r"$m_z/m_{\mathrm{sat}}$"), strict=True):
        for dimension, points in scans.items():
            index = (dimension - 2) % len(markers)
            ax.plot(list(points), [scan[key] for scan in points.values()],
                    marker=markers[index], color=colors[index], markersize=4, label=rf"$D={dimension}$")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right" if key == ENERGY else "upper left", fontsize=9)
        guides.append(ax.axvline(FIELD_MIN, color="0.35", ls=":", lw=1, zorder=2.5))
        circle, = ax.plot([], [], marker="o", ls="none", ms=11, mfc="none", mec="black", mew=1.5, zorder=6)
        circles.append(circle)
    energy_ax.tick_params(labelbottom=False)
    magnetization_ax.set_xlabel(r"$h$")
    update_spins = create_spin_panel(spin_ax)

    active = initial_dimension if initial_dimension in scans else next(iter(scans))
    def selectable_fields(dimension):
        return [h for h, scan in scans[dimension].items() if scan.get("local_magnetization") and scan.get("spin_spin_correlations")]
    start = min(selectable_fields(active), key=lambda h: abs(h - initial_field))
    slider_ax = fig.add_axes((0.12, 0.073, 0.49, 0.033))
    slider = Slider(slider_ax, r"$h$", FIELD_MIN, FIELD_MAX, valinit=start, valstep=selectable_fields(active), valfmt="%1.4g", color="#27776c", initcolor="none")
    buttons = {}

    def select_field(field):
        scan = scans[active][field]
        for circle, guide, key in zip(circles, guides, keys, strict=True):
            circle.set_data([field], [scan[key]])
            guide.set_xdata([field, field])
        update_spins(scan)
        fig.canvas.draw_idle()

    def select_dimension(dimension):
        nonlocal active
        active = dimension
        fields = selectable_fields(active)
        slider.valstep = fields
        for button_dimension, button in buttons.items():
            button.color = "#c4e2dc" if button_dimension == active else "#eef1f2"
            button.ax.set_facecolor(button.color)
        slider.set_val(min(fields, key=lambda field: abs(field - slider.val)))

    for i, dimension in enumerate(scans):
        button_ax = fig.add_axes((0.73 + 0.057 * i, 0.063, 0.051, 0.052))
        button = Button(button_ax, rf"$D={dimension}$", color="#eef1f2", hovercolor="#a5d2c9")
        button.on_clicked(lambda _event, dimension=dimension: select_dimension(dimension))
        buttons[dimension] = button
    slider.on_changed(select_field)
    select_dimension(active)
    return fig, slider, buttons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="Dossier des résultats CTMRG contenant les scan.json.")
    parser.add_argument("--D", type=int, choices=DIMENSIONS, default=4, help="Valeur de D affichée au lancement.")
    parser.add_argument("--h", type=float, default=2.025, help="Champ initial. Le point calculé le plus proche est sélectionné.")
    args = parser.parse_args()
    explorer = create_explorer(load_scans(args.data_dir), args.D, args.h)
    plt.show()
    return explorer


if __name__ == "__main__":
    main()
