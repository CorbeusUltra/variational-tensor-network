import json
from itertools import combinations
from math import dist, hypot
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from shastry_sutherland_bilayer_visualization.geometry import plane_position
from shastry_sutherland_bilayer_visualization.lattice import build_lattice

from variational_tensor_network._utils.path import find_project_root
from variational_tensor_network.tilings import TILINGS

root = find_project_root(__file__)

# m=1/2 plateau : "23a1d96655d6087d30e2b9cbe01d6ecb1bb4b5ee3c486ccf3f6a0142c6835b14/scan.json"
# canted state  : "7deb078e9a155d31778dd4ee192190f7b7cb79c304ccf9e555ffe7ca9aa690e1/scan.json"
# m=1 plateau   : "ae23d774600bd85b4dc9b84454f0e7c2f79b4e64cf34a9c1037a4b9e065095e9/scan.json"

scan_path = root / "data/campaigns/D3-lowest-energy/ctmrg" / "23a1d96655d6087d30e2b9cbe01d6ecb1bb4b5ee3c486ccf3f6a0142c6835b14/scan.json"

scan = json.loads(Path(scan_path).read_text(encoding="utf-8"))

_, unitcell_coord = TILINGS[scan["point"]["tiling"]]

moments = {tuple(row["site"]): list(zip(row["S_x"], row["S_z"], strict=True)) for row in scan["local_magnetization"]}
correlations = {(tuple(row["site"]), row["direction"]): {tuple(pair["spins"]): pair["value"] for pair in row["pairs"]} for row in scan["spin_spin_correlations"]}

lattice = build_lattice(L=3)
positions = {}
for spin in lattice.spins:
    x, y, _ = plane_position(spin)
    positions[spin] = (x, -y)



def endpoint_index(spin):
    return 0 if spin.endpoint == -1 else 1


def correlation_for_bond(bond):
    source = bond.spin1.dimer.cell
    target = bond.spin2.dimer.cell
    rep = unitcell_coord(*source)

    if source == target:
        return correlations[(rep, "r")][(1, 2)]

    direction = "r" if source[0] != target[0] else "d"
    i = 1 + endpoint_index(bond.spin1)
    j = 3 + endpoint_index(bond.spin2)
    return correlations[(rep, direction)][(i, j)]


styles = {
    "JD": (2.5, "-", 3),
    "J":  (1.3, "-", 2),
    "J2": (1.3, ":", 1),
}

norm = Normalize(vmin=-0.75, vmax=0.25)
cmap = LinearSegmentedColormap.from_list("xss_corr",
    [
        (0.00, "#09665c"),  # Cij = -0.75
        (0.75, "#dededa"),  # Cij =  0
        (1.00, "#8c510a"),  # Cij = +0.25
    ], N=257)


def draw_bond(ax, bond, color):
    x1, y1 = positions[bond.spin1]
    x2, y2 = positions[bond.spin2]
    width, linestyle, zorder = styles[bond.coupling]
    ax.plot((x1, x2), (y1, y2), color=color, linewidth=width, linestyle=linestyle, zorder=zorder, path_effects=[ pe.Stroke(linewidth=width + 0.9, foreground=ax.get_facecolor()), pe.Normal(), ])


fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")
for bond in lattice.bonds:
    value = correlation_for_bond(bond)
    display_value = max(-0.75, min(0.25, value))
    draw_bond(ax, bond, cmap(norm(display_value)))

cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.8)
cbar.ax.set_title(r"$C_{ij}$", pad=8)
cbar.set_ticks([-0.75, -0.5, -0.25, 0, 0.25], labels=["−0.75", "−0.5", "−0.25", "0", "+0.25"])

d_min = min(dist(a, b) for a, b in combinations(positions.values(), 2))
scale = 0.4 * d_min / 0.5

max_length = 0.4 * d_min
gamma = 0.1

for spin, (x, y) in positions.items():
    rep = unitcell_coord(*spin.dimer.cell)
    sx, sz = moments[rep][endpoint_index(spin)]
    magnitude = hypot(sx, sz)
    if magnitude < 1e-9:
        continue
    length = max_length * min(magnitude / 0.5, 1.0) ** gamma
    factor = length / magnitude
    ax.add_patch(FancyArrowPatch((x, y), (x + factor * sx, y + factor * sz), arrowstyle="-|>", mutation_scale=8, shrinkA=0, shrinkB=0, linewidth=1.1, color="black", zorder=5, path_effects=[pe.Stroke(linewidth=2.8, foreground=ax.get_facecolor()), pe.Normal()]))


xs, ys = zip(*positions.values())
ax.scatter(xs, ys, s=24, color="0.15", zorder=4)
ax.set_aspect("equal")
ax.set_axis_off()
ax.margins(0.12)

labels = {"JD": r"$J_D$", "J": r"$J_1$", "J2": r"$J_2$"}

handles = [Line2D([], [], color="0.25", linewidth=width, linestyle=linestyle, label=labels[family]) for family, (width, linestyle, _) in styles.items()]
ax.legend(handles=handles, loc="upper center",
          bbox_to_anchor=(0.5, -0.03), ncol=3, frameon=False)

params = scan["point"]["parameters"]
jd = params["JD"]

fig.suptitle(rf"$J_1/J_D = {params['J1'] / jd:.4g}" rf"\qquad J_2/J_D = {params['J2'] / jd:.4g}" rf"\qquad h = {params['h']:.4g}$", fontsize=12)

plt.savefig("/Users/lucas/Downloads/half_plateau.pdf")
plt.show()
