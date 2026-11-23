import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shastry_sutherland_bilayer_visualization.geometry import plane_position
from shastry_sutherland_bilayer_visualization.lattice import build_lattice

lattice = build_lattice(L=3)
positions = {}
for spin in lattice.spins:
    x, y, _ = plane_position(spin)
    positions[spin] = (x, -y)

styles = {
    "JD": {"color": "0.15", "linewidth": 2.5, "linestyle": "-", "zorder": 3},
    "J": {"color": "tab:blue", "linewidth": 1.5, "linestyle": "-", "zorder": 2},
    "J2": {"color": "0.55", "linewidth": 1.0, "linestyle": "--", "zorder": 1},
}
labels = {"JD": r"$J_D$", "J": r"$J_1$", "J2": r"$J_2$"}

fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
for bond in lattice.bonds:
    x1, y1 = positions[bond.spin1]
    x2, y2 = positions[bond.spin2]
    ax.plot((x1, x2), (y1, y2), **styles[bond.coupling])

xs, ys = zip(*positions.values())
ax.scatter(xs, ys, s=24, color="0.15", zorder=4)
ax.set_aspect("equal")
ax.set_axis_off()
ax.legend(handles=[Line2D([], [], label=labels[key], **style) for key, style in styles.items()], loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False)
plt.show()
