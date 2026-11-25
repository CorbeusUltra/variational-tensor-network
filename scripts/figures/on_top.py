import json
import pickle
from math import isfinite
from pathlib import Path

import matplotlib.pyplot as plt

with Path("outputs/figures/energy/pkl/D2_energy_continuation.pkl").open("rb") as stream:
    fig = pickle.load(stream)

points = {}
for path in Path("data/campaigns/from-D3/ctmrg").glob("*/scan.json"):
    scan = json.loads(path.read_text())
    point = scan["point"]
    couplings = point["parameters"].copy()
    h = couplings.pop("h")

    if (point["D"], point["model"], point["tiling"]) != (3, "x_shastry_sutherland", "abcd"):
        continue
    if couplings != {"JD": 1.0, "J1": 0.35, "J2": 0.25} or scan.get("sweep_direction") is not None:
        continue

    energy = scan.get("energy_per_spin")
    if scan["status"] == "converged" and energy is not None and isfinite(energy):
        if h in points:
            raise ValueError(f"Several converged runs at h={h:g}")
        points[h] = energy

h, energy = zip(*sorted(points.items()))
ax = fig.axes[0]
ax.plot(h, energy, color="#00836a", marker="D", markersize=2, zorder=3, label=r"$D=3$")
ax.set_title(r"$D=2$ and $D=3$")
ax.legend()

print(f"{len(points)} D=3 points, from h={h[0]:g} to h={h[-1]:g}")
fig.tight_layout()
fig.savefig("outputs/figures/energy/D2_D3_energy.pdf")
plt.show()
