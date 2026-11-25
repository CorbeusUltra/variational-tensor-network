"""Plot the D=3 field continuation against the random-start reference."""

import json
from math import nan
from pathlib import Path

import matplotlib.pyplot as plt

from variational_tensor_network._utils.path import find_project_root
from variational_tensor_network.plots._utils import finish_plot


def read_summaries(directory: Path) -> dict[float, dict]:
    points = {}
    for path in directory.glob("h=*/summary.json"):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        points[summary["h"]] = summary
    return points


def read_reference(root: Path) -> dict[float, dict]:
    points = {}
    couplings = {"JD": 1.0, "J1": 0.35, "J2": 0.25}
    for path in (root / "data/campaigns/from-D3/ctmrg").glob("*/scan.json"):
        scan = json.loads(path.read_text(encoding="utf-8"))
        if scan.get("sweep_direction") is not None:
            continue
        point = scan["point"]
        if (point["D"], point["model"], point["tiling"]) != (3, "x_shastry_sutherland", "abcd"):
            continue
        h = point["parameters"]["h"]
        if point["parameters"] != {**couplings, "h": h}:
            continue
        if h in points:
            raise ValueError(f"Multiple random references at h={h:.3f}")
        points[h] = scan
    return points


def series(points: dict[float, dict], fields: list[float], key: str) -> list[float]:
    values = [points.get(h, {}).get(key) for h in fields]
    return [value if value is not None else nan for value in values]


def build_figure(root: Path):
    sweep = read_summaries(root / "outputs/diagnostics/nearby_h_D3_up")
    if not sweep:
        raise SystemExit("No continuation summary has been saved yet.")
    fields = sorted(sweep)
    nearby = {h: point for h, point in sweep.items() if point["chi_status"] == "converged"}
    if not nearby:
        raise SystemExit("No continuation point has a converged chi scan yet.")
    reference = read_reference(root)
    converged_reference = {h: point for h, point in reference.items() if point["status"] == "converged"}
    reference_fields = sorted(converged_reference)
    fig, (energy_ax, magnetization_ax) = plt.subplots(2, 1, figsize=(10, 6))
    magnetization_ax.sharex(energy_ax)
    energy_ax.tick_params(labelbottom=False)
    fig.suptitle(r"Field continuation at $D=3$")
    for ax, key, ylabel in ((energy_ax, "energy_per_spin", r"$E/N_{\rm spins}$"), (magnetization_ax, "m_z_over_m_sat", r"$m_z/m_{\rm sat}$")):
        ax.plot(fields, series(nearby, fields, key), "o-", color="red", ms=3, label="From previous field")
        ax.plot(reference_fields, series(converged_reference, reference_fields, key), "s-", color="blue", ms=3, label="Random start")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.20)
    energy_ax.legend(loc="best")
    return fig


def main() -> None:
    root = find_project_root(Path.cwd() / "pyproject.toml")
    fig = build_figure(root)
    finish_plot(fig, show=True, output_path=root / "outputs/diagnostics/nearby_h_D3_up/comparison_full")


if __name__ == "__main__":
    main()
