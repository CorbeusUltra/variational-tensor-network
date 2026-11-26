import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from shastry_sutherland_bilayer_visualization.geometry import plane_position
from shastry_sutherland_bilayer_visualization.lattice import Dimer, spins_of

from variational_tensor_network._utils.path import find_project_root
from variational_tensor_network.plots._utils import finish_plot
from variational_tensor_network.tilings import TILINGS

type ScansByField = dict[float, dict[int, dict[str, Any]]]

ROOT = find_project_root(__file__)

CAMPAIGN = "D3-lowest-energy"
DATA_DIR = ROOT / "data" / "campaigns" / CAMPAIGN / "ctmrg"
FIGURES_DIR = ROOT / "outputs" / "figures"


SCAN_HASHES = {
    2.165: {
        3: "610565bb91b7933c598ae70c4e286b41dd3ef8c6de68f80cb25a306558309ffd",
        4: "804b75586324421435f1650e93d2cbb805e6d075bc02a944754bf61a66624c8d",
        5: "174f6a0628e9d9d9e8e0e01b519ef9781ceb1ffdad8d6c1bb9452cd43ea4aa30",
        6: "7034c67abd107c32e9ca81a2a9acdd2177ada1efcf48a01dfd859d7290ad850e",
        7: "b0f84dd2b3dec7565ef22a7bbbd6aae0e5b8e54beac342732e5abc7992f0a666",
    },
    2.170: {
        3: "0d5e7b589837f2c4effeee51c6d60e7940cf428796f3859ea083377ee07f78c9",
        4: "7deb078e9a155d31778dd4ee192190f7b7cb79c304ccf9e555ffe7ca9aa690e1",
        5: "f004631069b162f00a78286796a1e877a3455f34030694c7a3ddba46fb0025e0",
        6: "8c490f73ced8606f215d99cddc8e6a73ed6d66b04ec2438f7be00fb769bfab74",
        7: "bc56b71aef3aa938280dd21670e2fca7d91076e44db53acb08267877c5f60ee5",
    },
}


def load_scans(scan_hashes: dict[float, dict[int, str]]) -> ScansByField:
    scans: ScansByField = {}
    for h, hashes_by_d in scan_hashes.items():
        scans[h] = {}
        for d, scan_hash in hashes_by_d.items():
            path = DATA_DIR / scan_hash / "scan.json"
            scan = json.loads(path.read_text(encoding="utf-8"))
            point = scan["point"]
            if scan["status"] != "converged" or (point["parameters"]["h"], point["D"]) != (h, d):
                raise ValueError(f"Unexpected scan for h={h:.3f}, D={d}: {path}")
            scans[h][d] = scan
    return scans


def plot_energy_vs_inverse_D(h: float, scans_by_d: dict[int, dict[str, Any]], *, show: bool = False, output_path: Path | None = None) -> None:
    dimensions = sorted(scans_by_d, reverse=True)
    inverse_d = [1 / d for d in dimensions]
    energies = [scans_by_d[d]["energy_per_spin"] for d in dimensions]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(inverse_d, energies, marker="o", color="blue")
    ax.set(title=rf"$h={h:.3f}$", xlabel=r"$1/D$", ylabel=r"$E/N_{\mathrm{spins}}$")
    ax.set_xticks(inverse_d, [rf"$1/{d}$" for d in dimensions])
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    ax.grid(alpha=0.3)
    finish_plot(fig, show=show, output_path=output_path)


def plot_local_magnetization_vs_inverse_D(h: float, scans_by_d: dict[int, dict[str, Any]], *, show: bool = False, output_path: Path | None = None) -> None:
    dimer_sites, _ = TILINGS["abcd"]
    spin_styles = (("o", "-"), ("x", "--"))
    dimensions = sorted(scans_by_d, reverse=True)
    inverse_d = [1 / d for d in dimensions]
    moments_by_d = {d: {tuple(row["site"]): row for row in scan["local_magnetization"]} for d, scan in scans_by_d.items()}
    fig = plt.figure(figsize=(7.5, 6))
    grid = fig.add_gridspec(2, 2, width_ratios=(4, 1.25))
    ax_z = fig.add_subplot(grid[0, 0])
    ax_x = fig.add_subplot(grid[1, 0], sharex=ax_z)
    cell_ax = fig.add_subplot(grid[:, 1])
    components = ((ax_z, "S_z", r"$\langle S_i^z\rangle$"), (ax_x, "S_x", r"$\langle S_i^x\rangle$"))
    for dimer_index, site in enumerate(dimer_sites):
        color = f"C{dimer_index}"
        positions = [plane_position(spin)[:2] for spin in spins_of(Dimer(site))]
        cell_ax.plot(*zip(*positions), color=color, alpha=0.5)
        for spin_index, (marker, linestyle) in enumerate(spin_styles):
            for ax, key, _ in components:
                moments = [moments_by_d[d][site][key][spin_index] for d in dimensions]
                ax.plot(inverse_d, moments, color=color, marker=marker, linestyle=linestyle, markerfacecolor="none")
            cell_ax.plot(*positions[spin_index], color=color, marker=marker, linestyle="none", markerfacecolor="white", markersize=7)
    for ax, _, ylabel in components:
        ax.set_ylabel(ylabel)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.grid(alpha=0.3)
    ax_z.tick_params(labelbottom=False)
    ax_x.set_xlabel(r"$1/D$")
    ax_x.set_xticks(inverse_d, [rf"$1/{d}$" for d in dimensions])
    ax_x.axhline(0, color="0.5", linewidth=0.8, zorder=0)
    cell_ax.set_aspect("equal")
    cell_ax.margins(0.2)
    cell_ax.invert_yaxis()
    cell_ax.set_axis_off()
    cell_ax.set_title("Unit cell", fontsize=9)
    finish_plot(fig, title=rf"$h={h:.3f}$", show=show, output_path=output_path)


def main() -> None:
    scans = load_scans(SCAN_HASHES)
    for h, scans_by_d in sorted(scans.items()):
        output_path = FIGURES_DIR / "D_convergence" / "energy" / f"xss_abcd_{CAMPAIGN}_h={h:.3f}_energy_vs_inverse_D"
        plot_energy_vs_inverse_D(h, scans_by_d, show=True, output_path=output_path)
        output_path = FIGURES_DIR / "D_convergence" / "local_magnetization" / f"xss_abcd_{CAMPAIGN}_h={h:.3f}_local_magnetization_vs_inverse_D"
        plot_local_magnetization_vs_inverse_D(h, scans_by_d, show=True, output_path=output_path)


if __name__ == "__main__":
    main()
