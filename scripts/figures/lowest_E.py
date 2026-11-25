"""Plot the selected D=3 energies and magnetizations."""

import json

import matplotlib.pyplot as plt

from variational_tensor_network._utils.path import find_project_root
from variational_tensor_network.plots._utils import finish_plot


def main() -> None:
    root = find_project_root(__file__)
    data_dir = root / "data/campaigns/D3-lowest-energy/ctmrg"
    points = []
    for path in data_dir.glob("*/scan.json"):
        scan = json.loads(path.read_text(encoding="utf-8"))
        if scan["status"] == "converged" and scan["point"]["D"] == 3:
            h = scan["point"]["parameters"]["h"]
            points.append((h, scan["energy_per_spin"], scan["m_z_over_m_sat"]))
    if not points:
        raise SystemExit(f"No converged scans found in {data_dir}.")
    fields, energies, magnetizations = zip(*sorted(points))
    fig, (energy_ax, mag_ax) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    energy_ax.plot(fields, energies, "bo-", ms=3)
    energy_ax.set_ylabel(r"$E/N_{\rm spins}$")
    mag_ax.plot(fields, magnetizations, "bo-", ms=3)
    mag_ax.set(xlabel=r"$h$", ylabel=r"$m_z/m_{\rm sat}$")
    for ax in (energy_ax, mag_ax):
        ax.grid(alpha=0.22)
    finish_plot(fig, title=r"Lowest measured energy per field at $D=3$", show=True, output_path=root / "outputs/figures/lowest_D3")


if __name__ == "__main__":
    main()
