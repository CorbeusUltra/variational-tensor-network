"""Plot CTMRG diagnostics from a saved scan."""

import json
from math import isfinite
from pathlib import Path

import torch

from variational_tensor_network._utils.path import find_project_root
from variational_tensor_network.plots.ctmrg import (
    plot_ctmrg_convergence,
    plot_energy_vs_chi,
)


def read_energy_vs_chi(scan_path: Path) -> tuple[int, dict[int, float]]:
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    energies: dict[int, float] = {}
    for attempt in scan["attempts"]:
        if attempt["ctmrg_status"] != "converged":
            continue
        path = scan_path.parent / attempt["file"]
        if not path.is_file():
            print(f"Missing CTMRG result: {path}")
            continue
        result = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        observables = result["observables"]
        if observables is not None and isfinite(observables["energy_per_spin"]):
            energies[attempt["chi"]] = observables["energy_per_spin"]
    return scan["point"]["D"], energies


def main() -> None:
    root = find_project_root(__file__)
    scan_dir = root / "data" / "ctmrg" / "3465ddae33c00622187a14be54776b52f40170a7c0ed91b640a3b7a2476f2fb0"
    plot_chi_scan = True
    ctmrg_chi: int | None = 20
    show_fig = True
    save_fig = False

    if not (show_fig or save_fig):
        return
    output_dir = root / "outputs" / "figures" / "ctmrg" / scan_dir.name
    if plot_chi_scan:
        scan_path = scan_dir / "scan.json"
        if scan_path.is_file():
            D, energies = read_energy_vs_chi(scan_path)
            if len(energies) >= 2:
                chi_values = sorted(energies)
                plot_energy_vs_chi(chi_values, [energies[chi] for chi in chi_values], D, show=show_fig, output_path=output_dir / "energy_vs_chi" if save_fig else None)
            else:
                print(f"Need at least two converged results in {scan_path}.")
        else:
            print(f"Missing scan summary: {scan_path}")
    if ctmrg_chi is not None:
        path = scan_dir / f"chi={ctmrg_chi}.pt"
        if not path.is_file():
            print(f"Missing CTMRG result: {path}")
            return
        result = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        info = result["info_ctmrg"]
        errors = [error.max().item() for error in info["error_history"]]
        if not errors:
            print(f"No CTMRG cycle history in {path}.")
            return
        plot_ctmrg_convergence(errors, title=rf"CTMRG | $\chi={ctmrg_chi}$ | {info['status']}", tolerance=info["tolerance"], show=show_fig, output_path=output_dir / f"cycles_chi={ctmrg_chi}" if save_fig else None)


if __name__ == "__main__":
    main()
