"""Plot chi diagnostics from a saved scan."""

import json
from math import isfinite
from pathlib import Path

import torch

from variational_tensor_network._utils.path import find_project_root
from variational_tensor_network.plots.ctmrg import plot_energy_vs_chi


def read_energy_vs_chi(scan_path: Path) -> dict[int, float]:
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    energies: dict[int, float] = {}
    for attempt in scan["attempts"]:
        if attempt["ctmrg_status"] != "converged":
            continue
        if "energy_per_spin" in attempt:
            energy = attempt["energy_per_spin"]
            if energy is not None and isfinite(energy):
                energies[attempt["chi"]] = energy
            continue
        path = scan_path.parent / attempt["file"]
        if not path.is_file():
            print(f"Missing CTMRG result: {path}")
            continue
        result = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        observables = result["observables"]
        if observables is not None and isfinite(observables["energy_per_spin"]):
            energies[attempt["chi"]] = observables["energy_per_spin"]
    return energies


def main() -> None:
    root = find_project_root(__file__)
    scan_id = "a3a01a903774d84d02c07cc794a8211ef738dab3e454efe6564ba21728037069" # "d1a5f6bd94012508a4c48f0365ec4b2848392fef343f89d0e074fed736cc806a"
    scan_path = root / "data" / "ctmrg" / scan_id / "scan.json"
    show_fig = True
    save_fig = False

    if not (show_fig or save_fig):
        return
    if not scan_path.is_file():
        print(f"Missing scan summary: {scan_path}")
        return
    energies = read_energy_vs_chi(scan_path)
    chi_values = sorted(energies)
    output_dir = root / "outputs" / "figures" / "ctmrg" / scan_id
    plot_energy_vs_chi(chi_values, [energies[chi] for chi in chi_values], show=show_fig, output_path=output_dir / "energy_vs_chi" if save_fig else None)


if __name__ == "__main__":
    main()