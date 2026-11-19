"""Add the selected CTMRG energy to existing scan summaries."""

import json
from math import isclose, isfinite
from pathlib import Path
from typing import Any

import torch

from variational_tensor_network._utils.path import find_project_root
from variational_tensor_network.simulation.ctmrg import save_scan_summary


def read_selected_energy(scan: dict[str, Any], directory: Path) -> float:
    chi = scan["selected_chi"]
    attempt = next((item for item in scan["attempts"] if item["chi"] == chi), None)
    if attempt is None or attempt["ctmrg_status"] != "converged":
        raise ValueError("No converged attempt for selected_chi.")
    result = torch.load(directory / attempt["file"], map_location="cpu", weights_only=True, mmap=True)
    metadata = result["metadata"]
    source = metadata["su_metadata"]
    if metadata["chi"] != chi or metadata["su_sha256"] != scan["su_sha256"]:
        raise ValueError("The .pt and scan.json have different provenance.")
    if any(source[key] != value for key, value in scan["point"].items()):
        raise ValueError("The .pt describes a different physical point.")
    if any(metadata[key] != value for key, value in scan["ctmrg_settings"].items()):
        raise ValueError("The CTMRG settings differ from the summary.")
    observables = result["observables"]
    if result["info_ctmrg"]["status"] != "converged" or observables is None:
        raise ValueError("The selected CTMRG result is not usable.")
    if not isclose(observables["m_z_over_m_sat"], scan["m_z_over_m_sat"], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("The magnetization differs from the summary.")
    energy = float(observables["energy_per_spin"])
    if not isfinite(energy):
        raise ValueError("Non-finite energy.")
    return energy


def main() -> None:
    directory = find_project_root(__file__) / "data" / "ctmrg"
    updated = failed = 0
    for path in sorted(directory.glob("*/scan.json")):
        try:
            scan = json.loads(path.read_text(encoding="utf-8"))
            if scan["status"] != "converged":
                continue
            existing = scan.get("energy_per_spin")
            if existing is not None and isfinite(existing):
                continue
            scan["energy_per_spin"] = read_selected_energy(scan, path.parent)
            save_scan_summary(path.parent, scan)
            updated += 1
            print(f"Updated: {path}")
        except Exception as error: # noqa: BLE001
            failed += 1
            print(f"Skipped: {path} | {type(error).__name__}: {error}")
    print(f"Updated: {updated} | Errors: {failed}")


if __name__ == "__main__":
    main()
