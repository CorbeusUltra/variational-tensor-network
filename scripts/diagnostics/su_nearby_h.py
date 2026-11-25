"""Follow a D=3 PEPS as the magnetic field increases."""

import json
from pathlib import Path

import numpy as np
import torch

from variational_tensor_network._utils.path import campaign_start_D, find_project_root
from variational_tensor_network.defaults import DEFAULT_SU_MAX_STEPS
from variational_tensor_network.simulation.chi_scan import run_chi_scan
from variational_tensor_network.simulation.simple_update import (
    compute_simple_update,
    save_simple_update_state,
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def energies_by_chi(scan: dict) -> dict[int, float]:
    return {attempt["chi"]: attempt["energy_per_spin"] for attempt in scan.get("attempts", []) if attempt["ctmrg_status"] == "converged" and attempt["energy_per_spin"] is not None}


def main() -> None:
    if campaign_start_D() != 3:
        raise ValueError("Run this diagnostic with VTN_START_D=3.")
    root = find_project_root(__file__)
    output_dir = root / "outputs" / "diagnostics" / "nearby_h_D3_up"
    source_path = root / "outputs" / "diagnostics" / "nearby_h_D3_2.105_to_2.110" / "45bd930f440decafa2ab7a730d81fe03fb44dbce40506f5394c09d5d0d4792ad.pt"
    model, tiling, D, seed = "x_shastry_sutherland", "abcd", 3, 1
    couplings = {"JD": 1.0, "J1": 0.35, "J2": 0.25}
    schedule = ((.01, 1e-6), (.005, 1e-7), (.001, 1e-8))
    last_h = 2.180
    references = {}
    for path in (root / "data" / "campaigns" / "from-D3" / "ctmrg").glob("*/scan.json"):
        scan = read_json(path)
        point = scan["point"]
        if scan.get("sweep_direction") is not None:
            continue
        if (point["model"], point["tiling"], point["D"]) != (model, tiling, D):
            continue
        parameters = point["parameters"]
        if {key: value for key, value in parameters.items() if key != "h"} == couplings:
            references[parameters["h"]] = scan
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    source_h = source["metadata"]["parameters"]["h"]
    h_step = 0.005
    fields = np.arange(source_h + h_step, last_h + h_step / 2, h_step).round(4)
    for h in fields.tolist():
        parameters = {model: {**couplings, "h": h}}
        point_dir = output_dir / f"h={h:.3f}"
        point_dir.mkdir(parents=True, exist_ok=False)
        state = compute_simple_update(model, parameters, tiling, D, seed, schedule, initialization="nearby_h", source_path=source_path, max_sweeps=DEFAULT_SU_MAX_STEPS, verbosity=2)
        state["metadata"]["sweep_direction"] = "up"
        state_path = save_simple_update_state(state, point_dir)
        result = run_chi_scan(model, parameters, tiling, D, seed, su_state_path=state_path)
        paths = result.get("result_paths_by_chi", {})
        scan_path = paths[max(paths)].parent / "scan.json" if paths else None
        scan = read_json(scan_path) if scan_path is not None else {}
        reference = references.get(h, {})
        converged = (result["status"] == "converged")
        nearby = energies_by_chi(scan) if converged else {}
        random = energies_by_chi(reference)
        chi = max(nearby.keys() & random.keys(), default=None)
        delta = nearby[chi] - random[chi] if chi is not None else None
        summary = {
            "h": h,
            "source_h": state["metadata"]["source_h"],
            "source_checkpoint": str(source_path.relative_to(root)),
            "checkpoint": str(state_path.relative_to(root)),
            "su_status": state["info_su"]["status"],
            "chi_status": result["status"],
            "chi_scan": str(scan_path.relative_to(root)) if scan_path is not None else None,
            "selected_chi": scan.get("selected_chi"),
            "energy_per_spin": scan.get("energy_per_spin"),
            "m_z_over_m_sat": scan.get("m_z_over_m_sat"),
            "random_scan_status": reference.get("status"),
            "comparison_chi": chi,
            "nearby_energy_per_spin": nearby.get(chi),
            "random_energy_per_spin": random.get(chi),
            "delta_energy_per_spin": delta,
        }
        summary_path = point_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"h={h:.3f} | chi scan: {result['status']} | {summary_path}", flush=True)
        if not converged:
            break
        if chi is None:
            print("No converged energy at a common chi.", flush=True)
        else:
            print(f"chi={chi} | E_nearby - E_random = {delta:+.3e} per spin", flush=True)
            if delta > 0:
                print("Continuation energy is higher. Stopping at this field.", flush=True)
                break
        source_path = state_path


if __name__ == "__main__":
    main()
