"""Provide functions to optimize peps state, and measure them."""

from hashlib import file_digest
from typing import Any

import torch

from variational_tensor_network._utils.path import find_project_root
from variational_tensor_network._utils.reporting import (
    print_observables,
    print_run_summary,
)
from variational_tensor_network.defaults import (
    DEFAULT_CHI_MAX_STEPS,
    DEFAULT_CHI_STABLE_STEPS,
    DEFAULT_CHI_TOL,
    DEFAULT_CTMRG_MAX_STEPS,
    DEFAULT_CTMRG_TOL,
)
from variational_tensor_network.models import MODELS
from variational_tensor_network.plots.ctmrg import plot_ctmrg_diagnostics

from .ctmrg import (
    PreparedCTMRG,
    compute_ctmrg_environment,
    ctmrg_scan_dir,
    prepare_ctmrg,
    save_ctmrg_result,
    save_scan_summary,
)
from .measurement import (
    measure_magnetization_x,
    measure_magnetization_z,
    measure_spin_spin_correlations,
    validate_observables,
)
from .simple_update import is_usable_su_state, select_or_run_su

root = find_project_root(__file__)


def compute_chi_result(prepared: PreparedCTMRG, chi: int, *, su_provenance: dict[str, Any], max_cycles: int, conv_tol: float, verbosity: int, label: str) -> dict[str, Any]:
    _, info_ctmrg = compute_ctmrg_environment(prepared, chi, max_cycles=max_cycles, conv_tol=conv_tol, verbosity=verbosity, label=label)
    print_run_summary("CTMRG", info_ctmrg, verbosity)
    spins_per_peps_site = prepared.spins_per_peps_site
    energy_per_spin = info_ctmrg["energy_per_peps_site"] / spins_per_peps_site
    observables = None
    measurement_error = None
    if info_ctmrg["status"] == "converged":
        local_x, m_x_over_m_sat = measure_magnetization_x(info_ctmrg["bond_rdms"], spins_per_peps_site)
        local_z, m_z_over_m_sat = measure_magnetization_z(info_ctmrg["bond_rdms"], spins_per_peps_site)
        correlations = measure_spin_spin_correlations(info_ctmrg["bond_rdms"], spins_per_peps_site)
        observables = {"energy_per_spin": energy_per_spin, "m_x_over_m_sat": m_x_over_m_sat, "local_magnetization_x": local_x, "m_z_over_m_sat": m_z_over_m_sat, "local_magnetization_z": local_z, "spin_spin_correlations": correlations}
        try:
            validate_observables(observables)
        except FloatingPointError as error:
            measurement_error = str(error)
            observables = None
    metadata = {
        "schema": 1,
        "su_sha256": su_provenance["sha256"],
        "su_metadata": su_provenance["su_metadata"],
        "chi": chi,
        "tolerance": info_ctmrg["tolerance"],
        "max_cycles": info_ctmrg["max_cycles"],
        "required_stable_cycles": info_ctmrg["required_stable_cycles"],
        "stop_on_convergence": info_ctmrg["stop_on_convergence"],
        "boundary_initialization": info_ctmrg["boundary_initialization"],
        "environment_initialization": info_ctmrg["environment_initialization"],
    }
    return {"metadata": metadata, "info_ctmrg": info_ctmrg, "observables": observables, "measurement_error": measurement_error,}


def finish_chi_scan(info: dict[str, Any], summary: dict[str, Any], *, stable_steps: int, verbosity: int) -> None:
    status = info["status"]
    attempts = summary["attempts"]
    last_chi = attempts[-1]["chi"]
    required_stable_steps = summary["criterion"]["required_stable_chi_iteration"]
    converged = (status == "converged")
    summary["status"] = status
    if converged:
        summary.update(selected_chi=last_chi, m_z_over_m_sat=info["m_z_over_m_sat"], energy_per_spin=info["energy_per_spin"])
        obs = info["selected_observables"]
        local_x = obs["local_magnetization_x"]
        local_z = obs["local_magnetization_z"]
        summary["m_x_over_m_sat"] = obs["m_x_over_m_sat"]
        summary["local_magnetization"] = [ {"site": list(site), "S_x": list(local_x[site]), "S_z": list(local_z[site])} for site in sorted(local_x) ]
        summary["spin_spin_correlations"] = [ { "site": list(site), "direction": direction, "pairs": [{"spins": [i, j], "value": value} for (i, j), value in pairs.items() ]} for (site, direction), pairs in sorted(obs["spin_spin_correlations"].items())]
    scan_dir = info["result_paths_by_chi"][last_chi].parent
    summary_path = save_scan_summary(scan_dir, summary)
    if verbosity >= 1:
        print(f"Chi scan: {status} | last chi={last_chi} | "
              f"{len(attempts)} attempts | "
              f"stable energy differences={stable_steps}/{required_stable_steps} | "
              f"{summary_path}",
              flush=True)
        print(f"Folder hash : {scan_dir.name}")


def chi_scan(prepared: PreparedCTMRG, *, max_chi_iterations: int = DEFAULT_CHI_MAX_STEPS, conv_tol_chi: float = DEFAULT_CHI_TOL, max_ctmrg_cycles: int = DEFAULT_CTMRG_MAX_STEPS, conv_tol_ctmrg: float, required_stable_chi_iteration: int, verbosity: int = 1, su_provenance: dict[str, Any]) -> dict[str, Any]:
    if max_chi_iterations < 1:
        raise ValueError("max_chi_iterations must be at least 1.")
    D = prepared.D
    info = {"history_ctmrg": {}, "result_paths_by_chi": {}, "status": "not_converged", "m_z_over_m_sat": None, "energy_per_spin": None}
    previous_energy_per_spin = None
    stable_steps = 0
    su_metadata = su_provenance["su_metadata"]
    attempts: list[dict[str, Any]] = []
    summary = {
        "point": {key: su_metadata[key] for key in ("model", "D", "tiling", "parameters")},
        "su_sha256": su_provenance["sha256"],
        "criterion": {"observable": "energy_per_spin", "tolerance": conv_tol_chi, "required_stable_chi_iteration": required_stable_chi_iteration, "max_chi_iterations": max_chi_iterations},
        "ctmrg_settings": {"tolerance": conv_tol_ctmrg, "max_cycles": max_ctmrg_cycles},
        "status": "running",
        "attempts": attempts,
        "selected_chi": None,
        "m_z_over_m_sat": None,
        "energy_per_spin": None,
    }
    for index in range(1, max_chi_iterations + 1):
        chi = D**2 + 16 + 2 * D * (index - 1)
        result = compute_chi_result(prepared, chi, su_provenance=su_provenance, max_cycles=max_ctmrg_cycles, conv_tol=conv_tol_ctmrg, verbosity=verbosity, label=f"CTMRG {index}/{max_chi_iterations}")
        info_ctmrg = result["info_ctmrg"]
        observables = result["observables"]
        info["history_ctmrg"][chi] = info_ctmrg
        if index == 1:
            save_scan_summary(ctmrg_scan_dir(result["metadata"]), summary)
        path = save_ctmrg_result(result)
        info["result_paths_by_chi"][chi] = path
        delta_energy = None
        if observables is not None and previous_energy_per_spin is not None:
            delta_energy = abs(observables["energy_per_spin"] - previous_energy_per_spin)
        attempts.append({"chi": chi, "file": path.name, "ctmrg_status": info_ctmrg["status"], "delta_energy_per_spin": delta_energy, "measurement_error": result["measurement_error"],})
        if observables is None:
            info["status"] = ("measurement_failed" if result["measurement_error"] is not None else "ctmrg_failed")
            break
        info["m_z_over_m_sat"] = observables["m_z_over_m_sat"]
        info["energy_per_spin"] = observables["energy_per_spin"]
        if verbosity >= 1:
            print_observables(observables["local_magnetization_z"], observables["m_z_over_m_sat"], observables["energy_per_spin"])
        if delta_energy is not None:
            if delta_energy <= conv_tol_chi:
                stable_steps += 1
            else:
                stable_steps = 0
            if stable_steps >= required_stable_chi_iteration:
                info["status"] = "converged"
                info["selected_observables"] = observables
                break
        previous_energy_per_spin = observables["energy_per_spin"]
    finish_chi_scan(info, summary, stable_steps=stable_steps, verbosity=verbosity)
    return info


def run_chi_scan(model_name: str, parameters_by_model: dict[str, dict[str, float]], tiling_name: str, D: int, seed: int, *, max_chi_iterations: int = DEFAULT_CHI_MAX_STEPS, conv_tol_chi: float = DEFAULT_CHI_TOL, required_stable_chi_iteration: int = DEFAULT_CHI_STABLE_STEPS, max_ctmrg_cycles: int = DEFAULT_CTMRG_MAX_STEPS, conv_tol_ctmrg: float = DEFAULT_CTMRG_TOL, verbosity: int = 1, show_fig: bool = False, save_fig: bool = False) -> dict:
    state_path, _ = select_or_run_su( model_name, parameters_by_model, tiling_name, D, seed, verbosity=verbosity, )
    with state_path.open("rb") as stream:
        su_sha256 = file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
    su_provenance = {"sha256": su_sha256, "su_metadata": state["metadata"]}
    info_su = state["info_su"]
    if not is_usable_su_state(state):
        if verbosity >= 1:
            print(f"Chi scan: su_failed | SU state: {state_path}", flush=True)
        return {"status": "su_failed", "su_state_path": str(state_path), "info_su": info_su, "history_ctmrg": {}, "m_z_over_m_sat": None}
    su_metadata = state["metadata"]
    spins_per_peps_site = MODELS[model_name]["spins_per_peps_site"]
    energy_per_spin = state['info_su']['energy_per_peps_site'] / spins_per_peps_site
    if verbosity >= 1:
        print(f"\nSelected state: {state_path}")
        print(f"SU schedule: {su_metadata['schedule']}")
        print(f"SU energy per spin: {energy_per_spin:.12f}\n", flush=True)
    prepared = prepare_ctmrg(state)
    info = chi_scan(prepared, max_chi_iterations=max_chi_iterations, conv_tol_chi=conv_tol_chi, max_ctmrg_cycles=max_ctmrg_cycles, conv_tol_ctmrg=conv_tol_ctmrg, required_stable_chi_iteration=required_stable_chi_iteration, verbosity=verbosity, su_provenance=su_provenance)
    if show_fig or save_fig:
        plot_ctmrg_diagnostics( info["history_ctmrg"], su_metadata, spins_per_peps_site, show=show_fig, output_dir=root / "outputs" / "figures" / "ctmrg" if save_fig else None)
    return info
