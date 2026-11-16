"""Provide functions to optimize peps state, and measure them."""

from hashlib import file_digest
from math import isfinite
from typing import Any

import torch

from variational_tensor_network._utils.path import find_project_root
from variational_tensor_network._utils.reporting import (
    print_boxed,
    print_run_summary,
)
from variational_tensor_network.models import MODELS
from variational_tensor_network.plots.ctmrg import plot_ctmrg_diagnostics

from .ctmrg import (
    PreparedCTMRG,
    compute_ctmrg_environment,
    prepare_ctmrg,
    save_ctmrg_result,
)
from .measurement import measure_magnetization_z, print_ctmrg_observables
from .simple_update import matching_states, run_simple_update

root = find_project_root(__file__)


def chi_scan(prepared: PreparedCTMRG, *, max_chi_iterations: int = 10, conv_tol_chi: float = 1e-6, max_ctmrg_cycles: int = 2000, conv_tol_ctmrg: float = 1e-7, verbosity: int = 1, su_provenance: dict[str, Any]) -> dict[str, Any]:
    if max_chi_iterations < 1:
        raise ValueError("max_chi_iterations must be at least 1.")
    D = prepared.D
    spins_per_peps_site = prepared.spins_per_peps_site
    info = {"history_ctmrg": {}, "result_paths_by_chi": {}, "status": "not_converged", "m_z_over_m_sat": None}
    previous_energy_per_spin = None
    stable_steps = 0
    required_stable_steps = 2
    for index in range(1, max_chi_iterations + 1):
        chi = D**2 + 16 + 2 * D * (index - 1)
        _, info_ctmrg = compute_ctmrg_environment(prepared, chi, max_cycles=max_ctmrg_cycles, conv_tol=conv_tol_ctmrg, verbosity=verbosity, label=f"CTMRG {index}/{max_chi_iterations}")
        info["history_ctmrg"][chi] = info_ctmrg
        print_run_summary("CTMRG", info_ctmrg, verbosity)
        energy_per_spin = info_ctmrg["energy_per_peps_site"] / spins_per_peps_site
        observables = None
        if info_ctmrg["status"] == "converged" and isfinite(energy_per_spin):
            local_z, m_z_over_m_sat = measure_magnetization_z(info_ctmrg["bond_rdms"], spins_per_peps_site)
            if isfinite(m_z_over_m_sat):
                observables = {"energy_per_spin": energy_per_spin, "m_z_over_m_sat": m_z_over_m_sat, "local_magnetization_z": local_z}
        metadata = {
            "schema" : 1,
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
        result = {"metadata": metadata, "info_ctmrg": info_ctmrg, "observables": observables}
        path = save_ctmrg_result(result)
        info["result_paths_by_chi"][chi] = path
        if observables is None:
            info["status"] = "ctmrg_failed"
            break
        info["m_z_over_m_sat"] = observables["m_z_over_m_sat"]
        if verbosity >= 1:
            print_ctmrg_observables(observables["local_magnetization_z"], observables["m_z_over_m_sat"], energy_per_spin)
        if previous_energy_per_spin is not None:
            delta_energy = abs(energy_per_spin - previous_energy_per_spin)
            if delta_energy <= conv_tol_chi:
                stable_steps += 1
            else:
                stable_steps = 0
            if stable_steps >= required_stable_steps:
                info["status"] = "converged"
                break
        previous_energy_per_spin = energy_per_spin
    return info


def run_chi_scan(model_name: str, parameters_by_model: dict[str, dict[str, float]], tiling_name: str, D: int, seed: int, *, max_chi_iterations: int = 10, conv_tol_chi: float = 1e-6, max_ctmrg_cycles: int = 2000, conv_tol_ctmrg: float = 1e-7, verbosity: int = 1, show_fig: bool = False, save_fig: bool = False) -> dict:
    parameters = parameters_by_model[model_name]
    state_query = {"model": model_name, "parameters": parameters, "tiling": tiling_name, "D": D}
    state_dir = root / "data" / "simple_update"
    candidates = ((path, state) for path, state in matching_states(state_dir, state_query) if state["info_su"]["status"] == "converged" and isfinite(state["info_su"]["energy_per_peps_site"]))
    selected = min(candidates, key=lambda item: item[1]["info_su"]["energy_per_peps_site"], default=None)
    if selected is None:
        if verbosity >= 1:
            print("\nNo converged saved state found. Running Simple Update.", flush=True)
        selected = run_simple_update(model_name, parameters_by_model, tiling_name, D, seed, verbosity=verbosity, show_fig=False, save_fig=False)
    elif verbosity >= 1:
        model = MODELS[model_name]
        parameters_text = ", ".join(f"{name}={value:g}" for name, value in parameters.items())
        heading = f"{model['label']} ({parameters_text}) | {tiling_name} | (D={D}, d={model["d"]})"
        print_boxed(heading)
    state_path, state = selected
    with state_path.open("rb") as stream:
        su_sha256 = file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        state = torch.load(stream, map_location="cpu", weights_only=True)
    su_provenance = {"sha256": su_sha256, "su_metadata": state["metadata"]}
    info_su = state["info_su"]
    if info_su["status"] != "converged" or not isfinite(info_su["energy_per_peps_site"]):
        return {"status": "su_failed", "su_state_path": str(state_path), "info_su": info_su, "history_ctmrg": {}, "m_z_over_m_sat": None}
    su_metadata = state["metadata"]
    spins_per_peps_site = MODELS[model_name]["spins_per_peps_site"]
    energy_per_spin = state['info_su']['energy_per_peps_site'] / spins_per_peps_site
    if verbosity >= 1:
        print(f"\nSelected state: {state_path}")
        print(f"SU schedule: {su_metadata['schedule']}")
        print(f"SU energy per spin: {energy_per_spin:.12f}\n", flush=True)
    prepared = prepare_ctmrg(state)
    info = chi_scan(prepared, max_chi_iterations=max_chi_iterations, conv_tol_chi=conv_tol_chi, max_ctmrg_cycles=max_ctmrg_cycles, conv_tol_ctmrg=conv_tol_ctmrg, verbosity=verbosity, su_provenance=su_provenance)
    if show_fig or save_fig:
        plot_ctmrg_diagnostics( info["history_ctmrg"], su_metadata, spins_per_peps_site, show=show_fig, output_dir=root / "outputs" / "figures" / "analysis" if save_fig else None)
    return info
