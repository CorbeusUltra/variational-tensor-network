import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import torch

from variational_tensor_network._utils.path import campaign_data_dir, find_project_root
from variational_tensor_network.ctmrg import ctmrg
from variational_tensor_network.initialize import (
    initial_boundary,
    initial_env_sites,
)
from variational_tensor_network.models import MODELS, model_bond_hamiltonians
from variational_tensor_network.objects import BondOperators, Config, EnvSites
from variational_tensor_network.peps import (
    TensorSites,
    absorb_sqrt_weights,
    double_layer,
    get_site_weights,
)
from variational_tensor_network.plots.ctmrg import plot_ctmrg_convergence
from variational_tensor_network.tilings import TILINGS, UnitCellCoord

from .simple_update import is_usable_su_state, select_or_run_su

root = find_project_root(__file__)


@dataclass
class PreparedCTMRG:
    D: int
    d: int
    H: BondOperators
    unitcell_coord: UnitCellCoord
    a_tilde_sites: TensorSites
    A_sites: TensorSites
    spins_per_peps_site: int


def prepare_ctmrg(state: dict[str, Any]) -> PreparedCTMRG:
    metadata = state["metadata"]
    model = MODELS[metadata["model"]]
    sites_coords, unitcell_coord = TILINGS[metadata["tiling"]]
    H = model_bond_hamiltonians(metadata["model"], metadata["parameters"], sites_coords)
    _, unitcell_coord = TILINGS[metadata["tiling"]]
    a_tilde_sites: TensorSites = {}
    for site, a in state["a_sites"].items():
        weights = get_site_weights(site, state["bond_weights"], unitcell_coord)
        a_tilde_sites[site] = absorb_sqrt_weights(a, weights)
    A_sites = {site: double_layer(a_tilde) for site, a_tilde in a_tilde_sites.items()}
    return PreparedCTMRG(D=metadata["D"], d=metadata["d"], H=H, unitcell_coord=unitcell_coord, a_tilde_sites=a_tilde_sites, A_sites=A_sites, spins_per_peps_site=model["spins_per_peps_site"])


def compute_ctmrg_environment(prepared: PreparedCTMRG, chi: int, *, max_cycles: int, conv_tol: float, verbosity: int = 1, label: str = "CTMRG") -> tuple[EnvSites, dict[str, Any]]:
    config = Config(D=prepared.D, d=prepared.d, chi=chi)
    boundary_method = "ones"
    environment_method = "from_bulk"
    boundary = initial_boundary(config, method=boundary_method)
    initial_env = initial_env_sites(prepared.A_sites, boundary, config, method=environment_method)
    env_sites, info_ctmrg = ctmrg(initial_env, prepared.a_tilde_sites, prepared.unitcell_coord, config, prepared.H, max_cycles=max_cycles, conv_tol=conv_tol, verbosity=verbosity, label=label)
    info_ctmrg["boundary_initialization"] = boundary_method
    info_ctmrg["environment_initialization"] = environment_method
    return env_sites, info_ctmrg


def run_ctmrg(model_name: str, parameters_by_model: dict[str, dict[str, float]], tiling_name: str, D: int, seed: int, *, chi: int, max_cycles: int, conv_tol: float, verbosity: int = 1, show_fig: bool = False, save_fig: bool = False, by_site: bool = False) -> tuple[Path, dict[str, Any]]:
    """Run and save a fixed-chi CTMRG diagnostic."""
    state_path, state = select_or_run_su(model_name, parameters_by_model, tiling_name, D, seed, verbosity=verbosity)
    if not is_usable_su_state(state):
        raise RuntimeError(f"Unusable Simple Update state: {state_path}")
    print(f"PEPS state: {state_path}", flush=True)
    _, info = compute_ctmrg_environment(prepare_ctmrg(state), chi, max_cycles=max_cycles, conv_tol=conv_tol, verbosity=verbosity)
    result = {"metadata": {"su_metadata": state["metadata"]}, "info_ctmrg": info}
    results_dir = ctmrg_scan_dir(result["metadata"])
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"diagnostic_chi={chi}.pt"
    with NamedTemporaryFile("wb", dir=results_dir, delete=False) as stream:
        torch.save(result, stream)
    Path(stream.name).replace(result_path)
    print(f"CTMRG result saved to {result_path}", flush=True)
    if show_fig or save_fig:
        output_dir = root / "outputs" / "figures" / "ctmrg" / results_dir.name
        output_path = output_dir / f"cycles_{result_path.stem}" if save_fig else None
        plot_ctmrg_convergence(
            result, by_site=by_site, show=show_fig, output_path=output_path,
        )
    return result_path, result


def ctmrg_scan_dir(metadata: dict[str, Any]) -> Path:
    su_metadata = metadata["su_metadata"]
    scan_identity = {key: su_metadata[key] for key in ("schema", "model", "parameters", "tiling", "D", "d") }
    direction = su_metadata.get("sweep_direction")
    if direction is not None:
        scan_identity["sweep_direction"] = direction
    key = json.dumps(scan_identity, sort_keys=True, allow_nan=False).encode("utf-8")
    tag = sha256(key).hexdigest()
    return campaign_data_dir(root) / "ctmrg" / tag


def save_ctmrg_result(result: dict[str, Any]) -> Path:
    metadata = result["metadata"]
    chi = metadata["chi"]
    output_path = ctmrg_scan_dir(metadata) / f"chi={chi}.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", dir=output_path.parent, delete=False) as stream:
        torch.save(result, stream)
    Path(stream.name).replace(output_path)
    return output_path


def save_scan_summary(scan_dir: Path, summary: dict[str, Any]) -> Path:
    scan_dir.mkdir(parents=True, exist_ok=True)
    output_path = scan_dir / "scan.json"
    with NamedTemporaryFile("w", dir=scan_dir, suffix=".tmp", encoding="utf-8", delete=False) as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")
    Path(stream.name).replace(output_path)
    return output_path
