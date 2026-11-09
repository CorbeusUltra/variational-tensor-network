"""Measure a saved PEPS state at one CTMRG bond dimension."""


from math import isfinite # noqa: I001

from variational_tensor_network.ctmrg import ctmrg
from variational_tensor_network.initialize import initial_boundary, initial_env_sites
from variational_tensor_network.models import MODELS
from variational_tensor_network.objects import Config
from variational_tensor_network.peps import absorb_sqrt_weights, double_layer, get_site_weights
from variational_tensor_network.plots._utils import find_project_root
from variational_tensor_network.reporting import print_run_summary
from variational_tensor_network.state_io import matching_states
from variational_tensor_network.tilings import TILINGS


def main() -> None:
    chi = 4
    verbosity = 1
    state_query = {
        "model": "heisenberg",
        "parameters": {"J": 1.0, "h": 0},
        "tiling": "checkerboard",
        "D": 2,
        "seed": 0,
    }
    root = find_project_root(__file__)
    state_dir = root / "data" / "simple_update"
    candidates = (
        (path, state)
        for path, state in matching_states(state_dir, state_query)
        if state["su_info"]["status"] == "converged"
        and isfinite(state["su_info"]["energy_per_peps_site"])
    )
    selected = min(candidates, key=lambda item: item[1]["su_info"]["energy_per_peps_site"], default=None)
    if selected is None:
        raise ValueError(f"No converged state matching {state_query} in '{state_dir}'.")
    state_path, state = selected
    metadata = state["metadata"]
    if verbosity >= 1:
        print(f"\nSelected state: {state_path}")
        print(f"SU schedule: {metadata['schedule']}")
        print(f"SU energy per PEPS site: {state['su_info']['energy_per_peps_site']:.12f}\n", flush=True)

    H = MODELS[metadata["model"]]["bond"](**metadata["parameters"])
    config = Config(D=metadata["D"], d=metadata["d"], chi=chi)
    _, unitcell_coord = TILINGS[metadata["tiling"]]
    a_tilde_sites = {}
    for site, a in state["a_sites"].items():
        weights = get_site_weights(site, state["bond_weights"], unitcell_coord)
        a_tilde_sites[site] = absorb_sqrt_weights(a, weights)
    A_sites = {site: double_layer(a_tilde) for site, a_tilde in a_tilde_sites.items()}
    env_sites = initial_env_sites(A_sites, initial_boundary(config), config)
    _, info = ctmrg(env_sites, a_tilde_sites, unitcell_coord, config, H, conv_tol=1e-8, verbosity=verbosity)
    print_run_summary("CTMRG", info, verbosity)


if __name__ == "__main__":
    main()
