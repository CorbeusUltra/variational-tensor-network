"""Do a simple update."""

from time import perf_counter
from warnings import warn

import torch
from torch import Tensor, einsum
from torch.linalg import vector_norm

from ._utils.reporting import format_duration, print_progress
from .objects import BondWeights, SiteWeights
from .observables import expectation_from_rdm
from .peps import TensorSites, get_site_weights, rotate_site_weights
from .rdms import simple_update_rdms
from .tilings import UnitCellCoord


def inverse_weights(weights: Tensor, rtol: float = 1e-12) -> Tensor:
    scale = weights.max()
    if scale == 0:
        raise ValueError("Cannot invert an identically zero bond weight.")
    mask = weights > rtol * scale
    inverse = torch.zeros_like(weights)
    inverse[mask] = 1 / weights[mask]
    return inverse


def simple_update_iteration(a: Tensor, b: Tensor, weights_a: SiteWeights, weights_b: SiteWeights, G: Tensor) -> tuple[Tensor,Tensor,Tensor]:
    """
    Assuming the common weight is the down weight of a and the up weight of b.
    """
    abar = einsum("rulds,r,u,l->rulds", a, weights_a.r, weights_a.u, weights_a.l)
    bbar = einsum("rulds,r,l,d->rulds", b, weights_b.r, weights_b.l, weights_b.d)

    abarbbar = einsum("rulqs,RqLDS,q->rulLDRsS",abar,bbar,weights_a.d)
    M = einsum("rulLDRsS,tTsS->rulLDRtT", abarbbar, G) # M or abarbbar_gated
    D, d = a.shape[3], a.shape[-1]
    M = M.permute(0,1,2,6,3,4,5,7)
    M = M.reshape(D**3 * d, D**3 * d)
    U, S, Vh = torch.linalg.svd(M, full_matrices=False)
    V = Vh.mT
    U, S, V = U[:, :D], S[:D], V[:, :D]
    U, V = U.reshape(D,D,D,d,D), V.reshape(D,D,D,d,D)
    U = U.permute(0,1,2,4,3)
    V = V.permute(2,4,0,1,3)
    a_new = einsum("rulds,r,u,l->rulds", U, inverse_weights(weights_a.r), inverse_weights(weights_a.u), inverse_weights(weights_a.l))
    b_new = einsum("rulds,r,l,d->rulds", V, inverse_weights(weights_b.r), inverse_weights(weights_b.l), inverse_weights(weights_b.d))
    a_new /= vector_norm(a_new)
    b_new /= vector_norm(b_new)
    S /= vector_norm(S)
    return a_new, b_new, S


def simple_update_sweep(a_sites: TensorSites, bond_weights: BondWeights, unitcell_coord: UnitCellCoord, G: Tensor) -> None:
    """Update site tensors and bond weights in place during one sweep."""
    for site, direction in bond_weights:
        x, y = site
        if direction == "d":
            neighbor = unitcell_coord(x, y + 1)
        elif direction == "r":
            neighbor = unitcell_coord(x + 1, y)
        if neighbor == site:
            warn(f"Bond {site}--{direction}-->{neighbor} connects the same site representative. " "This sweep overwrites one of the two updated tensors; results may be unreliable.", RuntimeWarning)
        a, b = a_sites[site], a_sites[neighbor]
        weights_a = get_site_weights(site, bond_weights, unitcell_coord)
        weights_b = get_site_weights(neighbor, bond_weights, unitcell_coord)
        if direction == "r":
            a = a.permute(1, 2, 3, 0, 4)
            b = b.permute(1, 2, 3, 0, 4)
            weights_a = rotate_site_weights(weights_a)
            weights_b = rotate_site_weights(weights_b)
        a_new, b_new, lambda_new = simple_update_iteration(a, b, weights_a, weights_b, G)
        if direction == "r":
            a_new = a_new.permute(3, 0, 1, 2, 4)
            b_new = b_new.permute(3, 0, 1, 2, 4)
        a_sites[site] = a_new
        a_sites[neighbor] = b_new
        bond_weights[(site, direction)] = lambda_new


def simple_update(a_sites: TensorSites, bond_weights: BondWeights, unitcell_coord: UnitCellCoord, H: Tensor, *, schedule: tuple[tuple[float, float], ...] = ((.05, 1e-4), (.01, 1e-6), (.005, 1e-8)), max_sweeps: int = 10000, required_stable_sweeps: int = 100, verbosity: int = 1) -> tuple[TensorSites, BondWeights, dict]:
    """Optimize a PEPS without modifying the input dictionaries or their tensors."""
    if max_sweeps < 1:
        raise ValueError("max_sweeps must be at least 1.")
    if not schedule:
        raise ValueError("schedule must contain at least one tau stage.")
    if verbosity >= 1:
        print("Simple update", flush=True)
    total_start = perf_counter()
    a_sites = a_sites.copy()
    bond_weights = bond_weights.copy()
    sweep_width = len(str(max_sweeps))
    info = {
        "schedule": schedule,
        "max_sweeps_per_stage": max_sweeps,
        "tau_error_history": [],
        "energy_history": [],
        "stages": [],
    }
    d = next(iter(a_sites.values())).shape[-1]
    rdms = simple_update_rdms(a_sites, bond_weights, unitcell_coord)
    energies = {bond: expectation_from_rdm(rho, H).item() for bond, rho in rdms.items()}
    info["initial_energy_per_peps_site"] = sum(energies.values()) / len(a_sites)
    for tau, tol in schedule:
        start = last_update = perf_counter()
        update_interval = 0.1 if verbosity >= 3 else 1.0
        if verbosity >= 1:
            print(f"\n  tau={tau:.2e}, tol={tol:.2e}", flush=True)
        G = torch.linalg.matrix_exp(-tau * H).reshape(d, d, d, d)
        converged = False
        stable_sweeps = 0
        for sweep in range(1, max_sweeps + 1):
            old_energies = energies
            simple_update_sweep(a_sites, bond_weights, unitcell_coord, G)
            rdms = simple_update_rdms(a_sites, bond_weights, unitcell_coord)
            energies = {bond: expectation_from_rdm(rho, H).item() for bond, rho in rdms.items()}
            energy_per_peps_site = sum(energies.values()) / len(a_sites)
            # error = abs(sum(energies[bond] - old_energies[bond] for bond in energies)) # |∑_bonds (E_new - E_old)| ; tracks total unit-cell energy, but bond changes can cancel
            error = max(abs(energies[bond] - old_energies[bond]) for bond in energies) # max_bonds |E_new - E_old| ; requires every bond-energy change to be within tolerance
            stable_sweeps = stable_sweeps + 1 if error <= tol else 0
            info["tau_error_history"].append((tau, error))
            info["energy_history"].append(energy_per_peps_site)
            now = perf_counter()
            progress = f"    sweep={sweep:>{sweep_width}}/{max_sweeps} | error={error:9.2e} | stable={stable_sweeps}/{required_stable_sweeps} | time={format_duration(now - start)}"
            if verbosity >= 4 or sweep == 1 or now - last_update >= update_interval:
                print_progress(progress, verbosity)
                last_update = now
            if stable_sweeps >= required_stable_sweeps:
                converged = True
                break
        info["stages"].append({
            "tau": tau,
            "tolerance": tol,
            "sweeps": sweep,
            "final_error": error,
            "duration": perf_counter() - start,
            "status": "converged" if converged else "not_converged",
        })
        print_progress(progress, verbosity, final=True)
        if verbosity >= 1:
            status = "tau stage converged" if converged else "tau stage did not converge"
            print(f"    {status}", flush=True)
        if not converged:
            D = next(iter(a_sites.values())).shape[0]
            warn(f"\n\nSimple update did not converge after {max_sweeps} sweeps "
                f"(D={D}, tau={tau:.3g}).\n"
                f"Maximum bond-energy change: {error:.2e}.\n"
                f"Tolerance: {tol:.2e}.\n"
                "Returning the last state and skipping any remaining tau stages.\n",
                RuntimeWarning,
                stacklevel=2)
            break
    info.update({
        "status": "converged" if converged else "not_converged",
        "duration": perf_counter() - total_start,
        "energy_per_peps_site": energy_per_peps_site,
        "bond_energies": energies.copy(),
        "total_sweeps": len(info["tau_error_history"]),
    })
    return a_sites, bond_weights, info
