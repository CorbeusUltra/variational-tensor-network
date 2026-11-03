"""Do a simple update."""

from time import perf_counter
from warnings import warn

import torch
from torch import Tensor, einsum
from torch.linalg import vector_norm

from .objects import BondWeights, SiteWeights
from .observables import expectation_from_rdm, simple_update_rdms
from .peps import TensorSites, get_site_weights, rotate_site_weights
from .reporting import format_duration, format_scientific, print_progress
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
    torch.testing.assert_close(weights_a.d, weights_b.u)
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


def simple_update(a_sites: TensorSites, bond_weights: BondWeights, unitcell_coord: UnitCellCoord, H: Tensor, *, schedule: tuple[tuple[float, float], ...], max_sweeps: int = 10000, verbosity: int = 1) -> list[tuple[float, float]]:
    if max_sweeps < 1:
        raise ValueError("max_sweeps must be at least 1.")
    history = []
    d = next(iter(a_sites.values())).shape[-1]
    rdms = simple_update_rdms(a_sites, bond_weights, unitcell_coord)
    energies = {bond: expectation_from_rdm(rho, H).item() for bond, rho in rdms.items()}

    for tau, tol in schedule:
        start = last_update = perf_counter()
        if verbosity >= 1:
            print(f"\ntau={format_scientific(tau)}, tol={format_scientific(tol)}", flush=True)
        G = torch.linalg.matrix_exp(-tau * H).reshape(d, d, d, d)
        converged = False
        for sweep in range(1, max_sweeps + 1):
            old_energies = energies
            simple_update_sweep(a_sites, bond_weights, unitcell_coord, G)
            rdms = simple_update_rdms(a_sites, bond_weights, unitcell_coord)
            energies = {bond: expectation_from_rdm(rho, H).item() for bond, rho in rdms.items()}
            # error = abs(sum(energies[bond] - old_energies[bond] for bond in energies)) # |∑_bonds (E_new - E_old)| ; tracks total unit-cell energy, but bond changes can cancel
            error = max(abs(energies[bond] - old_energies[bond]) for bond in energies) # max_bonds |E_new - E_old| ; requires every bond-energy change to be within tolerance
            history.append((tau, error))
            now = perf_counter()
            if sweep == 1 or now - last_update >= 1.0:
                print_progress(f"  sweep={sweep}/{max_sweeps} | error={error:.2e} | time={format_duration(now - start)}", verbosity)
                last_update = now
            if error <= tol:
                converged = True
                break
        status = "Converged" if converged else "Not converged"
        print_progress(f"  {status} | sweep={sweep}/{max_sweeps} | error={error:.2e} | time={format_duration(perf_counter() - start)}", verbosity, final=True)
        if not converged:
            D = next(iter(a_sites.values())).shape[0]
            warn(f"\n\nSimple update did not converge after {max_sweeps} sweeps "
                f"(D={D}, tau={tau:.3g}).\n"
                f"Maximum bond-energy change: {error:.2e}.\n"
                f"Tolerance: {tol:.2e}.\n"
                "Returning the last state and skipping any remaining tau stages.\n",
                RuntimeWarning,
                stacklevel=2)
            return history
    return history
