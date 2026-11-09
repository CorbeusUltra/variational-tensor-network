"""Estimate energies using the simple-update product environment."""

from typing import Literal

import torch
from torch import Tensor, einsum

from .contractions import cached_einsum
from .ctmrg_moves import make_accessor
from .objects import BondWeights, EnvSites, SiteWeights
from .peps import TensorSites, get_site_weights, open_double_layer, rotate_site_weights
from .tilings import Site, UnitCellCoord

type Direction = Literal["r", "d"]
type BondRDMs = dict[tuple[Site, Direction], Tensor]


def simple_update_bond_rdm(a: Tensor, b: Tensor, weights_a: SiteWeights, weights_b: SiteWeights, *, direction: Direction) -> Tensor:
    """Build a real bond RDM with physical indices (s, t, S, T)."""
    if direction == "r":
        a, b = a.permute(1, 2, 3, 0, 4), b.permute(1, 2, 3, 0, 4)
        weights_a, weights_b = rotate_site_weights(weights_a), rotate_site_weights(weights_b)
    elif direction != "d":
        raise ValueError(f"Unsupported bond direction: {direction!r}.")
    # torch.testing.assert_close(weights_a.d, weights_b.u)
    abar = einsum("rulds,r,u,l->rulds", a, weights_a.r, weights_a.u, weights_a.l)
    bbar = einsum("rulds,r,l,d->rulds", b, weights_b.r, weights_b.l, weights_b.d)
    abarbbar = einsum("rulqs,RqLDS,q->rulLDRsS", abar, bbar, weights_a.d)
    d = a.shape[-1]
    abarbbar = abarbbar.reshape(-1, d**2)
    rho = einsum("vi,vj->ij", abarbbar, abarbbar)
    normalization = torch.trace(rho)
    if not torch.isfinite(normalization).item() or normalization.item() == 0:
        raise ValueError("The two-site normalization must be finite and nonzero.")
    return (rho / normalization).reshape(d, d, d, d)


def simple_update_rdms(a_sites: TensorSites, bond_weights: BondWeights, unitcell_coord: UnitCellCoord) -> BondRDMs:
    """Build each independent right/down bond RDM once."""
    rdms: BondRDMs = {}
    for site, direction in bond_weights:
        x, y = site
        if direction == "d":
            neighbor = unitcell_coord(x, y + 1)
        elif direction == "r":
            neighbor = unitcell_coord(x + 1, y)
        else:
            raise ValueError(f"Unsupported bond direction: {direction!r}.")
        a, b = a_sites[site], a_sites[neighbor]
        weights_a = get_site_weights(site, bond_weights, unitcell_coord)
        weights_b = get_site_weights(neighbor, bond_weights, unitcell_coord)
        rdms[(site, direction)] = simple_update_bond_rdm(a, b, weights_a, weights_b, direction=direction)
    return rdms


def ctmrg_bond_rdm(a_tilde: Tensor, b_tilde: Tensor, env_sites: EnvSites, site: Site, unitcell_coord: UnitCellCoord, *, direction: Direction) -> Tensor:
    """Build a normalized bond RDM from real tensors with weights absorbed."""
    A_open = open_double_layer(a_tilde)
    B_open = open_double_layer(b_tilde)
    env_at = make_accessor(env_sites, site, unitcell_coord)
    if direction == "d":
        rho = cached_einsum("ai,iuI,IA,bla,rulqsS,ArB,cLb,RqLDtT,BRC,jc,JDj,CJ->stST", env_at(-1, -1).C_ul, env_at(0, -1).T_u, env_at(1, -1).C_ur, env_at(-1, 0).T_l, A_open, env_at(1, 0).T_r, env_at(-1, 1).T_l, B_open, env_at(1, 1).T_r, env_at(-1, 2).C_dl, env_at(0, 2).T_d, env_at(1, 2).C_dr)
    elif direction == "r":
        rho = cached_einsum("ia,aub,bUc,cj,Ili,quldsS,RUqDtT,jRJ,AI,BdA,CDB,JC->stST", env_at(-1, -1).C_ul,  env_at(0, -1).T_u, env_at(1, -1).T_u, env_at(2, -1).C_ur,  env_at(-1, 0).T_l, A_open, B_open, env_at(2, 0).T_r, env_at(-1, 1).C_dl, env_at(0, 1).T_d, env_at(1, 1).T_d, env_at(2, 1).C_dr)
    else:
        raise ValueError(f"Unsupported bond direction: {direction!r}.")
    normalization = einsum("stst->", rho)
    if not torch.isfinite(normalization).item() or normalization.item() == 0:
        raise ValueError("The two-site normalization must be finite and nonzero.")
    return rho / normalization


def ctmrg_rdms(a_tilde_sites: TensorSites, env_sites: EnvSites, unitcell_coord: UnitCellCoord) -> BondRDMs:
    """Build each independent right/down bond RDM once."""
    rdms: BondRDMs = {}
    for site, a_tilde in a_tilde_sites.items():
        x, y = site
        for direction, dx, dy in (("r", 1, 0), ("d", 0, 1)):
            neighbor = unitcell_coord(x + dx, y + dy)
            b_tilde = a_tilde_sites[neighbor]
            rdms[(site, direction)] = ctmrg_bond_rdm(a_tilde, b_tilde, env_sites, site, unitcell_coord, direction=direction)
    return rdms


def expectation_from_rdm(rho: Tensor, operator: Tensor) -> Tensor:
    """Measure a two-site matrix operator using rho[s, t, S, T]."""
    size = rho.shape[0] * rho.shape[1]
    rho_matrix = rho.reshape(size, size)
    if operator.shape != rho_matrix.shape:
        raise ValueError(f"Expected operator.shape == {tuple(rho_matrix.shape)}.")
    return (rho_matrix * operator.mT).sum()




# Legacy code, former method to compute energy per site

# def simple_update_bond_energy(a: Tensor, b: Tensor, weights_a: SiteWeights, weights_b: SiteWeights, h: Tensor) -> float:
#     """Estimate a vertical bond energy, with a above b."""
#     torch.testing.assert_close(weights_a.d, weights_b.u)
#     abar = einsum("rulds,r,u,l->rulds", a, weights_a.r, weights_a.u, weights_a.l)
#     bbar = einsum("rulds,r,l,d->rulds", b, weights_b.r, weights_b.l, weights_b.d)
#     abarbbar = einsum("rulqs,RqLDS,q->rulLDRsS", abar, bbar, weights_a.d)
#     d = a.shape[-1]
#     abarbbar = abarbbar.reshape(-1, d**2)
#     numerator = einsum("vi,ij,vj->", abarbbar, h, abarbbar)
#     denominator = abarbbar.square().sum()
#     if denominator.item() == 0:
#         raise ValueError("Cannot measure a zero-norm pair.")
#     energy = (numerator / denominator).item()
#     # print(f"bond energy : {energy}")
#     return energy


# def simple_update_energy(a_sites: TensorSites, bond_weights: BondWeights, unitcell_coord: UnitCellCoord, h: Tensor) -> float:
#     """Estimate the energy per site, counting each r/d bond once."""
#     total = 0.0
#     for site, direction in bond_weights:
#         x, y = site
#         if direction == "d":
#             neighbor = unitcell_coord(x, y + 1)
#         elif direction == "r":
#             neighbor = unitcell_coord(x + 1, y)
#         a, b = a_sites[site], a_sites[neighbor]
#         weights_a = get_site_weights(site, bond_weights, unitcell_coord)
#         weights_b = get_site_weights(neighbor, bond_weights, unitcell_coord)
#         if direction == "r":
#             a = a.permute(1, 2, 3, 0, 4)
#             b = b.permute(1, 2, 3, 0, 4)
#             weights_a = rotate_site_weights(weights_a)
#             weights_b = rotate_site_weights(weights_b)
#         total += simple_update_bond_energy(a, b, weights_a, weights_b, h)
#     return total / len(a_sites)
