"""Make one CTMRG move, here on the left of the environments."""

from collections.abc import Callable, Mapping

import torch
from torch import Tensor, einsum
from torch.linalg import norm

from .contractions import cached_einsum
from .objects import Config, Environment, EnvSites
from .peps import TensorSites
from .tilings import Site, UnitCellCoord, rotated_unitcell_coord


def make_accessor[Object](objects_sites: Mapping[Site, Object], site: Site, unitcell_coord: UnitCellCoord) -> Callable[[int, int], Object]:
    x, y = site

    def at(dx: int, dy: int) -> Object:
        return objects_sites[unitcell_coord(x + dx, y + dy)]

    return at


def get_halves(site: Site, A_sites: TensorSites, env_sites: EnvSites, unitcell_coord: UnitCellCoord) -> tuple[Tensor, Tensor]:
    """
    The reference site is the upper-left bulk tensor of the 2x2 patch.
    """
    A_at = make_accessor(A_sites, site, unitcell_coord)
    env_at = make_accessor(env_sites, site, unitcell_coord)
    Q_ul = cached_einsum("IA,AJB,iaI,bJaj->ijBb", env_at(-1,-1).C_ul, env_at(0,-1).T_u, env_at(-1,0).T_l, A_at(0,0))
    Q_ur = cached_einsum("BKC,CL,cKbk,Lcl->Bbkl", env_at(+1,-1).T_u, env_at(+2,-1).C_ur, A_at(+1,0), env_at(+2,0).T_r)
    Q_dl = cached_einsum("AI,BJA,Iai,bjaJ->ijBb", env_at(-1,+2).C_dl, env_at(0,+2).T_d, env_at(-1,+1).T_l, A_at(0,+1))
    Q_dr = cached_einsum("CKB,LC,ckbK,lcL->Bbkl", env_at(+1,+2).T_d, env_at(+2,+2).C_dr, A_at(+1,+1), env_at(+2,+1).T_r)
    # R    = einsum( "IA,AJB,BKC,CL,iaI,bJaj,cKbk,Lcl->ijkl", env_at(-1, -1).C_ul, env_at(0, -1).T_u, env_at(+1, -1).T_u, env_at(+2, -1).C_ur, env_at(-1, 0).T_l, A_at(0, 0), A_at(+1, 0), env_at(+2, 0).T_r)
    # Rbar = einsum( "AI,BJA,CKB,LC,Iai,bjaJ,ckbK,lcL->ijkl", env_at(-1, +2).C_dl, env_at(0, +2).T_d, env_at(+1, +2).T_d, env_at(+2, +2).C_dr, env_at(-1, +1).T_l, A_at(0, +1), A_at(+1, +1), env_at(+2, +1).T_r)
    R = cached_einsum("ijBb,Bbkl->ijkl", Q_ul, Q_ur)
    Rbar = cached_einsum("ijBb,Bbkl->ijkl", Q_dl, Q_dr)
    return R, Rbar


def get_projectors(R: Tensor, Rbar: Tensor, config: Config) -> tuple[Tensor, Tensor]:
    RRbar    = einsum("abij,abkl->ijkl",R,Rbar)
    dim_i, dim_j, dim_k, dim_l = RRbar.shape
    r_left   = dim_i * dim_j
    r_right  = dim_k * dim_l
    M = RRbar.reshape(r_left, r_right)
    scale = M.abs().max()
    if scale.item() == 0:
        raise ValueError("The projector matrix is identically zero.")
    U, S, Vh = torch.linalg.svd(M / scale, full_matrices=False)
    S = S * scale
    V = Vh.mT
    rtol = max(M.shape) * torch.finfo(S.dtype).eps
    rank = (S > rtol * S[0]).sum().item()
    if rank == 0:
        raise ValueError("M has no numerically significant singular values.")
    # print(f"\n\nlambda_0: {S[0]}\nlambda_1: {S[1]}\n")
    # xi = S.new_zeros(()) if rank == 1 else 1 / (torch.log(S[0]) - torch.log(S[1]))
    p = min(config.chi, rank)
    U, S, V = U[:, :p], S[:p], V[:, :p]
    U = U.reshape(dim_i, dim_j, p)
    V = V.reshape(dim_k, dim_l, p)
    Srsqrt = S.rsqrt()
    P    = einsum("ijkl,klm->ijm", R   , U) * Srsqrt
    Pbar = einsum("ijkl,klm->ijm", Rbar, V) * Srsqrt
    return P, Pbar


def absorb_left(site: Site, A_sites: TensorSites, env_sites: EnvSites, P_sites: TensorSites, Pbar_sites: TensorSites, unitcell_coord: UnitCellCoord) -> tuple[Tensor, Tensor, Tensor]:
    A_at = make_accessor(A_sites, site, unitcell_coord)
    env_at = make_accessor(env_sites, site, unitcell_coord)
    P_at = make_accessor(P_sites, site, unitcell_coord)
    Pbar_at = make_accessor(Pbar_sites, site, unitcell_coord)
    C_ul_new = einsum("ijk,il,ljm->km",Pbar_at(0,-1),env_at(-1,-1).C_ul,env_at(0,-1).T_u)
    T_l_new = einsum("ijkl,nkm,mjo,nlp->pio", A_at(0,0), env_at(-1,0).T_l, P_at(0,-1), Pbar_at(0,0))
    C_dl_new = einsum("ijk,li,mjl->mk", P_at(0,0), env_at(-1,+1).C_dl, env_at(0,+1).T_d)
    C_ul_new /= norm(C_ul_new)
    T_l_new /= norm(T_l_new)
    C_dl_new /= norm(C_dl_new)
    return C_ul_new, T_l_new, C_dl_new


def left_move_sites(A_sites: TensorSites, env_sites: EnvSites, unitcell_coord: UnitCellCoord, config: Config) -> EnvSites:
    """Perform a left move, updating env_sites in place."""
    P_sites: TensorSites = {}
    Pbar_sites: TensorSites = {}
    for site in A_sites:
        R,Rbar = get_halves(site, A_sites,env_sites, unitcell_coord)
        P,Pbar = get_projectors (R,Rbar,config)
        P_sites[site] = P
        Pbar_sites[site] = Pbar
    C_ul_updates: TensorSites = {}
    T_l_updates: TensorSites = {}
    C_dl_updates: TensorSites = {}
    for site in A_sites:
        x, y = site
        C_ul_new, T_l_new, C_dl_new = absorb_left(site, A_sites, env_sites, P_sites, Pbar_sites, unitcell_coord)
        C_ul_updates[unitcell_coord(x, y - 1)] = C_ul_new
        T_l_updates[unitcell_coord(x, y)] = T_l_new
        C_dl_updates[unitcell_coord(x, y + 1)] = C_dl_new
    for site in A_sites:
        env_sites[site].C_ul = C_ul_updates[site]
        env_sites[site].T_l = T_l_updates[site]
        env_sites[site].C_dl = C_dl_updates[site]
    return env_sites


def rotated_sites(A_sites: TensorSites, env_sites: EnvSites, unitcell_coord: UnitCellCoord) -> tuple[TensorSites, EnvSites, Callable[[int, int], Site]]:
    A_sites_rotated: TensorSites = {}
    env_sites_rotated: EnvSites = {}

    for (x, y), A in A_sites.items():
        new_site = (y, -x)
        A_sites_rotated[new_site] = A.permute(3, 0, 1, 2)

        C_ur, C_ul, C_dl, C_dr, T_r, T_u, T_l, T_d = env_sites[(x, y)]
        env_sites_rotated[new_site] = Environment(C_dr, C_ur, C_ul, C_dl, T_d, T_r, T_u, T_l)

    unitcell_coord_rotated = rotated_unitcell_coord(unitcell_coord)
    return A_sites_rotated, env_sites_rotated, unitcell_coord_rotated


def all_moves_sites(A_sites: TensorSites, env_sites: EnvSites, unitcell_coord: UnitCellCoord, config: Config) -> EnvSites:
    for _ in range(4):
        env_sites_new = left_move_sites(A_sites, env_sites, unitcell_coord, config)
        A_sites, env_sites, unitcell_coord = rotated_sites(A_sites, env_sites_new, unitcell_coord)
    return env_sites



# "Legacy" code from the single site CTMRG

def left_move(A: Tensor, env: Environment, config: Config) -> Environment:
    C_ur, C_ul, C_dl, C_dr, T_r, T_u, T_l, T_d = env
    Q_ul = einsum("IA,AJB,iaI,bJaj->ijBb", C_ul, T_u, T_l, A)
    Q_ur = einsum("BKC,CL,cKbk,Lcl->Bbkl", T_u, C_ur, A, T_r)
    Q_dl = einsum("AI,BJA,Iai,bjaJ->ijBb", C_dl, T_d, T_l, A)
    Q_dr = einsum("CKB,LC,ckbK,lcL->Bbkl", T_d, C_dr, A, T_r)
    R = einsum("ijBb,Bbkl->ijkl", Q_ul, Q_ur)
    Rbar = einsum("ijBb,Bbkl->ijkl", Q_dl, Q_dr)
    # # Works but not as efficient as the previous method using "quarters"
    # R    = einsum("IA,AJB,BKC,CL,iaI,bJaj,cKbk,Lcl->ijkl", C_ul, T_u, T_u, C_ur, T_l, A, A, T_r)
    # Rbar = einsum("AI,BJA,CKB,LC,Iai,bjaJ,ckbK,lcL->ijkl", C_dl, T_d, T_d, C_dr, T_l, A, A, T_r)
    RRbar    = einsum("abij,abkl->ijkl",R,Rbar)
    dim_i, dim_j, dim_k, dim_l = RRbar.shape
    r_left   = dim_i * dim_j
    r_right  = dim_k * dim_l
    M = RRbar.reshape(r_left, r_right)
    scale = M.abs().max()
    if scale.item() == 0:
        raise ValueError("The projector matrix is identically zero.")
    U, S, Vh = torch.linalg.svd(M / scale, full_matrices=False)
    S = S * scale
    V = Vh.mT
    rtol = max(M.shape) * torch.finfo(S.dtype).eps
    rank = (S > rtol * S[0]).sum().item()
    if rank == 0:
        raise ValueError("M has no numerically significant singular values.")
    p = min(config.chi, rank)
    # xi = S.new_zeros(()) if rank == 1 else 1 / torch.log(S[0] / S[1])
    U, S, V = U[:, :p], S[:p], V[:, :p]
    U = U.reshape(dim_i, dim_j, p)
    V = V.reshape(dim_k, dim_l, p)
    Srsqrt = S.rsqrt()
    P    = einsum("ijkl,klm->ijm", R   , U) * Srsqrt
    Pbar = einsum("ijkl,klm->ijm", Rbar, V) * Srsqrt
    C_ul_new = einsum("ijk,il,ljm->km",Pbar,C_ul,T_u)
    T_l_new = einsum("ijkl,nkm,mjo,nlp->pio", A, T_l, P, Pbar)
    C_dl_new = einsum("ijk,li,mjl->mk", P, C_dl, T_d)
    C_ul_new /= norm(C_ul_new)
    T_l_new /= norm(T_l_new)
    C_dl_new /= norm(C_dl_new)
    env_new = Environment(C_ur, C_ul_new, C_dl_new, C_dr, T_r, T_u, T_l_new, T_d, config)
    return env_new


def all_moves(A: Tensor, env: Environment, config: Config) -> Environment:
    for _ in range(4):
        env = left_move(A, env, config)
        C_ur, C_ul, C_dl, C_dr, T_r, T_u, T_l, T_d = env
        env = Environment(C_dr, C_ur, C_ul, C_dl, T_d, T_r, T_u, T_l)
        A = A.permute(3, 0, 1, 2)
    return env
