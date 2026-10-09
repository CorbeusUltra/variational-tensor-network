"""Make one CTMRG move, here on the left of the environment"""

import torch
from torch import Tensor, einsum

from .initialize import DEFAULT_A, DEFAULT_env
from .objects import Config, DEFAULT_config, Environment


def left_move(A: Tensor = DEFAULT_A, env: Environment = DEFAULT_env, config: Config = DEFAULT_config) -> Environment:
    C_ur, C_ul, C_dl, C_dr, T_r, T_u, T_l, T_d = env

    R        = einsum("IA,AJB,BKC,CL,iaI,bJaj,cKbk,Lcl->ijkl", C_ul, T_u, T_u, C_ur, T_l, A, A, T_r)
    Rbar     = einsum("AI,BJA,CKB,LC,Iai,bjaJ,ckbK,lcL->ijkl", C_dl, T_d, T_d, C_dr, T_l, A, A, T_r)
    RRbar    = einsum("abij,abkl->ijkl",R,Rbar)
    dim_i, dim_j, dim_k, dim_l = RRbar.shape
    r_left   = dim_i * dim_j
    r_right  = dim_k * dim_l
    M = RRbar.reshape(r_left, r_right)
    U, S, V = torch.svd(M)

    print("R shape:", R.shape)
    print("Rbar shape:", Rbar.shape)
    print("RRbar shape:", RRbar.shape)
    print("M shape:", M.shape)
    print("\n\nbefore truncation\n")
    print("U shape:", U.shape)
    print("V shape:", V.shape)
    print("S shape:", S.shape)
    print(f"S:{S}")
    rtol = max(M.shape) * torch.finfo(S.dtype).eps
    p = min(config.chi, (S > rtol * S[0]).sum().item()) # may be optimized with the 3 following lines

    print("\n\ntruncation infos\n")
    print(f"rtol:{rtol}")
    print(f"chi:{config.chi}")
    print(f"non-zeros count:{(S > rtol * S[0]).sum().item()}")

    U = U[:, :p]
    S = S[:p]
    V = V[:, :p]
    print("\n\nafter truncation\n")
    print("U shape:", U.shape)
    print("V shape:", V.shape)
    print("S shape:", S.shape)
    print(f"S:{S}")

    U = U.reshape(dim_i, dim_j, p)
    V = V.reshape(dim_k, dim_l, p)
    invsqrtS = torch.diag(S**(-.5))

    print("\n\nafter reshape\n")
    print("U shape:", U.shape)
    print("V shape:", V.shape)
    print("S shape:", S.shape)
    print(f"S:{S}")

    P    = einsum("ijkl,klm,ms->ijs", R   , U, invsqrtS)
    Pbar = einsum("ijkl,klm,ms->ijs", Rbar, V, invsqrtS)

    print("\n")
    print(f"P:{P.shape}")
    print(f"Pbar:{Pbar.shape}")
    print("\n")
    print(f"\n\n T_l:{T_l.shape}\n")

    C_ul_new = einsum("ijk,il,ljm->km",Pbar,C_ul,T_u)
    T_l_new = einsum("ijkl,nkm,mjo,nlp->pio", A, T_l, P, Pbar)
    C_dl_new = einsum("ijk,li,mjl->mk", P, C_dl, T_d)
    env_new = Environment(C_ur, C_ul_new, C_dl_new, C_dr, T_r, T_u, T_l_new, T_d, config)

    return env_new



if __name__ == "__main__":
    print(left_move())
