"""Make one CTMRG move, here on the left of the environment"""

from torch import Tensor, einsum
from torch.linalg import svd

from .initialize import DEFAULT_A, DEFAULT_env
from .objects import Config, DEFAULT_config, Environment


def left_move(A: Tensor = DEFAULT_A, env: Environment = DEFAULT_env, config: Config = DEFAULT_config) -> Environment:
    C_ul, C_ur, C_dl, C_dr, T_u, T_r, T_d, T_l = env

    R        = einsum("IA,AJB,BKC,CL,iaI,bJaj,cKbk,Lcl->ijkl", C_ul, T_u, T_u, C_ur, T_l, A, A, T_r)
    Rbar     = einsum("AI,BJA,CKB,LC,Iai,bjaJ,ckbK,lcL->ijkl", C_dl, T_d, T_d, C_dr, T_l, A, A, T_r)
    RRbar    = einsum("abij,abkl->ijkl",R,Rbar)
    r_left   = RRbar.shape[0] * RRbar.shape[1]
    r_right  = RRbar.shape[2] * RRbar.shape[3]
    M = RRbar.reshape(r_left, r_right)
    U, S, Vh = svd(M, full_matrices=False)

    print("R shape:", R.shape)
    print("Rbar shape:", Rbar.shape)
    print("RRbar shape:", RRbar.shape)
    print("M shape:", M.shape)

    print("U shape:", U.shape)
    print("S shape:", S.shape)
    print("Vh shape:", Vh.shape)

    U = U.reshape(*RRbar.shape[:2], U.shape[1])
    Vh = Vh.reshape(Vh.shape[0], *RRbar.shape[2:])
    print("U shape:", U.shape)
    print("S shape:", S.shape)
    print("Vh shape:", Vh.shape)



if __name__ == "__main__":
    left_move()
