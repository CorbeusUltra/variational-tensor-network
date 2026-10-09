"""Initialize the bulk state and its environment."""

import torch
from torch import Tensor, einsum, norm

from .objects import Config, DEFAULT_config, Environment

torch.set_default_dtype(torch.float64)

DEFAULT_delta = torch.ones(DEFAULT_config.D2)

def initial_ket(config: Config = DEFAULT_config) -> Tensor:
    D = config.D
    d = config.d
    # a = torch.zeros(D,D,D,D,d)
    # a[...,1] = 1
    a = torch.rand(D,D,D,D,d)
    return a/norm(a)

DEFAULT_a = initial_ket()

def initial_braket(a: Tensor = DEFAULT_a, config: Config = DEFAULT_config) -> Tensor:
    D2 = config.D2
    A = einsum("ijkls,IJKLs->iIjJkKlL", a, a)
    A = A.reshape((D2,)*4)
    return A

DEFAULT_A = initial_braket()

def initial_env(delta: Tensor = DEFAULT_delta, A: Tensor = DEFAULT_A, config: Config = DEFAULT_config) -> Environment:
    C0 = einsum("ijkl,k,l->ij", A, delta, delta)
    T0 = einsum("k,ijkl->lij", delta, A)

    D2 = config.D2
    chi = config.chi

    if chi <= D2:
        C = C0[:chi, :chi].clone()
        T = T0[:chi, :, :chi].clone()
    else:
        C = torch.zeros(chi, chi)
        T = torch.zeros(chi, D2, chi)

        C[:D2, :D2] = C0
        T[:D2, :, :D2] = T0

    C /= norm(C)
    T /= norm(T)
    
    env = Environment(C,C,C,C,T,T,T,T,config)

    return env

DEFAULT_env = initial_env()


def main():
    print(f"C_ul.shape : {DEFAULT_env.C_ul.shape}")
    print(f"T_l.shape : {DEFAULT_env.T_l.shape}")


if __name__ == "__main__":
    main()
