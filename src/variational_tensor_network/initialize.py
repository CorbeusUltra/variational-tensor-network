"""Initialize the bulk state and its environment."""

from torch import Tensor, einsum, ones, zeros

from .objects import Config, DEFAULT_config, Environment

DEFAULT_delta = ones(DEFAULT_config.D2)

def initial_ket(config: Config = DEFAULT_config) -> Tensor:
    D = config.D
    d = config.d
    a = zeros(D,D,D,D,d)
    a[...,1] = 1.0
    return a

DEFAULT_a = initial_ket()

def initial_braket(a: Tensor = DEFAULT_a) -> Tensor:
    D2 = a.shape[0]**2
    A = einsum("ijkls,IJKLs->iIjJkKlL", a, a)
    A = A.reshape((D2,)*4)
    return A

DEFAULT_A = initial_braket()

def initial_env(delta: Tensor = DEFAULT_delta, A: Tensor = DEFAULT_A, config: Config = DEFAULT_config) -> Environment:
    T = einsum("i,ijkl->jkl",delta,A)
    C = einsum("i,j,ijkl->kl", delta, delta, A)
    chi = config.chi
    D2 = config.D2
    env = Environment(C,C,C,C,T,T,T,T,chi=chi,D2=D2)
    return env

DEFAULT_env = initial_env()

def main():
    print(DEFAULT_env.C_ul)
    print(DEFAULT_env.T_l)

if __name__ == "__main__":
    main()
