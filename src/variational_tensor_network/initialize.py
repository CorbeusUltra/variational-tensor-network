"""Initialize the bulk state and its environment."""

import torch

from .objects import Config

config = Config()

print(config)

def initial_state(config):
    D = config.D
    d = config.d
    a = torch.zeros(D,D,D,D,d)
    a[...,1] = 1.0
    return a

def build_double_layer(a):
    D2 = a.shape[0]**2
    A = torch.einsum("ijkls,IJKLs->iIjJkKlL", a, a)
    A = A.reshape((D2,)*4)
    return A

def main():
    a = initial_state(config)
    A = build_double_layer(a)
    print(a)
    print(A)

if __name__ == "__main__":
    main()
