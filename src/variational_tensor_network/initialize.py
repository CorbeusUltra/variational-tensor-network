"""Initialize the bulk states and their environments."""

from typing import Literal, get_args

import torch
from torch import Tensor, ones, rand
from torch.linalg import vector_norm

from .objects import BondWeights, Config, Environment
from .peps import TensorSites, double_layer
from .tilings import Site

torch.set_default_dtype(torch.float64)

type InitMethod = Literal["random", "ones"]
_InitMethodChoices = ", ".join(repr(name) for name in get_args(InitMethod.__value__))


def initial_a(config: Config, method: InitMethod = "random") -> Tensor:
    D, d = config.D, config.d
    if method == "random":
        a = rand(D,D,D,D,d)
    elif method == "ones":
        a = ones(D,D,D,D,d)
    else:
        raise ValueError(f"Unknown initialization method {method!r}. Expected one of {_InitMethodChoices}.")
    return a / vector_norm(a)


def initial_a_sites(config: Config, sites_coords: tuple[Site, ...], method: InitMethod = "random") -> TensorSites:
    return {site: initial_a(config, method=method) for site in sites_coords}


def initial_bond_weights(a_sites: TensorSites, method: InitMethod = "random") -> BondWeights:
    bond_weights = {}
    for site, a in a_sites.items():
        for direction, axis in (("r", 0), ("d", 3)):
            D = a.shape[axis]
            if method == "random":
                weight = rand(D)
            elif method == "ones":
                weight = ones(D)
            else:
                raise ValueError(f"Unknown initialization method {method!r}. Expected one of {_InitMethodChoices}.")
            bond_weights[(site, direction)] = weight / vector_norm(weight)
    return bond_weights


def initial_single_and_double_layer_sites(config: Config, sites_coords: tuple[Site, ...],  method: InitMethod = "random") -> tuple[TensorSites, TensorSites]:
    """Initialize site tensors and build their double layers."""
    a_sites = initial_a_sites(config, sites_coords, method=method)
    A_sites = {site: double_layer(a) for site, a in a_sites.items()}
    return a_sites, A_sites




def initial_boundary(config: Config, method: InitMethod = "ones") -> Tensor:
    """Build the initial weight vector for a fused virtual leg."""
    D2 = config.D2
    if method == "ones":
        return ones(D2)
    elif method == "random":
        return rand(D2)
    else:
        raise ValueError(f"Unknown initialization method {method!r}. Expected one of {_InitMethodChoices}.")


def initial_env(delta: Tensor, A: Tensor, config: Config, method: Literal["from_bulk", "random", "ones"] = "from_bulk") -> Environment:
    D2 = config.D2
    chi = config.chi
    if method == "from_bulk":
        C0 = torch.einsum("ijkl,k,l->ij", A, delta, delta)
        T0 = torch.einsum("k,ijkl->lij", delta, A)
        if chi <= D2:
            C = C0[:chi, :chi].clone()
            T = T0[:chi, :, :chi].clone()
        else:
            C = A.new_zeros((chi, chi))
            T = A.new_zeros((chi, D2, chi))
            C[:D2, :D2] = C0
            T[:D2, :, :D2] = T0
    elif method == "random":
        C = rand(chi,chi)
        T = rand(chi,D2,chi)
    elif method == "ones":
        C = ones(chi,chi)
        T = ones(chi,D2,chi)
    else:
        raise ValueError(f"Unknown initialization method {method!r}. Expected 'from_bulk', 'ones' or 'random'.")
    C_norm, T_norm = vector_norm(C), vector_norm(T)
    C /= C_norm
    T /= T_norm
    env = Environment(C,C,C,C,T,T,T,T,config)
    # env = Environment(C.clone(),C.clone(),C.clone(),C.clone(),T.clone(),T.clone(),T.clone(),T.clone(),config)
    # Becareful here to indeed replace C's and T's with new tensors and not change them in place. In case of doubt replace with .clone() version.
    return env


def initial_env_sites(A_sites: TensorSites, delta: Tensor, config: Config, method: Literal["from_bulk", "random", "ones"] = "from_bulk") -> dict[Site, Environment]:
    env_sites: dict[Site, Environment] = {}
    for site, A in A_sites.items():
        env_sites[site] = initial_env(delta, A, config, method=method)
    return env_sites
