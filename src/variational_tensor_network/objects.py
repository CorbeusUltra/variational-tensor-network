"""Define tensor-network configuration and data structures."""

from collections.abc import Iterator
from dataclasses import InitVar, dataclass
from typing import Literal

import torch
from torch import Tensor

from .tilings import Site

type Direction = Literal["r", "d"]
type BondWeights = dict[tuple[Site, Direction], Tensor]
type EnvSites = dict[Site, Environment]


torch.set_default_dtype(torch.float64)


@dataclass
class Config:
    d: int = 2
    D: int = 3
    chi: int = 5
    def __post_init__(self) -> None:
        for name in ("d", "D", "chi"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be a Python integer.")
        if self.d <= 0 or self.D <= 0 or self.chi <= 0:
            raise ValueError("d, D and chi must be strictly positive.")
    @property
    def D2(self) -> int:
        return self.D**2


@dataclass
class Environment:
    C_ur: Tensor; C_ul: Tensor; C_dl: Tensor; C_dr: Tensor; T_r: Tensor; T_u: Tensor; T_l: Tensor; T_d: Tensor
    config: InitVar[Config | None] = None
    def __post_init__(self, config: Config | None) -> None:
        corners = ("C_ur", "C_ul", "C_dl", "C_dr")
        edges = ("T_r", "T_u", "T_l", "T_d")
        for names, ndim in ((corners, 2), (edges, 3)):
            for name in names:
                tensor = getattr(self, name)
                if tensor.ndim != ndim:
                    raise ValueError(f"{name} must have {ndim} axes.")
                if any(size <= 0 for size in tensor.shape):
                    raise ValueError(f"{name} must have strictly positive dimensions.")
        D2 = config.D2 if config is not None else self.T_r.shape[1]
        for name in edges:
            if getattr(self, name).shape[1] != D2:
                raise ValueError(f"{name}.shape[1] must equal D2 = {D2}.")
    def __iter__(self) -> Iterator[Tensor]:
        return iter((self.C_ur, self.C_ul, self.C_dl, self.C_dr, self.T_r, self.T_u, self.T_l, self.T_d))


@dataclass(frozen=True)
class SiteWeights:
    """Reference the bond weights around one square-lattice site."""
    r: Tensor
    u: Tensor
    l: Tensor
    d: Tensor
    config: InitVar[Config | None] = None
    def __post_init__(self, config: Config | None) -> None:
        directions = ("r", "u", "l", "d")
        for direction in directions:
            weight = getattr(self, direction)
            if weight.ndim != 1:
                raise ValueError(f"{direction} must be a vector, got shape {tuple(weight.shape)}.")
        D = config.D if config else self.r.shape[0]
        if D <= 0:
            raise ValueError("Bond dimension must be strictly positive.")
        for direction in directions:
            weight = getattr(self, direction)
            if weight.shape[0] != D:
                raise ValueError(f"{direction} must have shape ({D},), got {tuple(weight.shape)}.")
