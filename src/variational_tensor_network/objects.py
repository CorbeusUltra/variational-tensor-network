"""Define the objects exptected to be used later in make_move.py file"""

from collections.abc import Iterator
from dataclasses import InitVar, dataclass

import torch
from torch import Tensor

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
        if self.d != 2:
            raise ValueError("d must equal 2.")
        if self.D <= 0 or self.chi <= 0:
            raise ValueError("D and chi must be strictly positive.")

    @property
    def D2(self) -> int:
        return self.D**2

DEFAULT_config = Config()

@dataclass
class Environment:
    C_ur: Tensor
    C_ul: Tensor
    C_dr: Tensor
    C_dl: Tensor

    T_r: Tensor
    T_u: Tensor
    T_l: Tensor
    T_d: Tensor

    config: InitVar[Config | None] = None

    def __post_init__(self, config: Config | None) -> None:
        corners = ("C_ul", "C_ur", "C_dl", "C_dr")
        edges = ("T_u", "T_r", "T_d", "T_l")

        for names, ndim in ((corners, 2), (edges, 3)):
            for name in names:
                tensor = getattr(self, name)
                if tensor.ndim != ndim:
                    raise ValueError(f"{name} must have {ndim} axes.")
                if any(size <= 0 for size in tensor.shape):
                    raise ValueError(f"{name} must have strictly positive dimensions.")

        D2 = config.D2 if config is not None else self.T_u.shape[1]
        for name in edges:
            if getattr(self, name).shape[1] != D2:
                raise ValueError(f"{name}.shape[1] must equal D2 = {D2}.")

        connections=(("C_ul",0,"T_l",2),("C_ul",1,"T_u",0),("C_ur",0,"T_u",2),("C_ur", 1,"T_r", 0),("C_dr",0,"T_r",2),("C_dr",1,"T_d",0),("C_dl",0,"T_d",2),("C_dl",1,"T_l",0))

        for corner, c_axis, edge, t_axis in connections:
            c_size = getattr(self, corner).shape[c_axis]
            t_size = getattr(self, edge).shape[t_axis]
            if c_size != t_size:
                raise ValueError(
                    f"{corner}.shape[{c_axis}] = {c_size} does not match "
                    f"{edge}.shape[{t_axis}] = {t_size}."
                )
    
    def __iter__(self) -> Iterator[Tensor]:
        return iter((self.C_ul, self.C_ur, self.C_dl, self.C_dr, self.T_u, self.T_r, self.T_d, self.T_l))



# May be useful later

# @dataclass
# class aTensor:
#     tensor: Tensor
#     D: int
#     d: int

#     def __post_init__(self) -> None:
#         if tuple(self.tensor.shape) != (self.D,) * 4 + (self.d,):
#             raise ValueError("a tensor must have shape (D, D, D, D, d).")

# @dataclass
# class ATensor:
#     tensor: Tensor
#     D2: int

#     def __post_init__(self) -> None:
#         if tuple(self.tensor.shape) != (self.D2,) * 4:
#             raise ValueError("A tensor must have shape (D2, D2, D2, D2).")


# @dataclass
# class CTensor:
#     tensor: Tensor
#     chi: int

#     def __post_init__(self) -> None:
#         if tuple(self.tensor.shape) != (self.chi, self.chi):
#             raise ValueError("C tensor must have shape (chi, chi).")


# @dataclass
# class TTensor:
#     tensor: Tensor
#     D2: int
#     chi: int

#     def __post_init__(self) -> None:
#         if tuple(self.tensor.shape) != (self.chi, self.D2, self.chi):
#             raise ValueError("T tensor must have shape (chi, D2, chi).")
