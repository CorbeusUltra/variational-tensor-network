"""Define the objects exptected to be used later in make_move.py file"""

from dataclasses import InitVar, dataclass

import torch


@dataclass
class Config:
    d: int = 2
    D: int = 2
    chi: int = 4

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


@dataclass
class Environment:
    C_ul: torch.Tensor
    C_ur: torch.Tensor
    C_dl: torch.Tensor
    C_dr: torch.Tensor

    T_u: torch.Tensor
    T_r: torch.Tensor
    T_d: torch.Tensor
    T_l: torch.Tensor

    chi: InitVar[int]
    D2: InitVar[int]

    def __post_init__(self, chi: int, D2: int) -> None:
        for name in ("C_ul", "C_ur", "C_dl", "C_dr"):
            if tuple(getattr(self, name).shape) != (chi, chi):
                raise ValueError(f"{name} must have shape (χ, χ) = ({chi}, {chi}).")

        for name in ("T_u", "T_r", "T_d", "T_l"):
            if tuple(getattr(self, name).shape) != (chi, D2, chi):
                raise ValueError(f"{name} must have shape (χ, D², χ) = ({chi}, {D2}, {chi}).")



# @dataclass
# class aTensor:
#     tensor: torch.Tensor
#     D: int
#     d: int

#     def __post_init__(self) -> None:
#         if tuple(self.tensor.shape) != (self.D,) * 4 + (self.d,):
#             raise ValueError("a tensor must have shape (D, D, D, D, d).")

# @dataclass
# class ATensor:
#     tensor: torch.Tensor
#     D2: int

#     def __post_init__(self) -> None:
#         if tuple(self.tensor.shape) != (self.D2,) * 4:
#             raise ValueError("A tensor must have shape (D2, D2, D2, D2).")


# @dataclass
# class CTensor:
#     tensor: torch.Tensor
#     chi: int

#     def __post_init__(self) -> None:
#         if tuple(self.tensor.shape) != (self.chi, self.chi):
#             raise ValueError("C tensor must have shape (chi, chi).")


# @dataclass
# class TTensor:
#     tensor: torch.Tensor
#     D2: int
#     chi: int

#     def __post_init__(self) -> None:
#         if tuple(self.tensor.shape) != (self.chi, self.D2, self.chi):
#             raise ValueError("T tensor must have shape (chi, D2, chi).")
