from dataclasses import dataclass

import torch

D: int = 8
chi: int = 10


@dataclass
class BulkTensor:
    data: torch.Tensor
    D: int

    def __post_init__(self) -> None:
        if self.D < 1 or tuple(self.data.shape) != (self.D,) * 4:
            raise ValueError("Bulk tensor must have shape (D, D, D, D).")


@dataclass
class CornerTensor:
    data: torch.Tensor
    chi: int

    def __post_init__(self) -> None:
        if self.chi < 1 or tuple(self.data.shape) != (self.chi, self.chi):
            raise ValueError("Corner tensor must have shape (chi, chi).")


@dataclass
class EdgeTensor:
    data: torch.Tensor
    D: int
    chi: int

    def __post_init__(self) -> None:
        if self.D < 1 or self.chi < 1:
            raise ValueError("Dimensions must be positive.")
        if tuple(self.data.shape) != (self.chi, self.D, self.chi):
            raise ValueError("Edge tensor must have shape (chi, D, chi).")
