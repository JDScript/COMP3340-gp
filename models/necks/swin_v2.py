import torch
import torch.nn as nn


class Swin_V2_Neck(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
