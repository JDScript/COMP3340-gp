import torch
import torch.nn as nn


class ViTNeck(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
