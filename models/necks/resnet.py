import torch
import torch.nn as nn


class ResNetNeck(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool(x)
        return x.flatten(1)
