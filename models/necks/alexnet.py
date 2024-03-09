import torch
import torch.nn as nn


class AlexNetNeck(nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.avg_pool(x)
        return x.flatten(1)
