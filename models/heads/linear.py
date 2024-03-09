import torch
import torch.nn as nn


class LinearClsHead(nn.Module):
    def __init__(
        self,
        in_features: int = 2048,
        out_features: int = 1000,
        **kwargs,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.softmax = nn.LogSoftmax(1)

    def forward(self, x: torch.Tensor):
        out = self.fc(x)
        out = self.softmax(out)
        return out
