import torch
import torch.nn as nn
import open_clip


class CLIPHead(nn.Module):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__()
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor):
        similarity = self.softmax(x)
        return similarity
