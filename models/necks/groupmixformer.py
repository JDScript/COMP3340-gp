import torch
import torch.nn as nn


class GroupMixFormerNeck(nn.Module):
    def __init__(self, last_embedding_dim=320, **kwargs) -> None:
        super().__init__()
        self.last_embedding_dim = last_embedding_dim
        self.norm = nn.SyncBatchNorm(self.last_embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x[-1])
        x = x.mean(dim=(2, 3))
        return x
