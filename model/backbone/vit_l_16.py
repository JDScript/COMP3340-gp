import torch
import torch.nn as nn
from torchvision.models import vit_l_16, ViT_L_16_Weights


class ViT_L_16(nn.Module):
    def __init__(
        self,
        weights: ViT_L_16_Weights | str | None = None,
        frozen: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(weights, str):
            weights = ViT_L_16_Weights[weights]
        self.vit = vit_l_16(weights=weights)
        self.vit.heads = nn.Sequential()

        if frozen:
            for parameter in self.vit.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor):
        return self.vit(x)
