import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViT_B_16(nn.Module):
    def __init__(
        self,
        weights: ViT_B_16_Weights | str = ViT_B_16_Weights.IMAGENET1K_V1,
        frozen: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(weights, str):
            weights = ViT_B_16_Weights[weights]
        self.vit = vit_b_16(weights=weights)
        self.vit.heads = nn.Sequential()

        if frozen:
            for parameter in self.vit.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor):
        return self.vit(x)
