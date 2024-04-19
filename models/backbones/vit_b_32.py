import torch.nn as nn
from .vit import vit_b_32, ViT_B_32_Weights


class ViT_B_32(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        weights = None
        if pretrained:
            weights = ViT_B_32_Weights.IMAGENET1K_V1
        self.backbone = vit_b_32(weights=weights)
        self.backbone.heads = nn.Sequential()

    def forward(self, x, return_weights=False):
        out, weights = self.backbone(x)

        if return_weights:
            return out, weights

        return out
