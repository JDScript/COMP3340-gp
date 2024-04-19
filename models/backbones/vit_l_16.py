import torch.nn as nn
from .vit import vit_l_16, ViT_L_16_Weights


class ViT_L_16(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        weights = None
        if pretrained:
            weights = ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        self.backbone = vit_l_16(weights=weights)
        self.backbone.heads = nn.Sequential()

    def forward(self, x, return_weights=False):
        out, weights = self.backbone(x)

        if return_weights:
            return out, weights

        return out
