import torch.nn as nn
from .vit import vit_h_14, ViT_H_14_Weights


class ViT_H_14(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        weights = None
        if pretrained:
            weights = ViT_H_14_Weights.IMAGENET1K_SWAG_LINEAR_V1
        self.backbone = vit_h_14(weights=weights)
        self.backbone.heads = nn.Sequential()

    def forward(self, x, return_weights=False):
        out, weights = self.backbone(x)

        if return_weights:
            return out, weights

        return out
