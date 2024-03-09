import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights


class ViT_B_16(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        weights = None
        if pretrained:
            weights = ViT_B_16_Weights.IMAGENET1K_SWAG_LINEAR_V1
        self.backbone = vit_b_16(weights=weights)
        self.backbone.heads = nn.Sequential()

    def forward(self, x):
        return self.backbone(x)
