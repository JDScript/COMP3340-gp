import torch.nn as nn
from torchvision.models import swin_v2_b, Swin_V2_B_Weights


class Swin_V2_B(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        weights = None
        if pretrained:
            weights = Swin_V2_B_Weights.IMAGENET1K_V1
        self.backbone = swin_v2_b(weights=weights).features

    def forward(self, x):
        return self.backbone(x)
