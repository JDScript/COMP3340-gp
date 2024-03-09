import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MobileNet_V3_Small(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        weights = None
        if pretrained:
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        self.backbone = mobilenet_v3_small(weights=weights).features

    def forward(self, x):
        return self.backbone(x)
