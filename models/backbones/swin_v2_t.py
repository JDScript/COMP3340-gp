import torch.nn as nn
from torchvision.models import swin_v2_t, Swin_V2_T_Weights


class Swin_V2_T(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        weights = None
        if pretrained:
            weights = Swin_V2_T_Weights.IMAGENET1K_V1
        self.backbone = swin_v2_t(weights=weights).features

    def forward(self, x):
        return self.backbone(x)
