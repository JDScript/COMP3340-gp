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
        swin = swin_v2_t(weights=weights)
        self.backbone = swin.features
        self.neck = nn.Sequential(
            swin.norm,
            swin.permute,
            swin.avgpool,
            swin.flatten,
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return x
