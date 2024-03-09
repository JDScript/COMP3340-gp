import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights


class AlexNet(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        weights = None
        if pretrained:
            weights = AlexNet_Weights.IMAGENET1K_V1
        self.backbone = alexnet(weights=weights).features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
