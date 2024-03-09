import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class VGG19(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        weights = None
        if pretrained:
            weights = VGG19_Weights.IMAGENET1K_V1
        self.backbone = vgg19(weights=weights).features

    def forward(self, x):
        return self.backbone(x)
