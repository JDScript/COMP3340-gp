import torch.nn as nn
from torchvision.models import resnext101_32x8d, ResNeXt101_32X8D_Weights


class ResNeXt101_32X8D(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        weights = None
        if pretrained:
            weights = ResNeXt101_32X8D_Weights.IMAGENET1K_V2
        model = resnext101_32x8d(weights=weights)
        self.backbone = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
        )

    def forward(self, x):
        return self.backbone(x)
