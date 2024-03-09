import torch.nn as nn
from torchvision.models import resnext50_32x4d, ResNeXt50_32X4D_Weights


class ResNet50_32X4D(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        weights = None
        if pretrained:
            weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V2
        model = resnext50_32x4d(weights=weights)
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
