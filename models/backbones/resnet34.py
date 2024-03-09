import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights


class ResNet34(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        weights = None
        if pretrained:
            weights = ResNet34_Weights.IMAGENET1K_V1
        model = resnet34(weights=weights)
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
