import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18(nn.Module):
    def __init__(
        self,
        weights: ResNet18_Weights | str = ResNet18_Weights.IMAGENET1K_V1,
        frozen: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(weights, str):
            weights = ResNet18_Weights[weights]
        # Load and freeze ResNet
        self.resnet = resnet18(weights=weights)
        if frozen:
            for parameter in self.resnet.parameters():
                parameter.requires_grad = False
        # replace last full connected layer to specific dimensions
        self.resnet.fc = nn.Sequential()  # type: ignore

    def forward(self, x: torch.Tensor):
        return self.resnet.forward(x)
