import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50(nn.Module):
    def __init__(
        self,
        weights: ResNet50_Weights | str | None = None,
        frozen: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(weights, str):
            weights = ResNet50_Weights[weights]
        # Load and freeze ResNet
        self.resnet = resnet50(weights=weights)
        if frozen:
            for parameter in self.resnet.parameters():
                parameter.requires_grad = False
        # replace last full connected layer to specific dimensions
        self.resnet.fc = nn.Sequential()  # type: ignore

    def forward(self, x: torch.Tensor):
        return self.resnet.forward(x)
