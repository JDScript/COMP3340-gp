import torch
import torch.nn as nn
from torchvision.models import resnet152, ResNet152_Weights


class ResNet152(nn.Module):
    def __init__(
        self,
        weights: ResNet152_Weights | str = ResNet152_Weights.IMAGENET1K_V1,
        frozen: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(weights, str):
            weights = ResNet152_Weights[weights]
        # Load and freeze ResNet
        self.resnet = resnet152(weights=weights)
        if frozen:
            for parameter in self.resnet.parameters():
                parameter.requires_grad = False
        # replace last full connected layer
        self.resnet.fc = nn.Sequential()  # type: ignore

    def forward(self, x: torch.Tensor):
        return self.resnet.forward(x)


if __name__ == "__main__":
    resnet = ResNet152()
