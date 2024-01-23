import torch
import torch.nn as nn
from torchvision.models import alexnet, AlexNet_Weights


class AlexNet(nn.Module):
    def __init__(
        self,
        weights: AlexNet_Weights | str = AlexNet_Weights.IMAGENET1K_V1,
        frozen: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(weights, str):
            weights = AlexNet_Weights[weights]
        # Only use feature layer of alexnet
        self.alexnet = alexnet(weights=weights).features
        if frozen:
            for parameter in self.alexnet.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor):
        out = self.alexnet.forward(x)
        out = out.view(out.shape[0], -1)
        return out
