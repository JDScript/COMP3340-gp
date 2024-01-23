import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights


class VGG19(nn.Module):
    def __init__(
        self,
        weights: VGG19_Weights | str = VGG19_Weights.IMAGENET1K_V1,
        frozen: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(weights, str):
            weights = VGG19_Weights[weights]
        # Only use feature layer of alexnet
        self.vgg = vgg19(weights=weights).features
        if frozen:
            for parameter in self.vgg.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor):
        out = self.vgg.forward(x)
        out = out.view(out.shape[0], -1)
        return out
