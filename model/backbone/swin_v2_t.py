import torch
import torch.nn as nn
from torchvision.models import swin_v2_t, Swin_V2_T_Weights


class Swin_V2_T(nn.Module):
    def __init__(
        self,
        weights: Swin_V2_T_Weights | str | None = None,
        frozen: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if isinstance(weights, str):
            weights = Swin_V2_T_Weights[weights]
        # Load and freeze Swin
        self.swin = swin_v2_t(weights=weights).features
        if frozen:
            for parameter in self.swin.parameters():
                parameter.requires_grad = False

    def forward(self, x: torch.Tensor):
        return self.swin.forward(x)
