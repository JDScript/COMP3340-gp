import torch.nn as nn
import open_clip


class CLIP(nn.Module):
    def __init__(
        self,
        weights="laion2b_s34b_b79k",
        **kwargs,
    ) -> None:
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=weights
        )
        self.clip = model

    def forward(self, x):
        return  self.clip.encode_image(x)