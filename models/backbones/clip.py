import torch.nn as nn
import open_clip
import torch


class CLIP(nn.Module):
    def __init__(
        self,
        model_name="ViT-B-32",
        weights="laion2b_s34b_b79k",
        labels: list[str] = [],
        **kwargs,
    ) -> None:
        super().__init__()
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=weights
        )
        self.clip = model
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.text_tokens = self.tokenizer(labels)

    def forward(self, x):
        # if self.text_tokens.device != x.device:
        #     self.text_tokens = self.text_tokens.to(x.device)

        image_features = self.clip.encode_image(x)  # type: ignore
        # text_features = self.clip.encode_text(self.text_tokens)  # type: ignore

        # similarity = image_features @ text_features.T
        return image_features
