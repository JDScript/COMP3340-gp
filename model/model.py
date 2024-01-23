import torch
import torch.nn as nn
from config import Config
from utils import retrieve_class_from_string


class Model(nn.Module):
    def __init__(self, backbone: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x: torch.Tensor):
        out = self.backbone(x)
        out = self.classifier(out)
        return out


def instantiate_model(config: Config):
    if config.ckpt is not None:
        model = torch.load(config.ckpt)
        return model

    backbone = retrieve_class_from_string(config.model.backbone.target)(
        **config.model.backbone.params
    )
    classifer = retrieve_class_from_string(config.model.classifier.target)(
        **config.model.classifier.params
    )

    return Model(backbone, classifer)
