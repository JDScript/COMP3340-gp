import torch
import torch.nn as nn


class LinearClsHead(nn.Module):
    def __init__(
        self,
        in_features: int = 2048,
        out_features: int = 1000,
        **kwargs,
    ) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.softmax = nn.LogSoftmax(1)

    def forward(self, x: torch.Tensor):
        out = self.fc(x)
        out = self.softmax(out)
        return out


class MultiLinearClsHead(nn.Module):
    def __init__(
        self,
        in_features: int = 2048,
        out_features: int = 1000,
        embedding_paths: list[int] = [],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers = []
        last_in = in_features
        for embedding in embedding_paths:
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(last_in, embedding))
            layers.append(nn.ReLU(inplace=True))
            last_in = embedding

        layers.append(nn.Linear(last_in, out_features))
        self.classifier = nn.Sequential(*layers)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x: torch.Tensor):
        out = self.classifier(x)
        out = self.softmax(out)
        return out
