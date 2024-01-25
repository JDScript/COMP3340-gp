import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(
        self,
        in_features: int = 2048,
        out_features: int = 1000,
        **kwargs,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        # self.relu1 = nn.ReLU()
        # self.dropout1 = nn.Dropout(0.1)
        # self.fc2 = nn.Linear(1024, 17)
        # self.relu2 = nn.ReLU()
        self.softmax = nn.LogSoftmax(1)

    def forward(self, x: torch.Tensor):
        out = self.fc1(x)
        # out = self.relu1(out)
        # out = self.dropout1(out)
        # out = self.fc2(out)
        # out = self.relu2(out)
        out = self.softmax(out)
        return out
