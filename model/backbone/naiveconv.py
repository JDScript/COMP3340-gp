import torch
import torch.nn as nn
from typing import cast


class NaiveConv(nn.Module):
    def __init__(
        self,
        layers_configurations=[
            64,
            "M",
            128,
            "M",
            256,
            256,
            "M",
            512,
            512,
            "M",
            512,
            512,
            "M",
        ],
        kernel_size=3,
        batch_norm=True,
        **kwargs,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        self.layers_configurations = layers_configurations

        in_channels = 3
        for conf in layers_configurations:
            if isinstance(conf, int):
                out_channels = cast(int, conf)
                conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=1,
                )
                if batch_norm:
                    layers += [
                        conv,
                        nn.BatchNorm2d(out_channels),
                        nn.SiLU(inplace=True),
                    ]
                else:
                    layers += [
                        conv,
                        nn.SiLU(inplace=True),
                    ]
                in_channels = out_channels

            elif isinstance(conf, str) and conf == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
