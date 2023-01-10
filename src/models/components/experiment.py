from typing import Callable, Dict, Iterable
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        input: torch.Tensor,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        self.in_channels = input.size(1)
        self.out_channels = out_channels
        self.window_size = input.size(2)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.dilation = dilation

        self.conv = nn.Conv1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

         # Register a hook for each layer
        for name, layer in self.model.named_children():
            layer.__name__ = name
            layer.register_forward_hook(
                lambda layer, _, output: print(f"{layer.__name__}: {output.shape}")
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)