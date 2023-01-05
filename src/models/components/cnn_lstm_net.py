import torch
import torch.nn as nn
from torch.nn import Module, Sequential


class CNNLSTMNet(Module):
    def __init__(
        self,
        conv1_size: int = 14,
        conv2_size: int = 32,
        conv1_kernel: int = 5,
        conv1_stride: int = 2,
        conv2_kernel: int = 3,
        conv2_stride: int = 1,
    ):
        super().__init__()
        self.feature_extractor = Sequential(
            nn.Conv1d(
                in_channels=conv1_size,
                out_channels=conv2_size,
                kernel_size=conv1_kernel,
                stride=conv1_stride,
            ),
            nn.BatchNorm1d(conv2_size),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=conv2_size,
                out_channels=conv2_size,
                kernel_size=conv2_kernel,
                stride=conv2_stride,
            ),
            nn.BatchNorm1d(conv2_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),
        )

        self.lstm = nn.LSTM(
            input_size=conv2_size, hidden_size=128, num_layers=1, batch_first=True
        )

        self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x, _ = self.lstm(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
