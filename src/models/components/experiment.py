import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, out_channels*2, kernel_size=3, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm1d(out_channels*2)
        self.pooling = nn.MaxPool1d(2, 2)

        self.rnn = nn.GRU(
            448,
            out_channels,
            batch_first=True,
            bidirectional=False,
            dropout=0.2,
        )
        self.rnn_bn = nn.BatchNorm1d(out_channels * 2)
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(out_channels, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.relu(x)

        x = self.pooling(x)
        x = torch.flatten(x, 1)

        x, _ = self.rnn(x)
        x = self.tanh(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x.reshape(-1)
