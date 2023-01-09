import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ExpNet(nn.Module):
    def __init__(
        self,
        conv_out: int,
        lstm_hidden: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ):
        super(ExpNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            ConvBlock(14, conv_out, kernel_size, stride, padding),
            ConvBlock(conv_out, 2 * conv_out, kernel_size-2, stride, padding),
        )

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.recurrent = nn.LSTM(256, lstm_hidden, 2, batch_first=True, dropout=0.2)
        self.recurrent_activation = nn.Tanh()

        self.dense = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x, _ = self.recurrent(x)
        x = self.recurrent_activation(x)
        x = self.dense(x)
        return x.reshape(-1)
