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
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
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
    def __init__(self, conv_out: int, lstm_hidden: int):
        super(ExpNet, self).__init__()

        self.conv1 = ConvBlock(14, conv_out, kernel_size=5, stride=1, padding=1)
        self.conv2 = ConvBlock(
            conv_out, conv_out * 2, kernel_size=3, stride=2, padding=1
        )

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        lstm_size = self._get_lstm_size(kernel_size=3, stride=2)
        self.lstm1 = nn.LSTM(384, lstm_hidden, 2, batch_first=True, dropout=0.2)

        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.LeakyReLU()

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)

        x = x.view(x.size(0), -1)
        # print(x.shape)
        x, _ = self.lstm1(x)
        # print(x.shape)
        x = self.tanh(x)
        # x = x[:,:,-1]
        # print(x[:,:,-1].shape)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x.reshape(-1)

    def _get_lstm_size(self, kernel_size: int, stride: int):
        conv1_out_dim: int = (30 - kernel_size) // stride + 1
        conv2_out_dim = (conv1_out_dim - kernel_size) // stride + 1
        maxpool_out_dim: int = (conv2_out_dim - 2) // 2 + 1
        return maxpool_out_dim
