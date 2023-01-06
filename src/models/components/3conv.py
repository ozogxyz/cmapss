import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.bn(x)
        return x


class CNNLSTMNet(nn.Module):
    def __init__(self, conv_out: int, lstm_hidden: int):
        super(CNNLSTMNet, self).__init__()

        self.conv1 = ConvBlock(14, conv_out, kernel_size=5, stride=2)
        self.conv2 = ConvBlock(conv_out, conv_out * 2, kernel_size=3 ,stride=2)
        self.conv3 = ConvBlock(conv_out * 2, conv_out * 2, kernel_size=3 ,stride=2)
        self.conv4 = ConvBlock(conv_out * 2, conv_out * 4, kernel_size=3 ,stride=2)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm1 = nn.LSTM(conv_out * 4, lstm_hidden, 2, batch_first=True, dropout=0.2)

        self.fc1 = nn.Linear(lstm_hidden, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.pool(x)

        x = x.permute(0, 2, 1)

        x, _ = self.lstm1(x)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x.reshape(-1)
