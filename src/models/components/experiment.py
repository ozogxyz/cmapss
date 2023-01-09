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
            # padding_mode="reflect",
            # dilation=2,
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
    def __init__(self, conv_out: int, lstm_hidden: int):
        super(ExpNet, self).__init__()

        self.feature_extractor = nn.Sequential(
            ConvBlock(14, conv_out, kernel_size=5, stride=1, padding=1),
            ConvBlock(conv_out, 2 * conv_out, kernel_size=3, stride=2, padding=1),
        )

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # self.lstm = nn.LSTM(320, lstm_hidden, 2, batch_first=True, dropout=0.2)
        self.recurrent = nn.LSTM(448, lstm_hidden, 2, batch_first=True, dropout=0.2)
        self.recurrent_activation = nn.Tanh()

        self.dense = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Linear(16, 8),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            nn.Linear(8, 1),
        )

    def forward(self, x: torch.Tensor):
        x = self.feature_extractor(x)
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        # x = self.lstm(x)[0]
        # x = x.permute(0, 2, 1)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x, _ = self.recurrent(x)
        # print(x.shape)
        x = self.recurrent_activation(x)
        # print(x.shape)
        x = self.dense(x)
        print(f'##$^#$^#$%#$%^#$%^#$%^#$%^ Prediction: {x.flatten()} #$%^#$%^#$%^#$%^#$%^#$%^#$%^')
        return x.reshape(-1)
