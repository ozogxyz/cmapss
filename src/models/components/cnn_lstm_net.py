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
        # x = self.relu(x)
        # x = self.pool(x)
        # x = self.dropout(x)
        return x


# class CNNLSTMNet(nn.Module):
#     def __init__(self, conv_out: int, lstm_hidden: int):
#         super(CNNLSTMNet, self).__init__()

#         self.conv1 = nn.Conv1d(14, conv_out, kernel_size=5, stride=1, padding=1)
#         self.conv2 = nn.Conv1d(conv_out, 64, kernel_size=3, stride=2, padding=1)
#         self.conv3 = nn.Conv1d(64, 128, kernel_size=7, stride=3, padding=2)

#         self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

#         self.lstm1 = nn.LSTM(256, 64, 2, batch_first=True, dropout=0.2)

#         self.fc1 = nn.Linear(64, 32)
#         self.fc2 = nn.Linear(32, 16)
#         self.fc3 = nn.Linear(16, 8)
#         self.fc4 = nn.Linear(8, 1)

#         self.relu = nn.ReLU()
#         self.tanh = nn.Tanh()
#         self.dropout = nn.Dropout(p=0.2)
#         self.bn1 = nn.BatchNorm1d(32)
#         self.bn2 = nn.BatchNorm1d(64)
#         self.bn3 = nn.BatchNorm1d(128)

#     def forward(self, x: torch.Tensor):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.dropout(x)

#         x = self.conv2(x)
#         x = self.bn2(x)
#         x = self.relu(x)
#         x = self.dropout(x)

#         x = self.conv3(x)
#         x = self.bn3(x)
#         x = self.relu(x)
#         x = self.dropout(x)

#         x = self.maxpool(x)
#         x = x.view(x.size(0), -1)
#         x = torch.flatten(x, start_dim=1)

#         x, _ = self.lstm1(x)
#         x = self.tanh(x)

#         x = self.fc1(x)
#         x = self.relu(x)

#         x = self.fc2(x)
#         x = self.relu(x)

#         x = self.fc3(x)
#         x = self.relu(x)

#         x = self.fc4(x)

#         return x.reshape(-1)


class CNNLSTMNet(nn.Module):
    def __init__(self, conv_out: int, lstm_hidden: int):
        super(CNNLSTMNet, self).__init__()

        self.conv1 = ConvBlock(14, conv_out, kernel_size=5, stride=2)
        self.conv2 = ConvBlock(conv_out, 64, kernel_size=3 ,stride=2)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.lstm1 = nn.LSTM(64, lstm_hidden, 2, batch_first=True, dropout=0.2)

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
