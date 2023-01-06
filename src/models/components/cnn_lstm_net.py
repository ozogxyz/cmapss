import torch
import torch.nn as nn


class CNNLSTMNet(nn.Module):
    def __init__(self, conv_out: int, lstm_hidden: int):
        super(CNNLSTMNet, self).__init__()

        self.conv1 = nn.Conv1d(14, conv_out,
        kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(conv_out, 64, kernel_size=5, stride=2, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=7, stride=3, padding=2)

        self.lstm1 = nn.LSTM(512, 64, batch_first=True)

        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.relu(x)

        x = self.dropout(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)
        x = torch.flatten(x, start_dim=1)
        x, _ = self.lstm1(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        return x.reshape(-1)