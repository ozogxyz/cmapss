import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """Experimental network for multi variate time series foreacasting."""

    def __init__(self, conv_out: int, lstm_hidden: int):
        super().__init__()
        self.conv1 = nn.Conv1d(14, conv_out, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(conv_out)
        self.conv2 = nn.Conv1d(conv_out, conv_out * 2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(conv_out * 2)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lstm = nn.LSTM(448, lstm_hidden, 2, batch_first=True, dropout=0.2)
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(lstm_hidden, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

        self.relu = nn.ReLU(inplace=True)


    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x, _ = self.lstm(x)
        x = self.tanh(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x.reshape(-1)
