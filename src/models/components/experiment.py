import torch
import torch.nn as nn


class ExpNet(nn.Module):
    """Experimental network for time series foreacasting."""
    def __init__(self, conv_out: int, kernel_size: int, stride:int, lstm_hidden: int):
        super(ExpNet, self).__init__()

        self.conv1 = nn.Conv1d(14, conv_out, kernel_size, stride, padding=1)
        self.conv2 = nn.Conv1d(conv_out, conv_out*2, kernel_size-2, stride, padding=1)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        conv1_out_dim: int = (32 - kernel_size) // stride + 1
        conv2_out_dim: int = (conv1_out_dim + 2 - (kernel_size - 2)) // (stride) + 1
        pool_out_dim: int = (conv2_out_dim - 2) // 2 + 1


        self.lstm = nn.LSTM(conv_out * 2 * pool_out_dim, lstm_hidden, 2, batch_first=True, dropout=0.2)

        self.fc1 = nn.Linear(lstm_hidden, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
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
