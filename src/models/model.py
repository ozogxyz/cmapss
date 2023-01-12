import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """Experimental network for time series foreacasting."""

    def __init__(self, conv_out: int, kernel_size: int, stride: int, lstm_hidden: int):
        super().__init__()

        self.conv1 = nn.Conv1d(14, conv_out, kernel_size, stride, padding=1)
        self.conv2 = nn.Conv1d(conv_out, conv_out * 2, kernel_size - 2, stride, padding=1)

        self.pool = nn.MaxPool1d(kernel_size=2, stride=1)

        conv1_out_dim: int = (32 - kernel_size) // stride + 1
        conv2_out_dim: int = (conv1_out_dim + 2 - (kernel_size - 2)) // (stride) + 1
        pool_out_dim: int = (conv2_out_dim - 2) // 1 + 1

        self.lstm = nn.LSTM(conv_out * 2 * pool_out_dim, lstm_hidden, 2, batch_first=True, dropout=0.2)

        self.fc1 = nn.Linear(lstm_hidden, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, input: torch.Tensor):
        input = self.conv1(input)
        input = self.conv2(input)
        input = self.pool(input)

        input = input.view(input.size(0), -1)
        input, _ = self.lstm(input)
        input = self.tanh(input)

        input = self.fc1(input)
        input = self.relu(input)
        input = self.fc2(input)
        input = self.relu(input)
        output = self.fc3(input)

        output = output.reshape(-1)
        return output
