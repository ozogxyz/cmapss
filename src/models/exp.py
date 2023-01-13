import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class CNNLSTMTransformer(nn.Module):
    """Experimental network for multi variate time series foreacasting."""

    def __init__(self, conv_out: int, lstm_hidden: int):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model=30, nhead=6, dim_feedforward=420, batch_first=True)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=6)

        self.lstm = nn.LSTM(30, lstm_hidden, 2, batch_first=True, dropout=0.2)
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(lstm_hidden, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)
        # print(x.shape)

        # x = x.view(x.size(0), -1)
        x = self.lstm(x)[0]
        # print(x.shape)
        x = self.tanh(x[:, -1, :])

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x.reshape(-1)
