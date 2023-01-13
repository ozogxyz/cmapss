import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class CNNLSTMTransformer(nn.Module):
    """Experimental network for multi variate time series foreacasting."""

    def __init__(
        self,
        nhead: int,
        dim_feedforward: int,
        num_encoder_layers: int,
        lstm_hidden: int,
        num_lstm_layers: int,
    ):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=30, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.lstm = nn.LSTM(420, lstm_hidden, num_lstm_layers, batch_first=True, dropout=0.2)
        self.tanh = nn.Tanh()

        self.fc1 = nn.Linear(lstm_hidden, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor):
        x = self.encoder(x)

        x = x.view(x.size(0), -1)
        x, _ = self.lstm(x)
        x = self.tanh(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x.reshape(-1)
