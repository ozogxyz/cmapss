import pytest
import torch
import torch.nn as nn
from torch import Tensor

from src.models.exp import CNNLSTMTransformer


@pytest.fixture(autouse=True)
def input():
    return torch.randn(32, 14, 30)


@pytest.mark.parametrize("nhead", [6, 10])
@pytest.mark.parametrize("dim_feedforward", [1024, 2048])
@pytest.mark.parametrize("num_encoder_layers", [1, 2, 4])
@pytest.mark.parametrize("lstm_hidden", [16, 32, 50])
@pytest.mark.parametrize("num_lstm_layers", [1, 2, 3])
def test_cnn_lstm_transformer(
    input: Tensor,
    nhead: int,
    dim_feedforward: int,
    num_encoder_layers: int,
    lstm_hidden: int,
    num_lstm_layers: int,
) -> None:
    model = CNNLSTMTransformer(
        nhead=nhead,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=num_encoder_layers,
        lstm_hidden=lstm_hidden,
        num_lstm_layers=num_lstm_layers,
    )

    output = model(input)
    assert model.encoder is not None
    assert isinstance(output, Tensor)
    assert output.shape == torch.Size([32])
