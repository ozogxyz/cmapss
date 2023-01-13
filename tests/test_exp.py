import torch.nn as nn
from torch import Tensor
from src.models.exp import CNNLSTMTransformer
import pytest
import torch


@pytest.fixture(autouse=True)
def input():
    return torch.randn(32, 14, 30)


def test_cnn_lstm_transformer(input: Tensor) -> None:
    model = CNNLSTMTransformer(conv_out=32, lstm_hidden=50)
    output = model(input)
    assert model.encoder is not None
    assert isinstance(output, Tensor)
    assert output.shape == torch.Size([32])
