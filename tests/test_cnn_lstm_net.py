from src.models.components.cnn_lstm_net import CNNLSTMNet
import torch
import pytest


def test_cnn_lstm_net_init():
    cnn_lstm_net = CNNLSTMNet()
    assert cnn_lstm_net is not None
    assert isinstance(cnn_lstm_net, CNNLSTMNet)


@pytest.mark.parametrize("batch_size", [32, 64, 200])
def test_cnn_lstm_net_forward(batch_size: int):
    cnn_lstm_net = CNNLSTMNet()
    x = torch.randn(batch_size, 14, 30)
    y = cnn_lstm_net(x)
    assert y is not None
    assert isinstance(y, torch.Tensor)
    assert y.shape == (batch_size, 1)
