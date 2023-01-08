from src.models.components.experiment import ExpNet
import torch
import pytest


@pytest.mark.parametrize("batch_size", [32])
def test_conv_forward(batch_size: int):
    experiment_net = ExpNet(conv_out=32, lstm_hidden=32)
    x = torch.randn(batch_size, 14, 30)
    y = experiment_net(x)
    assert y is not None
    assert isinstance(y, torch.Tensor)
    assert y.shape == torch.Size([batch_size])
