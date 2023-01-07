from src.models.components.experiment import ConvBlock, ConvNet
import torch
import pytest


def test_conv_init():
    conv_net = ConvNet(conv_out=32, lstm_hidden=32)
    assert conv_net is not None
    assert isinstance(conv_net, ConvNet)


@pytest.mark.parametrize("batch_size", [32])
def test_conv_forward(batch_size: int):
    conv_net = ConvNet(conv_out=32, lstm_hidden=32)
    x = torch.randn(batch_size, 14, 30)
    y = conv_net(x)
    assert y is not None
    assert isinstance(y, torch.Tensor)
    assert y.shape == torch.Size([batch_size])
