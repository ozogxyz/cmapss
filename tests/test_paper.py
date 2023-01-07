from src.models.components.paper import Paper
import torch
import pytest


def test_cnn_lstm_net_init():
    paper_net = Paper(conv_out=32, lstm_hidden=32)
    assert paper_net is not None
    assert isinstance(paper_net, Paper)


@pytest.mark.parametrize("batch_size", [32])
def test_cnn_lstm_net_forward(batch_size: int):
    paper_net = Paper(conv_out=32, lstm_hidden=32)
    x = torch.randn(batch_size, 14, 30)
    y = paper_net(x)
    assert y is not None
    assert isinstance(y, torch.Tensor)
    assert y.shape == torch.Size([batch_size])
