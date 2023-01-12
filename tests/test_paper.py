import pytest
import torch

from src.models.model import CNNLSTM


def test_cnn_lstm_net_init():
    cnnlstm_net = CNNLSTM(conv_out=32, lstm_hidden=32)
    assert cnnlstm_net is not None
    assert isinstance(cnnlstm_net, CNNLSTM)


@pytest.mark.parametrize("batch_size", [32])
def test_cnn_lstm_net_forward(batch_size: int):
    cnnlstm_net = CNNLSTM(conv_out=32, lstm_hidden=32)
    x = torch.randn(batch_size, 14, 30)
    y = cnnlstm_net(x)
    assert y is not None
    assert isinstance(y, torch.Tensor)
    assert y.shape == torch.Size([batch_size])
