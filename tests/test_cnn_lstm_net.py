from src.models.components.cnn_lstm_net import CNNLSTMNet
import torch

def test_cnn_lstm_net_init():
    cnn_lstm_net = CNNLSTMNet()
    assert cnn_lstm_net is not None
    assert isinstance(cnn_lstm_net, CNNLSTMNet)

def test_cnn_lstm_net_forward():
    cnn_lstm_net = CNNLSTMNet()
    x = torch.randn(32, 14, 30)
    y = cnn_lstm_net(x)
    assert y is not None
    assert isinstance(y, torch.Tensor)
    assert y.shape == (32, 1)