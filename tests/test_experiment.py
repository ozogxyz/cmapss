from src.models.components.experiment import Experiment
import torch
import pytest


def test_conv_init():
    experiment_net = Experiment(conv_out=32, lstm_hidden=32)
    assert experiment_net is not None
    assert isinstance(experiment_net, Experiment)


@pytest.mark.parametrize("batch_size", [32])
def test_conv_forward(batch_size: int):
    experiment_net = Experiment(conv_out=32, lstm_hidden=32)
    x = torch.randn(batch_size, 14, 30)
    y = experiment_net(x)
    assert y is not None
    assert isinstance(y, torch.Tensor)
    assert y.shape == torch.Size([batch_size])
