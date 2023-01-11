import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.experiment import Net

@pytest.fixture()
def input():
    return torch.randn(32, 14, 30)


def test_net(input: torch.Tensor):
    model = Net(in_channels=14, out_channels=32, kernel_size=5)

    out = model(input)
    assert out.shape == (32, 1)

