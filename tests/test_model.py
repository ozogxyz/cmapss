import pytest
import torch

from src.models.model import CNNLSTM


@pytest.fixture
def batch_size():
    return 32


@pytest.mark.parametrize("conv_out", [32, 64])
@pytest.mark.parametrize("kernel_size", [5])
@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("lstm_hidden", [32, 50])
def test_conv_forward(
    batch_size: int, conv_out: int, kernel_size: int, stride: int, lstm_hidden: int
) -> None:
    model = CNNLSTM(conv_out, kernel_size, stride, lstm_hidden)
    input = torch.randn(batch_size, 14, 30)
    output = model(input)
    assert output.shape == torch.Size([batch_size])
