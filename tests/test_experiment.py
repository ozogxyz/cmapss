from src.models.components.experiment import ConvBlock
import torch
import pytest


@pytest.fixture
def batch_size():
    return 32


@pytest.fixture(autouse=True)
def in_channels():
    return 14


@pytest.fixture(autouse=True)
def window_size():
    return 30


@pytest.fixture(autouse=True)
def out_channels():
    return 32


@pytest.fixture(autouse=True)
def kernel_size():
    return 5


@pytest.fixture(autouse=True)
def stride():
    return 1


@pytest.fixture(autouse=True)
def padding(kernel_size: int):
    return kernel_size // 2


@pytest.fixture(autouse=True)
def dilation():
    return 1


@pytest.fixture(autouse=True)
def input_tensor(batch_size: int, in_channels: int, window_size: int):
    return torch.randn(batch_size, in_channels, window_size)


@pytest.mark.parametrize("out_channels", [32, 64])
@pytest.mark.parametrize("kernel_size", [3, 5])
@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("dilation", [1, 2])
def test_conv_block(
    input_tensor: torch.Tensor,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
):
    block = ConvBlock(
        input=input_tensor,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
    )
    assert block.in_channels == in_channels
    assert block.out_channels == out_channels
    assert block.kernel_size == kernel_size
    assert block.stride == stride
    assert block.padding == padding
    assert block.dilation == dilation
