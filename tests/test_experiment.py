from src.models.components.experiment import ConvBlock, ConvolutionLSTM
import torch
import pytest


# @pytest.mark.parametrize("batch_size", [32, 64])
# @pytest.mark.parametrize("window_size", [30, 40])
# @pytest.mark.parametrize("in_channels", [14, 21])
# @pytest.mark.parametrize("out_channels", [14, 21])
# @pytest.mark.parametrize("kernel_size", [3, 5])
# @pytest.mark.parametrize("stride", [1, 2])
# @pytest.mark.parametrize("padding", [1, 2])
# def test_conv_block(
#     batch_size: int, window_size: int, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int
# ):
#     input = torch.randn(batch_size, in_channels, window_size)
#     model = ConvBlock(in_channels, out_channels, kernel_size, stride, padding)
#     output = model(input)

#     assert output.shape == model._output_shape_hook
#     assert output.dtype == torch.float32

# @pytest.mark.parametrize("batch_size", [32, 64])
# @pytest.mark.parametrize("window_size", [30])
# @pytest.mark.parametrize("in_channels", [14, 21])
# @pytest.mark.parametrize("out_channels", [14, 21])
# @pytest.mark.parametrize("kernel_size", [3, 5])
# @pytest.mark.parametrize("stride", [1, 2])
# @pytest.mark.parametrize("padding", [1, 2])
# @pytest.mark.parametrize("num_layers", [1, 2])
# def test_convolution_lstm(
#     batch_size: int, window_size: int, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, num_layers: int
# ):
#     input = torch.randn(batch_size, in_channels, window_size)
#     model = ConvolutionLSTM(in_channels, out_channels, kernel_size, stride, padding, num_layers)
#     output = model(input)

#     assert output.shape == model._output_shape_hook
#     assert output.dtype == torch.float32



@pytest.mark.parametrize("batch_size", [32])
@pytest.mark.parametrize("window_size", [30])
@pytest.mark.parametrize("in_channels", [14])
@pytest.mark.parametrize("out_channels", [32])
@pytest.mark.parametrize("kernel_size", [5])
@pytest.mark.parametrize("stride", [2])
@pytest.mark.parametrize("padding", [1])
@pytest.mark.parametrize("num_layers", [1])
def test_convolution_lstm(
    batch_size: int, window_size: int, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int, num_layers: int
):
    input = torch.randn(batch_size, in_channels, window_size)
    model = ConvolutionLSTM(in_channels, out_channels, kernel_size, stride, padding, num_layers)
    output = model(input)

    assert output.shape == model._output
    assert output.dtype == torch.float32