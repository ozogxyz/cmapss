import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.experiment import ConvBlock, FeatureExtractor


def test_feature_extractor():
    input = torch.randn(32, 14, 30)
    model = ConvBlock(input=input, out_channels=30, kernel_size=3)

    fe = FeatureExtractor(model)
    assert 4 == 5


