import math

import numpy as np
import torch
import torch.nn.functional as F
from stylegan.equalized_lr import EqualLinear
from torch import Tensor, nn
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm2d,
    Conv2d,
    MaxPool2d,
    PReLU,
    ReLU,
    Sequential,
    Sigmoid,
)
from utils.utils import Resolution


class SEModule(nn.Module):
    """
    Squeeze and Excitation Block

    bottleneck architecture (reduce then restore): goal is to increase computational efficiency
    squeeze and excitation: global information incorporated when scaling feature maps (https://amaarora.github.io/2020/07/24/SeNet.html#intuition-behind-squeeze-and-excitation-networks) e.g. dealing with different means
    """

    def __init__(self, channel: int, reduction: int):
        super().__init__()
        self.se = nn.Sequential(
            # Can replace with global avg pool
            AdaptiveAvgPool2d(1),
            # Can just use normal linear
            Conv2d(channel, channel // reduction, kernel_size=1, padding=0, bias=False),
            ReLU(inplace=True),
            Conv2d(channel // reduction, channel, kernel_size=1, padding=0, bias=False),
            Sigmoid(),
        )

    def forward(self, input: Tensor) -> Tensor:
        return input * self.se(input)


class ResnetBlock(nn.Module):
    """
    Resnet block. Not standard implementation
    """

    def __init__(self, in_channel: int, out_channel: int, stride: int):
        super().__init__()

        if in_channel == out_channel:
            self.shortcut = MaxPool2d(1, stride)
        else:
            self.shortcut = Sequential(
                Conv2d(in_channel, out_channel, (1, 1), stride, bias=False),
                BatchNorm2d(out_channel),
            )

        self.convs = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, out_channel, (3, 3), (1, 1), 1, bias=False),
            PReLU(out_channel),
            Conv2d(out_channel, out_channel, (3, 3), stride, 1, bias=False),
            BatchNorm2d(out_channel),
            SEModule(out_channel, 16),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.convs(x) + self.shortcut(x)


class DownsampleBlock(nn.Sequential):
    """
    Resnet block for a given resolution (downsample then same resolution)
    """

    def __init__(
        self, in_channel: int, out_channel: int, n_layers: int, down_factor: int = 2
    ):
        super().__init__(
            ResnetBlock(in_channel, out_channel, down_factor),
            *[ResnetBlock(out_channel, out_channel, 1) for i in range(n_layers - 1)]
        )


class ToStyle(nn.Module):
    """
    Maps feature map to w+ style vector
    """

    def __init__(self, in_channel: int, out_channel: int, resolution: int):
        super().__init__()

        self.out_channel = out_channel
        self.resolution = resolution
        modules = [
            Conv2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(),
        ]
        for _ in range(int(np.log2(resolution)) - 1):
            modules += [
                Conv2d(out_channel, out_channel, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(),
            ]
        self.convs = nn.Sequential(*modules)
        # Why suddenly equalized learning rate?!
        self.linear = EqualLinear(out_channel, out_channel, lr_mult=1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.convs(x).view(-1, self.out_channel)
        return self.linear(x)


class Encoder(nn.Module):
    def __init__(self, *, resolution: Resolution):
        super().__init__()

        self.resize = torch.nn.AdaptiveAvgPool2d((256, 256))

        self.input_layer = Sequential(
            Conv2d(1, 64, (3, 3), 1, 1, bias=False),
            BatchNorm2d(64),
            PReLU(64),
        )

        # self.block_N means output from block is of size N x N
        self.block_128 = DownsampleBlock(in_channel=64, out_channel=64, n_layers=3)
        self.block_64 = DownsampleBlock(in_channel=64, out_channel=128, n_layers=4)
        self.block_32 = DownsampleBlock(in_channel=128, out_channel=256, n_layers=14)
        self.block_16 = DownsampleBlock(in_channel=256, out_channel=512, n_layers=3)

        self.coarse_map = nn.ModuleList(ToStyle(512, 512, 16) for _ in range(3))
        self.mid_map = nn.ModuleList(ToStyle(512, 512, 32) for _ in range(4))
        self.fine_map = nn.ModuleList(
            ToStyle(512, 512, 64)
            for _ in range(int(math.log(resolution, 2)) * 2 - 2 - 7)
        )

        self.lateral_32 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.lateral_64 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def upsample_add(self, x: Tensor, y: Tensor) -> Tensor:
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True) + y

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_layer(self.resize(x))

        feature_64 = self.block_64(self.block_128(x))
        feature_32 = self.block_32(feature_64)
        feature_16 = self.block_16(feature_32)

        latents = []
        latents.extend(layer(feature_16) for layer in self.coarse_map)

        combined_32 = self.upsample_add(feature_16, self.lateral_32(feature_32))
        latents.extend(layer(combined_32) for layer in self.mid_map)

        combined_64 = self.upsample_add(combined_32, self.lateral_64(feature_64))
        latents.extend(layer(combined_64) for layer in self.fine_map)

        return torch.stack(latents, dim=1)
