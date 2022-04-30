"""pSp encoder module"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from stylegan2_torch import EqualLinear, Resolution
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


class SEModule(nn.Module):
    def __init__(self, channel: int, reduction: int):
        """
        Squeeze and Excitation Block

        Info:
            SEModule adopts a bottleneck architecture (reduce then restore) to increase computational efficiency.
            SEModule incorporates global information when scaling feature maps (e.g. dealing with relative means).
            [Reference](https://amaarora.github.io/2020/07/24/SeNet.html#intuition-behind-squeeze-and-excitation-networks)
        """

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
        """
        Performs squeeze and excitation.
        """
        return input * self.se(input)


class ResnetBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, stride: int):
        """
        Non-standard ResNet block used by pSp.

        Warning:
            pSp's ResNet implementation is a mess. This ResNetBlock is copied from [TreB1eN](https://github.com/TreB1eN/InsightFace_Pytorch)'s implementation of [ArcFace](https://openaccess.thecvf.com/content_CVPR_2019/papers/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.pdf)'s ResNet because ArcFace's source code was written in MXNet (sorry but why?!).

            There are some differences between TreB1eN's implementation and the MXNet as shown [here](https://github.com/TreB1eN/InsightFace_Pytorch/issues/37). The `MaxPool` shortcut is probably taken from [here](https://arxiv.org/pdf/2004.04989.pdf).
        """

        super().__init__()

        if in_channel == out_channel:
            self.shortcut = MaxPool2d(1, stride)
        else:
            self.shortcut = Sequential(
                Conv2d(in_channel, out_channel, (1, 1), stride),
                BatchNorm2d(out_channel),
            )

        self.convs = Sequential(
            BatchNorm2d(in_channel),
            Conv2d(in_channel, out_channel, (3, 3), (1, 1), 1),
            PReLU(out_channel),
            BatchNorm2d(out_channel),
            Conv2d(out_channel, out_channel, (3, 3), stride, 1),
            BatchNorm2d(out_channel),
            SEModule(out_channel, 16),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        One ResNet block forward propagation.
        """

        return self.convs(x) + self.shortcut(x)


class DownsampleBlock(nn.Sequential):
    def __init__(
        self, in_channel: int, out_channel: int, n_layers: int, down_factor: int = 2
    ):
        """
        ResNet blocks for a given resolution (downsample then same resolution).
        It will contain 1 `in_channel to out_channel` ResNet block that downsamples the input features maps
        follwoed by n-1 `out_channel to out_channel` ResNet blocks of the same resolution.

        Args:
            in_channel (int): Number of input channels
            out_channel (int): Number of output channels
            n_layers (int): Number of ResNet blocks
            down_factor (int, optional): Downsampling factor. Defaults to 2.
        """

        super().__init__(
            ResnetBlock(in_channel, out_channel, down_factor),
            *[ResnetBlock(out_channel, out_channel, 1) for i in range(n_layers - 1)],
        )


class ToStyle(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, resolution: int):
        """
        Maps feature map to $W+$ style vector through a series of downsampling convolutions + LeakyReLU.
        """

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
        """
        Convert feature maps `x` to one $W+$ style vector.
        """

        x = self.convs(x).view(-1, self.out_channel)
        return self.linear(x)


class Encoder(nn.Module):
    def __init__(self, in_channel: int, resolution: Resolution):
        """
        pSp encoder (feature pyramid with ResNet backbone + feature-to-style maps).

        Info:
            For face similarity measurement, the SOTA progress is `SphereFace` => `CosFace` => `ArcFace`.
            SphereFace proposed a [CNN architecture](https://arxiv.org/pdf/1704.08063.pdf) (Table 2) slightly different from ResNet.
            They are also the ones who suggested using `PReLU` instead of `ReLU`.

            From SpherFace's [github](https://github.com/wy1iu/sphereface/search?p=2&q=prelu):
            > In SphereFace, our network architecures use residual units as building blocks, but are quite different from the standrad ResNets (e.g., BatchNorm is not used, the prelu replaces the relu, different initializations, etc). We proposed 4-layer, 20-layer, 36-layer and 64-layer architectures for face recognition

            Standard ResNets use bottleneck archiecture from ResNet50 onwards, so they would have 3 layers for each of the `[3, 4, 6, 3]` ResNet blocks. This [implementation](https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/resnet.py) follows that (just like original [ResNet]( https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py)). But [ArcFace's MXNet](https://github.com/deepinsight/insightface/blob/8b79096e70a10a4899f1ce59882ea4d56e634d40/recognition/arcface_mxnet/symbol/fresnet.py#L1175) did not use bottleneck so that have 2 layers per block, thus the use of `[3, 4, 14, 3]`. There is this convention to add additional blocks to `conv4_x`, forgot where that came from (SphereFace or ResNet)?

        Args:
            in_channel (int): _description_
            resolution (Resolution): _description_

        """

        super().__init__()

        self.input_layer = Sequential(
            Conv2d(in_channel, 64, (3, 3), 1, 1, bias=False),
            BatchNorm2d(64),
            PReLU(64),
        )

        # Backbone
        # self.block_N means output from block is of size N x N
        self.block_128 = DownsampleBlock(in_channel=64, out_channel=64, n_layers=3)
        self.block_64 = DownsampleBlock(in_channel=64, out_channel=128, n_layers=4)
        self.block_32 = DownsampleBlock(in_channel=128, out_channel=256, n_layers=14)
        self.block_16 = DownsampleBlock(in_channel=256, out_channel=512, n_layers=3)

        # Feature-style maps
        self.coarse_map = nn.ModuleList(ToStyle(512, 512, 16) for _ in range(3))
        self.mid_map = nn.ModuleList(ToStyle(512, 512, 32) for _ in range(4))
        self.fine_map = nn.ModuleList(
            ToStyle(512, 512, 64)
            for _ in range(int(math.log(resolution, 2)) * 2 - 2 - 7)
        )

        # Feature pyramid up-channelling (I doubt the use of the lateral layer)
        self.lateral_32 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.lateral_64 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def upsample_add(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Adds `x` to `y` after bilinear upsampling.
        """

        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True) + y

    def forward(self, x: Tensor) -> Tensor:
        """
        Converts input `x` into $W+$ style vectors
        """

        x = self.input_layer(x)
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
