import math
from typing import Optional

import torch
from stylegan.op import fused_leaky_relu
from torch import nn
from torch.functional import Tensor
from torch.nn import functional as F


class EqualConv2d(nn.Module):
    """
    Conv2d with equalized learning rate
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()

        # Equalized Learning Rate
        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        # std = gain / sqrt(fan_in)
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.stride = stride
        self.padding = padding
        self.bias = nn.Parameter(torch.zeros(out_channel)) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        return F.conv2d(
            input=input,
            weight=self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=1,
            groups=1,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    """
    Linear with equalized learning rate
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias_init: int = 0,
        lr_mul: float = 1,
    ):
        super().__init__()

        # Equalized Learning Rate
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul

        self.lr_mul = lr_mul

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class EqualLeakyReLU(nn.Module):
    """
    Leaky ReLU with equalized learning rate
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        lr_mul: float = 1,
    ):
        super().__init__()

        # Equalized Learning Rate
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        self.bias = nn.Parameter(torch.zeros(out_dim))

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul

        self.lr_mul = lr_mul

    def forward(self, input: Tensor) -> Tensor:
        out = F.linear(input, self.weight * self.scale)
        out = fused_leaky_relu(out, self.bias * self.lr_mul)
        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )
