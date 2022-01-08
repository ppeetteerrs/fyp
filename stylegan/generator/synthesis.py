import math
import random
from typing import List, Literal, Optional, Tuple

import torch
from stylegan.equalized_lr import EqualLeakyReLU, EqualLinear
from stylegan.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
from stylegan.parsers import Sizes, TrainArgs
from stylegan.utils import make_kernel
from torch import nn
from torch.functional import Tensor
from torch.nn import functional as F


class ConstantInput(nn.Module):
    """
    Constant input image
    """

    def __init__(self, args: TrainArgs, *, size: Sizes):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, args.channels[size], size, size))

    def forward(self, input: Tensor) -> Tensor:
        """
        Broadcast constant input to each sample
        """
        return self.input.repeat(input.shape[0], 1, 1, 1)


class ModConv(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        latent_dim: int,
        demod: bool,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.padding = kernel_size // 2

        # Affine mapping from W to style vector
        self.affine = EqualLinear(latent_dim, in_channel, bias_init=1)

        self.demod = demod

    def forward(self, input: Tensor, w: Tensor) -> Tensor:
        batch, in_channel, height, width = input.shape

        style = self.affine(w).view(batch, 1, in_channel, 1, 1)  # (N, 1, C_in, 1, 1)
        weight = self.scale * self.weight * style  # (N, C_out, C_in, K_h, K_w)

        # Divide weights by square sum across in_channel and spatial footprint
        if self.demod:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8).view(
                batch, self.out_channel, 1, 1, 1
            )  # (N, C_out, 1, 1, 1)
            weight = weight * demod

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        input = input.view(1, batch * in_channel, height, width)
        # Change padding to "same"
        out = F.conv2d(
            input=input,
            weight=weight,
            bias=None,
            stride=1,
            padding=self.padding,
            dilation=1,
            groups=batch,
        )
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)

        return out


class UpModConv(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        latent_dim: int,
        demod: bool,
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.padding = kernel_size // 2

        # Affine mapping from W to style vector
        self.affine = EqualLinear(latent_dim, in_channel, bias_init=1)

        self.demod = demod

    def forward(self, input: Tensor, w: Tensor) -> Tensor:
        batch, in_channel, height, width = input.shape

        style = self.affine(w).view(batch, 1, in_channel, 1, 1)  # (N, 1, C_in, 1, 1)
        weight = self.scale * self.weight * style  # (N, C_out, C_in, K_h, K_w)

        # Divide weights by square sum across in_channel and spatial footprint
        if self.demod:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8).view(
                batch, self.out_channel, 1, 1, 1
            )  # (N, C_out, 1, 1, 1)
            weight = weight * demod

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        input = input.view(1, batch * in_channel, height, width)
        # Change padding to "same"
        out = F.conv2d(
            input=input,
            weight=weight,
            bias=None,
            stride=1,
            padding=self.padding,
            dilation=1,
            groups=batch,
        )
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)

        return out


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        style_dim: int,
        demodulate: bool = True,
        upsample: bool = False,
        blur_kernel: List[int] = [1, 3, 3, 1],
    ):
        super().__init__()

        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample

        if upsample:
            # Compensate for kernel size of transposed convolution
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), up=factor)

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.padding = kernel_size // 2

        # Affine mapping from W to style vector
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample})"
        )

    def forward(self, input: Tensor, style: Tensor) -> Tensor:
        batch, in_channel, height, width = input.shape
        # Affine mapping and weight modulation
        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        # Divide weights by square sum across in_channel and spatial footprint
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        # Convolve with / without up sampling (in groups)
        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(
                input=input,
                weight=weight,
                bias=None,
                stride=2,
                padding=0,
                output_padding=0,
                dilation=1,
                groups=batch,
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(
                input=input,
                weight=weight,
                bias=None,
                stride=1,
                padding=self.padding,
                dilation=1,
                groups=batch,
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out
