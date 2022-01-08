import math
from typing import List

import torch
from stylegan.equalized_lr import EqualLinear
from stylegan.op import upfirdn2d
from stylegan.utils import make_kernel
from torch import nn
from torch.functional import Tensor
from torch.nn import functional as F


def mod(weight: Tensor, style: Tensor) -> Tensor:
    # weight: (1, C_out, C_in, K_h, K_w)
    # style: (N, 1, C_in, 1, 1)
    return weight * style


def demod(weight: Tensor) -> Tensor:
    # weight: (N, C_out, C_in, K_h, K_w)
    # demod: (N, C_out, 1, 1, 1)
    batch, out_channel, _, _, _ = weight.shape
    demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8).view(
        batch, out_channel, 1, 1, 1
    )
    return weight * demod


def blur(
    input: Tensor,
    blur_kernel: Tensor,
    up: int = 2,
    kernel_size: int = 3,
) -> Tensor:
    p = (len(blur_kernel) - up) - (kernel_size - 1)
    return upfirdn2d(input, blur_kernel, pad=((p + 1) // 2 + up - 1, p // 2 + 1))


def group_conv_up(
    input: Tensor,
    weight: Tensor,
    up: int = 2,
) -> Tensor:
    # weight: (N, C_out, C_in, K_h, K_w)
    batch, in_channel, height, width = input.shape
    _, out_channel, _, k_h, k_w = weight.shape

    weight = weight.transpose(1, 2).reshape(batch * in_channel, out_channel, k_h, k_w)
    input = input.view(1, batch * in_channel, height, width)

    out = F.conv_transpose2d(input=input, weight=weight, stride=up, groups=batch)
    _, _, out_h, out_w = out.shape
    return out.view(batch, out_channel, out_h, out_w)


def group_conv(input: Tensor, weight: Tensor) -> Tensor:
    # weight: (N, C_out, C_in, K_h, K_w)
    batch, in_channel, height, width = input.shape
    _, out_channel, _, k_h, k_w = weight.shape

    weight = weight.view(batch * out_channel, in_channel, k_h, k_w)
    input = input.view(1, batch * in_channel, height, width)

    # Change padding to "same"
    out = F.conv2d(input=input, weight=weight, padding="same", groups=batch)
    return out.view(batch, out_channel, height, width)


class ModConvBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        latent_dim: int,
    ):
        super().__init__()

        # Affine mapping from W to style vector
        self.affine = EqualLinear(latent_dim, in_channel, bias_init=1)

        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        # Assumed odd sized kernel
        self.padding = kernel_size // 2

    def forward(self, input: Tensor, w: Tensor) -> Tensor:
        batch, in_channel, height, width = input.shape

        style = self.affine(w).view(batch, 1, in_channel, 1, 1)  # (N, 1, C_in, 1, 1)
        weight = modulate(self.scale * self.weight, style)  # (N, C_out, C_in, K_h, K_w)

        # Divide weights by square sum across in_channel and spatial footprint
        weight = demodulate(weight)

        # Reshape to use group convolution
        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )
        input = input.view(1, batch * in_channel, height, width)

        # Change padding to "same"
        out = F.conv2d(input=input, weight=weight, padding=self.padding, groups=batch)

        return out.view(batch, self.out_channel, height, width)


class UpModConvBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        latent_dim: int,
        blur_kernel: List[int],
    ):
        super().__init__()

        # Affine mapping from W to style vector
        self.affine = EqualLinear(latent_dim, in_channel, bias_init=1)

        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.padding = kernel_size // 2

        # Compensate for kernel size of transposed convolution
        factor = 2
        self.kernel: Tensor
        self.register_buffer("kernel", make_kernel(blur_kernel, factor))

        p = (len(blur_kernel) - factor) - (kernel_size - 1)
        self.pad = ((p + 1) // 2 + factor - 1, p // 2 + 1)

    def forward(self, input: Tensor, w: Tensor) -> Tensor:
        batch, in_channel, height, width = input.shape

        style = self.affine(w).view(batch, 1, in_channel, 1, 1)  # (N, 1, C_in, 1, 1)
        weight = modulate(self.scale * self.weight, style)  # (N, C_out, C_in, K_h, K_w)

        # Divide weights by square sum across in_channel and spatial footprint
        weight = demodulate(weight)

        # Reshape to use group convolution
        weight = weight.transpose(1, 2).reshape(
            batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
        )
        input = input.view(1, batch * in_channel, height, width)
        out = F.conv_transpose2d(
            input=input,
            weight=weight,
            stride=2,
            padding=0,
            output_padding=0,
            groups=batch,
        )
        _, _, height, width = out.shape
        out = out.view(batch, self.out_channel, height, width)
        return upfirdn2d(out, self.kernel, pad=self.pad)


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        style_dim: int,
        upsample: bool = False,
        blur_kernel: List[int] = [1, 3, 3, 1],
        demodulate: bool = True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(
        self, input: Tensor, style: Tensor, noise: Optional[Tensor] = None
    ) -> Tensor:
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.activate(out)

        return out
