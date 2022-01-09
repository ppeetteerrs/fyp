import math
from typing import List

import torch
from stylegan.equalized_lr import Blur, EqualConv2d, EqualLeakyReLU, EqualLinear
from stylegan.op import FusedLeakyReLU
from stylegan.parsers import TrainArgs
from torch import nn
from torch.functional import Tensor


class ConvBlock(nn.Sequential):
    def __init__(self, in_channel: int, out_channel: int, kernel_size: int):
        super().__init__(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=kernel_size // 2,
                stride=1,
                bias=False,
            ),
            FusedLeakyReLU(out_channel, bias=True),
        )


class DownConvBlock(nn.Sequential):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        down: int,
        blur_kernel: List[int],
    ):
        super().__init__(
            Blur(blur_kernel, -down, kernel_size),
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=0,
                stride=2,
                bias=False,
            ),
            FusedLeakyReLU(out_channel, bias=True),
        )


class RGBDown(nn.Sequential):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        down: int,
        blur_kernel: List[int],
    ):

        super().__init__(
            Blur(blur_kernel, -down, kernel_size),
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=0,
                stride=2,
                bias=False,
            ),
        )


class ResBlock(nn.Module):
    def __init__(self, in_channel: int, out_channel: int, blur_kernel: List[int]):
        super().__init__()

        self.conv = ConvBlock(in_channel, in_channel, 3)
        self.down_conv = DownConvBlock(
            in_channel, out_channel, 3, down=2, blur_kernel=blur_kernel
        )
        self.skip = RGBDown(in_channel, out_channel, 1, down=2, blur_kernel=blur_kernel)

    def forward(self, input: Tensor) -> Tensor:
        out = self.conv(input)
        out = self.down_conv(out)
        skip = self.skip(input)
        return (out + skip) / math.sqrt(2)


class Discriminator(nn.Module):
    def __init__(self, args: TrainArgs):
        super().__init__()

        self.args = args

        # FromRGB followed by ResBlock
        self.n_layers = int(math.log(args.size, 2))

        self.blocks = nn.Sequential(
            ConvBlock(1, args.channels[args.size], 1),
            *[
                ResBlock(
                    args.channels[2 ** i], args.channels[2 ** (i - 1)], args.blur_kernel
                )
                for i in range(self.n_layers, 2, -1)
            ],
        )

        # Minibatch std settings
        self.stddev_group = 4
        self.stddev_feat = 1

        # Final layers
        self.final_conv = ConvBlock(args.channels[4] + 1, args.channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLeakyReLU(args.channels[4] * 4 * 4, args.channels[4]),
            EqualLinear(args.channels[4], 1),
        )

    def forward(self, input: Tensor) -> Tensor:
        # Downsampling blocks
        out: Tensor = self.blocks(input)

        # Minibatch stddev layer in Progressive GAN https://www.youtube.com/watch?v=V1qQXb9KcDY
        # Purpose is to provide variational information to the discriminator to prevent mode collapse
        # Other layers do not cross sample boundaries
        batch, channel, height, width = out.shape
        n_groups = min(batch, self.stddev_group)
        stddev = out.view(
            n_groups, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdim=True).squeeze(2)
        stddev = stddev.repeat(n_groups, 1, height, width)
        out = torch.cat([out, stddev], 1)

        # Final layers
        out = self.final_conv(out)
        return self.final_linear(out.view(batch, -1))
