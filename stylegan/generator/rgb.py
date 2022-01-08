import math
import random
from typing import List, Literal, Optional, Tuple

import torch
from stylegan.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
from stylegan.utils import make_kernel
from torch import nn
from torch.functional import Tensor
from torch.nn import functional as F


class ToRGB(nn.Module):
    def __init__(
        self,
        in_channel: int,
        style_dim: int,
        upsample: bool = True,
        blur_kernel: List[int] = [1, 3, 3, 1],
    ):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 1, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, input: Tensor, style: Tensor, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out
