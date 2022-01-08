import math
import random
from typing import List, Literal, Optional, Tuple

import torch
from stylegan.equalized_lr import EqualLeakyReLU
from stylegan.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d
from stylegan.parsers import Sizes, TrainArgs
from stylegan.utils import make_kernel
from torch import nn
from torch.functional import Tensor
from torch.nn import functional as F


class Normalize(nn.Module):
    """
    Normalize latent vector for each sample
    """

    def forward(self, input: Tensor) -> Tensor:
        # input: (N, style_dim)
        # Normalize z in each sample to N(0,1)
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class MappingNetwork(nn.Sequential):
    def __init__(self, args: TrainArgs) -> None:
        super().__init__(
            Normalize(),
            *[
                EqualLeakyReLU(
                    args.latent_dim,
                    args.latent_dim,
                    lr_mul=args.lr_mlp_mult,
                )
                for _ in range(args.n_mlp)
            ]
        )
