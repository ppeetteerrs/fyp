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
    def __init__(self, latent_dim: int, n_mlp: int, lr_mlp_mult: float):
        super().__init__(
            Normalize(),
            *[
                EqualLeakyReLU(
                    latent_dim,
                    latent_dim,
                    lr_mult=lr_mlp_mult,
                )
                for _ in range(n_mlp)
            ]
        )
