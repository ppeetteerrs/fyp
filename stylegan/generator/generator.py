import math
import random
from typing import Dict, List, Literal, Optional, Tuple

import torch
from stylegan.generator.conv_block import ModConvBlock, UpModConvBlock
from stylegan.generator.mapping import MappingNetwork
from stylegan.generator.rgb import ToRGB
from torch import nn
from torch.functional import Tensor
from utils.config import CONFIG
from utils.utils import Resolution


class ConstantInput(nn.Module):
    """
    Constant input image
    """

    def __init__(self, channels: int, size: int):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channels, size, size))

    def forward(self, input: Tensor) -> Tensor:
        """
        Broadcast constant input to each sample
        """
        return self.input.repeat(input.shape[0], 1, 1, 1)


class Generator(nn.Module):
    def __init__(
        self,
        *,
        resolution: Resolution,
        latent_dim: int,
        n_mlp: int,
        lr_mlp_mult: float,
        channels: Dict[Resolution, int],
        blur_kernel: List[int]
    ):
        super().__init__()

        self.latent_dim = latent_dim

        # Create mapping network
        self.mapping = MappingNetwork(latent_dim, n_mlp, lr_mlp_mult)

        # Create constant input
        self.input = ConstantInput(channels[4], 4)

        # Create Conv, UpConv and ToRGB Blocks
        self.convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        self.n_layers = int(math.log(resolution, 2))
        self.n_w_plus = self.n_layers * 2 - 2

        for layer_idx in range(2, self.n_layers + 1):
            # Upsample condition
            upsample = layer_idx > 2

            # Calculate image size and channels at the layer
            prev_layer_size = 2 ** (layer_idx - 1)
            layer_size: Resolution = 2 ** layer_idx
            layer_channel = channels[layer_size]

            # Upsampling Conv Block
            if upsample:
                self.up_convs.append(
                    UpModConvBlock(
                        channels[prev_layer_size],
                        layer_channel,
                        3,
                        latent_dim,
                        2,
                        blur_kernel,
                    )
                )

            # Normal Conv Block
            self.convs.append(ModConvBlock(layer_channel, layer_channel, 3, latent_dim))

            # ToRGB Block
            self.to_rgbs.append(
                ToRGB(
                    layer_channel,
                    latent_dim,
                    2 if upsample else 1,
                    blur_kernel,
                )
            )

    @classmethod
    def from_config(cls, config: CONFIG) -> "Generator":
        return cls(
            resolution=config.RESOLUTION,
            latent_dim=config.LATENT_DIM,
            n_mlp=config.N_MLP,
            lr_mlp_mult=config.LR_MLP_MULT,
            channels=config.STYLEGAN_CHANNELS,
            blur_kernel=config.BLUR_KERNEL,
        )

    def make_noise(self) -> List[Tensor]:
        noises = []

        for i in range(2, self.n_layers + 1):
            if i > 2:
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device="cuda"))

            noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device="cuda"))

        return noises

    def mean_latent(self, n_sample: int) -> Tensor:
        return (
            self.mapping(torch.randn(n_sample, self.latent_dim, device="cuda"))
            .mean(0, keepdim=True)
            .detach()
        )

    def forward(
        self,
        # Input tensors (N, latent_dim)
        input: List[Tensor],
        input_type: Literal["z", "w", "w_plus"] = "z",
        # Truncation options
        trunc_option: Optional[Tuple[float, Tensor]] = None,
        # Mixing regularization options
        mix_index: Optional[int] = None,
        noises: Optional[List[Optional[Tensor]]] = None,
    ):
        # Get w vectors (can have 2 w vectors for mixing regularization)
        ws: List[Tensor]

        if input_type == "z":
            ws = [self.mapping(z) for z in input]
        else:
            ws = input

        # Perform truncation
        if trunc_option:
            trunc_coeff, trunc_tensor = trunc_option
            ws = [trunc_tensor + trunc_coeff * (w - trunc_tensor) for w in ws]

        # Mixing regularization (why add dimension 1 not 0 lol)
        if len(ws) == 1:
            # No mixing regularization
            mix_index = self.n_w_plus

            if input_type == "w_plus":
                w_plus = ws[0]
            else:
                w_plus = ws[0].unsqueeze(1).repeat(1, mix_index, 1)

        else:
            mix_index = mix_index if mix_index else random.randint(1, self.n_w_plus - 1)

            w_plus1 = ws[0].unsqueeze(1).repeat(1, mix_index, 1)
            w_plus2 = ws[1].unsqueeze(1).repeat(1, self.n_w_plus - mix_index, 1)

            w_plus = torch.cat([w_plus1, w_plus2], 1)
        # Get noise
        noises_: List[Optional[Tensor]] = (
            noises if noises else [None] * (self.n_w_plus - 1)
        )

        # Constant input
        out = self.input(w_plus)

        # References for this weird indexing:
        # https://github.com/NVlabs/stylegan2-ada-pytorch/issues/50
        # https://github.com/rosinality/stylegan2-pytorch/issues/278
        img = None
        for i in range(self.n_layers - 1):
            if i > 0:
                out = self.up_convs[i - 1](
                    out, w_plus[:, i * 2 - 1], noises_[i * 2 - 1]
                )

            out = self.convs[i](out, w_plus[:, i * 2], noises_[i * 2])
            img = self.to_rgbs[i](out, w_plus[:, i * 2 + 1], img)

        return img, w_plus
