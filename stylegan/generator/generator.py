import math
import random
from typing import List, Literal, Optional, Tuple

import torch
from stylegan.generator.conv_block import ModConvBlock, UpModConvBlock
from stylegan.generator.mapping import MappingNetwork
from stylegan.generator.rgb import ToRGB
from stylegan.parsers import Sizes, TrainArgs
from torch import nn
from torch.functional import Tensor


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
    def __init__(self, args: TrainArgs):
        super().__init__()

        self.args = args

        # Create mapping network
        self.mapping = MappingNetwork(args.latent_dim, args.n_mlp, args.lr_mlp_mult)

        # Create constant input
        self.input = ConstantInput(args.channels[4], 4)

        # Create Conv, UpConv and ToRGB Blocks
        self.convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        self.n_layers = int(math.log(args.size, 2))
        self.n_w_plus = self.n_layers * 2 - 2

        for layer_idx in range(2, self.n_layers + 1):
            # Upsample condition
            upsample = layer_idx > 2

            # Calculate image size and channels at the layer
            prev_layer_size = 2 ** (layer_idx - 1)
            layer_size: Sizes = 2 ** layer_idx
            layer_channel = args.channels[layer_size]

            # Upsampling Conv Block
            if upsample:
                self.up_convs.append(
                    UpModConvBlock(
                        args.channels[prev_layer_size],
                        layer_channel,
                        3,
                        args.latent_dim,
                        2,
                        args.blur_kernel,
                    )
                )

            # Normal Conv Block
            self.convs.append(
                ModConvBlock(layer_channel, layer_channel, 3, args.latent_dim)
            )

            # ToRGB Block
            self.to_rgbs.append(
                ToRGB(
                    layer_channel,
                    args.latent_dim,
                    2 if upsample else 1,
                    args.blur_kernel,
                )
            )

    def make_noise(self) -> List[Tensor]:
        noises = []

        for i in range(2, self.n_layers + 1):
            if i > 2:
                noises.append(
                    torch.randn(1, 1, 2 ** i, 2 ** i, device=self.args.device)
                )

            noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=self.args.device))

        return noises

    def mean_latent(self, n_sample: int) -> Tensor:
        return self.mapping(
            torch.randn(n_sample, self.args.latent_dim, device=self.args.device)
        ).mean(0, keepdim=True)

    def forward(
        self,
        # Input tensors (N, latent_dim)
        input: List[Tensor],
        input_type: Literal["z", "w", "w_plus"] = "z",
        # Truncation options
        trunc_option: Optional[Tuple[int, Tensor]] = None,
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

        img = None
        for i in range(self.n_layers - 1):
            if i > 0:
                out = self.up_convs[i - 1](out, w_plus[:, i - 1], noises_[i - 1])

            out = self.convs[i](out, w_plus[:, i], noises_[i])
            img = self.to_rgbs[i](out, w_plus[:, i + 1], img)

        return img, w_plus
