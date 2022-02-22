import math

import torch
import torch.nn.functional as F
from stylegan2_torch import Resolution
from torch import Tensor, nn
from torch.nn import BatchNorm2d, Conv2d, PReLU, Sequential

from psp.encoder import DownsampleBlock, ToStyle


class EncoderDeep(nn.Module):
    def __init__(self, in_channel: int, resolution: Resolution):
        super().__init__()

        self.input_layer = Sequential(
            Conv2d(in_channel, 64, (3, 3), 1, 1, bias=False),
            BatchNorm2d(64),
            PReLU(64),
        )

        # self.block_N means output from block is of size N x N
        self.block_256 = DownsampleBlock(in_channel=64, out_channel=64, n_layers=3)
        self.block_128 = DownsampleBlock(in_channel=64, out_channel=128, n_layers=4)
        self.block_64 = DownsampleBlock(in_channel=128, out_channel=128, n_layers=6)
        self.block_32 = DownsampleBlock(in_channel=128, out_channel=256, n_layers=8)
        self.block_16 = DownsampleBlock(in_channel=256, out_channel=512, n_layers=3)

        self.map_16 = nn.ModuleList(ToStyle(512, 512, 16) for _ in range(3))
        self.map_32 = nn.ModuleList(ToStyle(512, 512, 32) for _ in range(3))
        self.map_64 = nn.ModuleList(ToStyle(512, 512, 64) for _ in range(3))
        self.map_128 = nn.ModuleList(ToStyle(512, 512, 128) for _ in range(3))
        self.map_256 = nn.ModuleList(
            ToStyle(512, 512, 256)
            for _ in range(int(math.log(resolution, 2)) * 2 - 2 - 12)
        )

        self.lateral_32 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.lateral_64 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        self.lateral_128 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)
        self.lateral_256 = nn.Conv2d(64, 512, kernel_size=1, stride=1, padding=0)

    def upsample_add(self, x: Tensor, y: Tensor) -> Tensor:
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True) + y

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_layer(x)
        feature_256 = self.block_256(x)
        feature_128 = self.block_128(feature_256)
        feature_64 = self.block_64(feature_128)
        feature_32 = self.block_32(feature_64)
        feature_16 = self.block_16(feature_32)

        latents = []
        latents.extend(layer(feature_16) for layer in self.map_16)

        combined_32 = self.upsample_add(feature_16, self.lateral_32(feature_32))
        latents.extend(layer(combined_32) for layer in self.map_32)

        combined_64 = self.upsample_add(combined_32, self.lateral_64(feature_64))
        latents.extend(layer(combined_64) for layer in self.map_64)

        combined_128 = self.upsample_add(combined_64, self.lateral_128(feature_128))
        latents.extend(layer(combined_128) for layer in self.map_128)

        combined_256 = self.upsample_add(combined_128, self.lateral_256(feature_256))
        latents.extend(layer(combined_256) for layer in self.map_256)

        latents = torch.stack(latents, dim=1)

        return latents
