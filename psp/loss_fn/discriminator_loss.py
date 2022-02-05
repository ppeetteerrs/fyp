from typing import Optional

import torch
from torch import Tensor, nn
from utils.config import config


class DiscriminatorLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, latent: Tensor, latent_avg: Optional[Tensor]) -> Tensor:
        excess_latent = latent - latent_avg if config.PSP_USE_LATENT else latent
        return torch.sum(excess_latent.norm(2, dim=(1, 2))) / excess_latent.shape[0]
