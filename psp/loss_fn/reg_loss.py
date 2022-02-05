from typing import Optional

import torch
from torch import Tensor, nn
from utils.config import CONFIG


class RegLoss(nn.Module):
    def forward(self, latent: Tensor, latent_avg: Optional[Tensor]) -> Tensor:
        excess_latent = latent - latent_avg if CONFIG.PSP_USE_MEAN else latent
        return torch.sum(excess_latent.norm(2, dim=(1, 2))) / excess_latent.shape[0]
