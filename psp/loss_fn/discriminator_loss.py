import torch.nn.functional as F
from torch import Tensor, nn


class DiscriminatorLoss(nn.Module):
    def forward(self, d_output: Tensor) -> Tensor:
        return F.softplus(-d_output).mean()
