from typing import Any, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models import densenet121
from utils.config import CONFIG


def l2_norm(input: Tensor, axis: int = 1) -> Tensor:
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


class DenseDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((224, 224))
        self.dense = densenet121(pretrained=True)
        self.dense.classifier = nn.Linear(1024, 2)

        if CONFIG.PSP_DISCRIMINATOR_CKPT != CONFIG.PROJECT_DIR / "input":
            self.load_state_dict(torch.load(CONFIG.PSP_DISCRIMINATOR_CKPT)["d"])

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.pool(x)
        features = self.dense.features(x.expand(-1, 3, -1, -1))
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.dense.classifier(out)
        return out, features


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, pred: Tensor) -> Tuple[Tensor, Tensor]:
        targ = Tensor([1 for _ in range(pred.shape[0])]).to(pred.device).long()
        loss = self.loss(pred, targ)
        prob = F.softmax(pred, dim=1)[:, 1].mean()
        return loss, prob


class IDLoss(nn.Module):
    def forward(self, y_hat_features: Tensor, y_features: Tensor) -> Tensor:
        n_samples = y_features.shape[0]
        y_hat_features = l2_norm(torch.flatten(y_hat_features, 1))
        y_features = l2_norm(torch.flatten(y_features, 1)).detach()
        loss: Any = 0
        for i in range(n_samples):
            diff_target = y_hat_features[i].dot(y_features[i])
            loss += 1 - diff_target

        return loss / n_samples
