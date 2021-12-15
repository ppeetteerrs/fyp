import math
import random
from typing import Literal, Tuple, Union

import torch
from op import conv2d_gradfix
from torch import autograd, nn
from torch.functional import Tensor
from torch.nn import functional as F


# optimize?
def requires_grad(model: nn.Module, flag: bool) -> None:
    # for p in model.parameters():
    #     p.requires_grad = flag
    model.requires_grad_(flag)


def accumulate(
    model1: nn.Module,
    model2: nn.Module,
    decay: float = 0.5 ** (32 / (10 * 1000)),
) -> None:
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


# def sample_data(loader: DataLoader[int]) -> Generator[Tensor, None, None]:
#     # while True:
#     for batch in loader:
#         yield batch


def d_logistic_loss(real_pred: Tensor, fake_pred: Tensor) -> Tensor:
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred: Tensor, real_img: Tensor) -> Tensor:
    with conv2d_gradfix.no_weight_gradients():
        (grad_real,) = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred: Tensor) -> Tensor:
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(
    fake_img: Tensor,
    latents: Tensor,
    mean_path_length: Union[Tensor, Literal[0]],
    decay: float = 0.01,
) -> Tuple[Tensor, Tensor, Tensor]:
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    (grad,) = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch: int, latent_dim: int, n_noise: int, device: str):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    return torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)


def mixing_noise(batch: int, latent_dim: int, prob: float, device: str):
    if random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None
