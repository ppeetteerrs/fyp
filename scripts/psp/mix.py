"""Mix pSp"""

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, cast

import lpips
import torch
import torch.nn.functional as F
import wandb
from dataset import MultiImageDataset
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from utils import repeat, to_device
from utils.cli import OPTIONS
from utils.cli.psp import PSPArch, PSPGenerate, PSPMix, PSPTrain

from psp import pSp
from psp.loss import IDLoss, RegLoss
from psp.ranger import Ranger

ARCH_OPTIONS = cast(PSPArch, OPTIONS.arch)
MIX_OPTIONS = cast(PSPMix, ARCH_OPTIONS.cmd)


class Task:
    def __init__(self):
        """Class that wraps a pSp model to perform training, generation or mixing."""

        # Load checkpoint
        ckpt = torch.load(ARCH_OPTIONS.ckpt)
        self.net = pSp(
            ckpt,
            use_mean=ARCH_OPTIONS.use_mean,
            e_in_channel=len(ARCH_OPTIONS.inputs),
            e_resolution=ARCH_OPTIONS.input_resolution,
            g_resolution=ARCH_OPTIONS.output_resolution,
            g_latent_dim=ARCH_OPTIONS.latent_dim,
            g_n_mlp=ARCH_OPTIONS.n_mlp,
            g_lr_mlp_mult=ARCH_OPTIONS.lr_mlp_mult,
            g_channels=ARCH_OPTIONS.channels_map,
            g_blur_kernel=ARCH_OPTIONS.blur_kernel,
        ).to("cuda")

        assert (
            self.net.resumed
        ), "Please provide a checkpoint to a pretrained pSp model."

        # Initialize dataset
        self.dataset = MultiImageDataset(
            [Path(f"input/data/{item}/train") for item in MIX_OPTIONS.datasets],
            ARCH_OPTIONS.classes,
        )

        self.dataloader = repeat(
            DataLoader(
                self.dataset,
                batch_size=2,
                shuffle=False,
                num_workers=2,
                prefetch_factor=2,
                drop_last=True,
            )
        )

        # Start wandb
        wandb.init(
            project="FYP",
            entity="ppeetteerrs",
            group="psp",
            job_type="generate",
            name=OPTIONS.name,
            config=OPTIONS.to_dict(),
        )

    def forward(self, imgs: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Forward propagate input images and add output image to dictionary.
        Also returns $W+$ style vectors
        """

        # Concat image along channel direction
        img_in = torch.cat([imgs[key] for key in ARCH_OPTIONS.inputs], dim=1)
        img_in1 = img_in[0].unsqueeze(0)
        img_in2 = img_in[1].unsqueeze(0)

        img_out, w_plus = self.net((img_in1, img_in2), "generate", MIX_OPTIONS.mix_mode)

        return {**imgs, "out": img_out}, w_plus

    def mix(self):
        self.net.eval()

        with torch.no_grad():
            for idx in tqdm(list(range(MIX_OPTIONS.n_images))):
                output_imgs, _ = self.forward(to_device(next(self.dataloader)))
                wandb.log(
                    {
                        k: wandb.Image(
                            make_grid(
                                v,
                                nrow=1,
                                normalize=True,
                                value_range=(-1, 1),
                            )
                        )
                        for k, v in output_imgs.items()
                    },
                    step=idx,
                )


def psp_mix():
    Task().mix()
