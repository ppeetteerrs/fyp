"""Generate pSp"""

from typing import Dict, Tuple, cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from utils import to_device
from utils.cli import OPTIONS
from utils.cli.psp import PSPArch, PSPGenerate
from utils.dataset import LMDBImageDataset

from psp import pSp

ARCH_OPTIONS = cast(PSPArch, OPTIONS.arch)
GEN_OPTIONS = cast(PSPGenerate, ARCH_OPTIONS.cmd)


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
        self.dataset = LMDBImageDataset(
            GEN_OPTIONS.dataset,
            ARCH_OPTIONS.classes,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2,
            prefetch_factor=1,
            drop_last=True,
        )

    def forward(self, imgs: Dict[str, Tensor]) -> Tuple[Dict[str, Tensor], Tensor]:
        """
        Forward propagate input images and add output image to dictionary.
        Also returns $W+$ style vectors.
        """

        # Concat image along channel direction
        img_in = torch.cat([imgs[key] for key in ARCH_OPTIONS.inputs], dim=1)

        img_out, w_plus = self.net(img_in, "generate")
        return {**imgs, "out": img_out}, w_plus

    def generate(self):
        """Generates images using pSp model."""

        self.net.eval()
        with torch.no_grad():
            for idx, item in tqdm(enumerate(self.dataloader)):
                imgs = to_device(item)
                output_imgs, _ = self.forward(imgs)

                for k, v in output_imgs.items():
                    out_dir = OPTIONS.output_dir / f"generated/{k}"
                    out_dir.mkdir(parents=True, exist_ok=True)

                    save_image(
                        make_grid(
                            v,
                            nrow=1,
                            normalize=True,
                            value_range=(-1, 1),
                        ),
                        (out_dir / f"{str(idx).zfill(10)}.png"),
                    )


def psp_generate():
    Task().generate()
