"""Train StyleGAN"""
from os import environ
from typing import cast

import torch
import wandb
from dataset import LMDBImageDataset
from stylegan2_torch import Discriminator, Generator
from stylegan2_torch.loss import d_loss as get_d_loss
from stylegan2_torch.loss import d_reg_loss as get_d_reg_loss
from stylegan2_torch.loss import g_loss as get_g_loss
from stylegan2_torch.loss import g_reg_loss as get_g_reg_loss
from stylegan2_torch.utils import mixing_noise
from torch import distributed, optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.utils import make_grid
from tqdm import tqdm
from utils import accumulate, repeat
from utils.cli import OPTIONS
from utils.cli.stylegan import StyleGANArch, StyleGANGenerate, StyleGANTrain
from utils.distributed import reduce_loss_dict, reduce_sum, setup_distributed

ARCH_OPTIONS = cast(StyleGANArch, OPTIONS.arch)
GEN_OPTIONS = cast(StyleGANGenerate, ARCH_OPTIONS.cmd)


class Task:
    def __init__(self) -> None:

        # Use deterministic algorithms

        environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

        # Create models
        self.g_ema = Generator(ARCH_OPTIONS.output_resolution).to("cuda")
        self.g_ema.eval()

        # Load checkpoint
        ckpt = torch.load(ARCH_OPTIONS.ckpt)
        self.g_ema.load_state_dict(ckpt["g_ema"])
        self.mean_latent = self.g_ema.mean_latent(10000, "cuda")

    def generate(self):
        wandb.init(
            project="FYP",
            entity="ppeetteerrs",
            group="stylegan",
            job_type="generate",
            name=OPTIONS.name,
            config=OPTIONS.to_dict(),
        )

        with torch.no_grad():
            for step in tqdm(
                range(0, GEN_OPTIONS.n_images),
                total=GEN_OPTIONS.n_images,
            ):
                noise = mixing_noise(
                    1,
                    ARCH_OPTIONS.latent_dim,
                    GEN_OPTIONS.mixing,
                    "cuda",
                )

                img = self.g_ema(
                    noise, trunc_option=(GEN_OPTIONS.trunc, self.mean_latent)
                )

                wandb.log(
                    {
                        "img": wandb.Image(
                            make_grid(
                                img,
                                nrow=1,
                                normalize=True,
                                value_range=(-1, 1),
                            )
                        )
                    },
                    step=step,
                )


def stylegan_generate():
    Task().generate()
