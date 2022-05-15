"""Train StyleGAN"""
from os import environ
from typing import cast

import torch
from stylegan2_torch import Generator
from stylegan2_torch.utils import mixing_noise
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from utils.cli import OPTIONS
from utils.cli.stylegan import StyleGANArch, StyleGANGenerate

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
        if "latent_avg" in ckpt:
            self.mean_latent = ckpt["latent_avg"].to("cuda")
        else:
            self.mean_latent = self.g_ema.mean_latent(10000, "cuda")

        # Make image output folders
        (OPTIONS.output_dir / "generated").mkdir(parents=True, exist_ok=True)

    def generate(self):

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

                save_image(
                    make_grid(
                        img,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1),
                    ),
                    (OPTIONS.output_dir / f"generated/{str(step).zfill(10)}.png"),
                )


def stylegan_generate():
    Task().generate()
