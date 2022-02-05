from dataclasses import dataclass, field
from distutils.util import strtobool
from os import environ as ENV
from pathlib import Path
from typing import Dict, List, Literal, cast

from dotenv import load_dotenv

from utils.utils import Resolution

load_dotenv()


@dataclass(frozen=True)
class CONFIG:
    # Project
    PROJECT_DIR = Path(ENV["PROJECT_DIR"])

    # Dataset
    CHEXPERT_DIR = Path(ENV["CHEXPERT_DIR"])
    COVID_19_DIR = Path(ENV["COVID_19_DIR"])
    CHEXPERT_TRAIN_LMDB = Path(ENV["CHEXPERT_TRAIN_LMDB"])
    CHEXPERT_TEST_LMDB = Path(ENV["CHEXPERT_TEST_LMDB"])
    COVID_19_TRAIN_LMDB = Path(ENV["COVID_19_TRAIN_LMDB"])
    COVID_19_TEST_LMDB = Path(ENV["COVID_19_TEST_LMDB"])

    # Image
    RESOLUTION: Resolution = cast(Resolution, int(ENV["RESOLUTION"]))

    # StyleGAN Model
    LATENT_DIM = 512
    N_MLP = 8
    LR_MLP_MULT = 0.01
    N_CHANNEL = 1
    BLUR_KERNEL: List[int] = field(default_factory=lambda: [1, 3, 3, 1])
    STYLEGAN_CHANNELS: Dict[Resolution, int] = field(
        default_factory=lambda: {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 512,
            128: 256,
            256: 128,
            512: 64,
            1024: 32,
        }
    )

    # StyleGAN Training
    STYLEGAN_ITER = int(ENV["STYLEGAN_ITER"])
    STYLEGAN_BATCH = int(ENV["STYLEGAN_BATCH"])
    STYLEGAN_SAMPLES = int(ENV["STYLEGAN_SAMPLES"])
    STYLEGAN_R1 = float(ENV["STYLEGAN_R1"])
    STYLEGAN_PATH_REG = float(ENV["STYLEGAN_PATH_REG"])
    STYLEGAN_PATH_BATCH_SHRINK = int(ENV["STYLEGAN_PATH_BATCH_SHRINK"])
    STYLEGAN_D_REG_INTERVAL = float(ENV["STYLEGAN_D_REG_INTERVAL"])
    STYLEGAN_G_REG_INTERVAL = float(ENV["STYLEGAN_G_REG_INTERVAL"])
    STYLEGAN_MIXING = float(ENV["STYLEGAN_MIXING"])
    STYLEGAN_CKPT = None if ENV["STYLEGAN_CKPT"] == "" else ENV["STYLEGAN_CKPT"]
    STYLEGAN_LR = float(ENV["STYLEGAN_LR"])
    STYLEGAN_OUTPUT = Path(ENV["PROJECT_DIR"]) / ENV["STYLEGAN_OUTPUT_DIR"]
    PSP_PRETRAINED = Path(ENV["PROJECT_DIR"]) / ENV["PSP_PRETRAINED"]
    PSP_ENCODER: Literal["v1", "v2"] = (
        "v1" if ENV["PSP_ENCODER"].lower() == "v1" else "v2"
    )
    PSP_OPTIM: Literal["ranger", "adam"] = (
        "adam" if ENV["PSP_CKPT_TYPE"].lower() == "adam" else "ranger"
    )
    PSP_LR = float(ENV["PSP_LR"])
    PSP_CKPT_TYPE: Literal["stylegan", "psp"] = (
        "stylegan" if ENV["PSP_CKPT_TYPE"].lower() == "stylegan" else "psp"
    )
    PSP_CKPT = (
        Path(ENV["PROJECT_DIR"])
        / ENV["PSP_PRETRAINED"]
        / f"{ENV['PSP_CKPT_TYPE'].lower()}.pt"
    )
    PSP_USE_LATENT = bool(
        ENV["PSP_USE_LATENT"] != "" and strtobool(ENV["PSP_USE_LATENT"])
    )
    PSP_OUTPUT_DIR = Path(ENV["PROJECT_DIR"]) / ENV["PSP_OUTPUT_DIR"]
    PSP_ITER = int(ENV["PSP_ITER"])
    PSP_BATCH_SIZE = int(ENV["PSP_BATCH_SIZE"])
    PSP_SAMPLE_INTERVAL = int(ENV["PSP_SAMPLE_INTERVAL"])
    PSP_TEST_INTERVAL = int(ENV["PSP_TEST_INTERVAL"])
    PSP_CKPT_INTERVAL = int(ENV["PSP_CKPT_INTERVAL"])
    PSP_USE_LOCALIZER = bool(
        ENV["PSP_USE_LOCALIZER"] != "" and strtobool(ENV["PSP_USE_LOCALIZER"])
    )
    PSP_LOCALIZER_WEIGHT = float(ENV["PSP_LOCALIZER_WEIGHT"])
    PSP_LOSS_L2 = float(ENV["PSP_LOSS_L2"])
    PSP_LOSS_ID = float(ENV["PSP_LOSS_ID"])
    PSP_LOSS_LPIPS = float(ENV["PSP_LOSS_LPIPS"])
    PSP_LOSS_REG = float(ENV["PSP_LOSS_REG"])
    PSP_LOSS_DISCRIMINATOR = float(ENV["PSP_LOSS_DISCRIMINATOR"])


config = CONFIG()
(config.STYLEGAN_OUTPUT / "sample").mkdir(parents=True, exist_ok=True)
(config.STYLEGAN_OUTPUT / "checkpoint").mkdir(parents=True, exist_ok=True)
(config.PSP_OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
(config.PSP_OUTPUT_DIR / "sample").mkdir(parents=True, exist_ok=True)
(config.PSP_OUTPUT_DIR / "checkpoint").mkdir(parents=True, exist_ok=True)


def load_config(loaded: CONFIG):
    global config
    config = loaded
