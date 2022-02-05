from argparse import ArgumentParser
from dataclasses import dataclass, field
from distutils.util import strtobool
from optparse import Option
from os import environ as ENV
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, TypeVar, Union, cast

from dotenv import load_dotenv

from utils.utils import Resolution

# Pass environment file from command line
parser = ArgumentParser()
parser.add_argument(
    "--env", type=str, default=".env", help="Environment (settings) file"
)
env_file_path = parser.parse_args().env
print(f"Using environment file: {env_file_path}")
load_dotenv(env_file_path)

T = TypeVar("T")


def env_get(
    key: str, default: T = "", transform: Callable[[str], T] = lambda x: x
) -> T:
    if ENV[key] == "":
        return default
    else:
        return transform(ENV[key])


def env_or(key: str, default: str, alt: str) -> str:
    return alt if ENV[key].lower() == alt else default


def to_bool(key: str) -> bool:
    return bool(ENV[key] != "" and strtobool(ENV[key]))


class CONFIG:
    EXPERIMENT_NAME = env_get("EXPERIMENT_NAME", "default")

    # Project
    PROJECT_DIR = Path(ENV["PROJECT_DIR"])
    RESOLUTION: Resolution = cast(Resolution, int(ENV["RESOLUTION"]))

    # Dataset
    DATA_DIR = Path(ENV["DATA_DIR"])
    CHEXPERT_DIR = DATA_DIR / ENV["CHEXPERT_DIR"]
    COVID_19_DIR = DATA_DIR / ENV["COVID_19_DIR"]
    CHEXPERT_TRAIN_LMDB = DATA_DIR / ENV["CHEXPERT_TRAIN_LMDB"]
    CHEXPERT_TEST_LMDB = DATA_DIR / ENV["CHEXPERT_TEST_LMDB"]
    COVID_19_TRAIN_LMDB = DATA_DIR / ENV["COVID_19_TRAIN_LMDB"]
    COVID_19_TEST_LMDB = DATA_DIR / ENV["COVID_19_TEST_LMDB"]

    # Output
    STYLEGAN_OUTPUT_DIR = (
        Path(ENV["PROJECT_DIR"]) / "output" / EXPERIMENT_NAME / "stylegan"
    )
    PSP_OUTPUT_DIR = Path(ENV["PROJECT_DIR"]) / "output" / EXPERIMENT_NAME / "psp"

    # StyleGAN Model
    LATENT_DIM = 512
    N_MLP = 8
    LR_MLP_MULT = 0.01
    N_CHANNEL = 1
    BLUR_KERNEL: List[int] = [1, 3, 3, 1]
    STYLEGAN_CHANNELS: Dict[Resolution, int] = {
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

    # StyleGAN Training
    STYLEGAN_ITER = int(ENV["STYLEGAN_ITER"])
    STYLEGAN_BATCH = int(ENV["STYLEGAN_BATCH"])
    STYLEGAN_SAMPLES = int(ENV["STYLEGAN_SAMPLES"])
    STYLEGAN_LR = float(ENV["STYLEGAN_LR"])
    STYLEGAN_MIXING = float(ENV["STYLEGAN_MIXING"])
    STYLEGAN_CKPT = env_get("STYLEGAN_CKPT", None)

    # StyleGAN Regularization
    STYLEGAN_R1 = float(ENV["STYLEGAN_R1"])
    STYLEGAN_PATH_REG = float(ENV["STYLEGAN_PATH_REG"])
    STYLEGAN_PATH_BATCH_SHRINK = int(ENV["STYLEGAN_PATH_BATCH_SHRINK"])
    STYLEGAN_D_REG_INTERVAL = float(ENV["STYLEGAN_D_REG_INTERVAL"])
    STYLEGAN_G_REG_INTERVAL = float(ENV["STYLEGAN_G_REG_INTERVAL"])

    # pSp Model
    PSP_ENCODER: str = env_or("PSP_ENCODER", "original", "deep")
    PSP_USE_MEAN: bool = to_bool("PSP_USE_MEAN")

    # pSp Training
    PSP_ITER = int(ENV["PSP_ITER"])
    PSP_BATCH_SIZE = int(ENV["PSP_BATCH_SIZE"])
    PSP_OPTIM: str = env_or("PSP_OPTIM", "ranger", "adam")
    PSP_LR = float(ENV["PSP_LR"])
    PSP_CKPT = Path(ENV["PROJECT_DIR"]) / ENV["PSP_CKPT"]

    PSP_SAMPLE_INTERVAL = int(ENV["PSP_SAMPLE_INTERVAL"])
    PSP_TEST_INTERVAL = int(ENV["PSP_TEST_INTERVAL"])
    PSP_CKPT_INTERVAL = int(ENV["PSP_CKPT_INTERVAL"])

    # pSp Training
    PSP_USE_LOCALIZER = to_bool("PSP_USE_LOCALIZER")
    PSP_LOCALIZER_WEIGHT = float(ENV["PSP_LOCALIZER_WEIGHT"])
    PSP_LOSS_L2 = float(ENV["PSP_LOSS_L2"])
    PSP_LOSS_ID = float(ENV["PSP_LOSS_ID"])
    PSP_LOSS_ID_DISCRIMINATOR = float(ENV["PSP_LOSS_ID_DISCRIMINATOR"])
    PSP_LOSS_LPIPS = float(ENV["PSP_LOSS_LPIPS"])
    PSP_LOSS_REG = float(ENV["PSP_LOSS_REG"])
    PSP_LOSS_DISCRIMINATOR = float(ENV["PSP_LOSS_DISCRIMINATOR"])


(CONFIG.STYLEGAN_OUTPUT_DIR / "sample").mkdir(parents=True, exist_ok=True)
(CONFIG.STYLEGAN_OUTPUT_DIR / "checkpoint").mkdir(parents=True, exist_ok=True)
(CONFIG.PSP_OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
(CONFIG.PSP_OUTPUT_DIR / "sample").mkdir(parents=True, exist_ok=True)
(CONFIG.PSP_OUTPUT_DIR / "checkpoint").mkdir(parents=True, exist_ok=True)
