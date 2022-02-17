from argparse import ArgumentParser
from os import environ as ENV
from pathlib import Path

from dotenv import load_dotenv
from IPython import get_ipython

ip = get_ipython()

# Pass environment file from command line
parser = ArgumentParser()
parser.add_argument(
    "--env", type=str, default=".env", help="Environment (settings) file"
)

load_dotenv(".env", override=True)

if ip is None:
    env_file: Path = Path(parser.parse_args().env)
    print(f"Using environment file: {env_file}")

    load_dotenv(env_file, override=True)

import shutil as sh
from distutils.util import strtobool
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, cast

from stylegan2_torch import Resolution

T = TypeVar("T")


def env_get(
    key: str, default: T = "", transform: Callable[[str], T] = lambda x: x
) -> T:
    if ENV.get(key, "") == "":
        return default
    else:
        return transform(ENV.get(key, ""))


def env_or(key: str, default: str, alt: str) -> str:
    return alt if ENV.get(key, "").lower() == alt else default


def to_bool(key: str) -> bool:
    return bool(strtobool(ENV.get(key, "f")))


def parse_loss(option: Optional[str]) -> List[Tuple[str, str, float]]:
    if option is None or option == "":
        return []
    pairs = option.split(",")
    outputs: List[Tuple[str, str, float]] = []
    for pair in pairs:
        items = pair.split(":")
        if len(items) == 2:
            outputs.append((items[0], "", float(items[1])))
        elif len(items) == 3:
            outputs.append((items[0], items[1], float(items[2])))
        else:
            raise NotImplementedError(f"Invalid option {pair} in {option}.")
    return outputs


class CONFIG:
    EXPERIMENT_NAME = env_get("EXPERIMENT_NAME", "default")

    # Project
    PROJECT_DIR = Path(ENV.get("PROJECT_DIR", ""))
    RESOLUTION: Resolution = cast(Resolution, int(ENV.get("RESOLUTION", "")))

    # Dataset
    CHEXPERT_DIR = Path(ENV.get("CHEXPERT_DIR", ""))
    COVID_19_DIR = Path(ENV.get("COVID_19_DIR", ""))

    DATA_DIR = PROJECT_DIR / "input/data"
    CHEXPERT_TRAIN_LMDB = DATA_DIR / "chexpert_train"
    CHEXPERT_TEST_LMDB = DATA_DIR / "chexpert_test"
    COVID_19_TRAIN_LMDB = DATA_DIR / "covid_ct_train"
    COVID_19_TEST_LMDB = DATA_DIR / "covid_ct_test"

    # Output
    OUTPUT_DIR = Path(PROJECT_DIR / "output" / EXPERIMENT_NAME)

    # StyleGAN Model
    LATENT_DIM = 512
    N_MLP = 8
    LR_MLP_MULT = 0.01
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
    STYLEGAN_ITER = int(ENV.get("STYLEGAN_ITER", 0))
    STYLEGAN_BATCH = int(ENV.get("STYLEGAN_BATCH", 0))
    STYLEGAN_SAMPLES = int(ENV.get("STYLEGAN_SAMPLES", 0))
    STYLEGAN_LR = float(ENV.get("STYLEGAN_LR", 0))
    STYLEGAN_MIXING = float(ENV.get("STYLEGAN_MIXING", 0))
    STYLEGAN_CKPT = PROJECT_DIR / "input" / env_get("STYLEGAN_CKPT", "")

    # StyleGAN Regularization
    STYLEGAN_R1 = float(ENV.get("STYLEGAN_R1", 0))
    STYLEGAN_PATH_REG = float(ENV.get("STYLEGAN_PATH_REG", 0))
    STYLEGAN_PATH_BATCH_SHRINK = int(ENV.get("STYLEGAN_PATH_BATCH_SHRINK", 0))
    STYLEGAN_D_REG_INTERVAL = float(ENV.get("STYLEGAN_D_REG_INTERVAL", 0))
    STYLEGAN_G_REG_INTERVAL = float(ENV.get("STYLEGAN_G_REG_INTERVAL", 0))

    # pSp Model
    PSP_IN = ENV.get("PSP_IN", "").split(",")
    PSP_MERGE = ENV.get("PSP_MERGE", "").split(",")
    # PSP_ENCODER: str = env_or("PSP_ENCODER", "original", "deep")
    PSP_MERGER_LAYERS = int(ENV.get("PSP_MERGER_LAYERS", 5))
    PSP_MERGER_CHANNELS = int(ENV.get("PSP_MERGER_CHANNELS", 128))
    PSP_USE_MEAN: bool = to_bool("PSP_USE_MEAN")

    # pSp Training
    PSP_ITER = int(ENV.get("PSP_ITER", 0))
    PSP_BATCH_SIZE = int(ENV.get("PSP_BATCH_SIZE", 0))
    PSP_LR = float(ENV.get("PSP_LR", 0))
    PSP_CKPT = PROJECT_DIR / "input" / ENV.get("PSP_CKPT", "")
    PSP_DISCRIMINATOR_CKPT = (
        PROJECT_DIR / "input" / ENV.get("PSP_DISCRIMINATOR_CKPT", "")
    )

    PSP_SAMPLE_INTERVAL = int(ENV.get("PSP_SAMPLE_INTERVAL", 0))
    PSP_TEST_INTERVAL = int(ENV.get("PSP_TEST_INTERVAL", 0))
    PSP_TEST_BATCHES = int(ENV.get("PSP_TEST_BATCHES", 10000))
    PSP_CKPT_INTERVAL = int(ENV.get("PSP_CKPT_INTERVAL", 0))

    # pSp Training
    PSP_LOSS_L2 = parse_loss(ENV.get("PSP_LOSS_L2", ""))
    PSP_LOSS_ID = parse_loss(ENV.get("PSP_LOSS_ID", ""))
    PSP_LOSS_LPIPS = parse_loss(ENV.get("PSP_LOSS_LPIPS", ""))
    PSP_LOSS_SSIM = parse_loss(ENV.get("PSP_LOSS_SSIM", ""))
    PSP_LOSS_REG = float(ENV.get("PSP_LOSS_REG", 0))
    PSP_LOSS_DISCRIMINATOR = float(ENV.get("PSP_LOSS_DISCRIMINATOR", 0))

    # Classifier Training
    EFF_ARCH = ENV.get("EFF_ARCH", "")
    EFF_TRAIN_POSITIVE = ENV.get("EFF_TRAIN_POSITIVE", "")
    EFF_TRAIN_NEGATIVE = ENV.get("EFF_TRAIN_NEGATIVE", "")
    EFF_TEST_POSITIVE = ENV.get("EFF_TEST_POSITIVE", "")
    EFF_TEST_NEGATIVE = ENV.get("EFF_TEST_NEGATIVE", "")
    EFF_ITER = int(ENV.get("EFF_ITER", 0))
    EFF_BATCH_SIZE = int(ENV.get("EFF_BATCH_SIZE", 0))
    EFF_LR = float(ENV.get("EFF_LR", 0))
    EFF_CKPT = PROJECT_DIR / "input" / ENV.get("EFF_CKPT", "")
    EFF_TEST_INTERVAL = int(ENV.get("EFF_TEST_INTERVAL", 0))
    EFF_CKPT_INTERVAL = int(ENV.get("EFF_CKPT_INTERVAL", 0))


def guard(create: bool = True):
    if create:
        (CONFIG.OUTPUT_DIR / "sample").mkdir(parents=True, exist_ok=True)
        (CONFIG.OUTPUT_DIR / "checkpoint").mkdir(parents=True, exist_ok=True)
        (CONFIG.OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)
        sh.copyfile(
            str(env_file), str(CONFIG.OUTPUT_DIR / f"{CONFIG.EXPERIMENT_NAME}.env")
        )
