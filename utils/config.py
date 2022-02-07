import shutil as sh
from argparse import ArgumentParser
from distutils.util import strtobool
from os import environ as ENV
from pathlib import Path
from typing import Callable, Dict, List, TypeVar, cast

from dotenv import load_dotenv

from utils.utils import Resolution

# Pass environment file from command line
parser = ArgumentParser()
parser.add_argument(
    "--env", type=str, default=".env", help="Environment (settings) file"
)

project_dir = Path(ENV.get("PROJECT_DIR", ""))
env_file: Path = (project_dir / parser.parse_args().env).resolve()
print(f"Using environment file: {env_file}")

load_dotenv(project_dir / ".env")
load_dotenv(env_file)

T = TypeVar("T")


def env_get(
    key: str, default: T = "", transform: Callable[[str], T] = lambda x: x
) -> T:
    if ENV[key] == "":
        return default
    else:
        return transform(ENV[key])


def env_or(key: str, default: str, alt: str) -> str:
    return alt if ENV.get(key, "").lower() == alt else default


def to_bool(key: str) -> bool:
    return bool(strtobool(ENV.get(key, "f")))


class CONFIG:
    EXPERIMENT_NAME = env_get("EXPERIMENT_NAME", "default")

    # Project
    PROJECT_DIR = Path(ENV.get("PROJECT_DIR", ""))
    RESOLUTION: Resolution = cast(Resolution, int(ENV.get("RESOLUTION", "")))

    # Dataset
    CHEXPERT_DIR = ENV.get("CHEXPERT_DIR", "")
    COVID_19_DIR = ENV.get("COVID_19_DIR", "")

    DATA_DIR = PROJECT_DIR / "input/data"
    CHEXPERT_TRAIN_LMDB = DATA_DIR / "chexpert_train"
    CHEXPERT_TEST_LMDB = DATA_DIR / "chexpert_test"
    COVID_19_TRAIN_LMDB = DATA_DIR / "covid_train"
    COVID_19_TEST_LMDB = DATA_DIR / "covid_test"

    # Output
    OUTPUT_DIR = Path(PROJECT_DIR / "output" / EXPERIMENT_NAME)

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
    STYLEGAN_ITER = int(ENV.get("STYLEGAN_ITER", ""))
    STYLEGAN_BATCH = int(ENV.get("STYLEGAN_BATCH", ""))
    STYLEGAN_SAMPLES = int(ENV.get("STYLEGAN_SAMPLES", ""))
    STYLEGAN_LR = float(ENV.get("STYLEGAN_LR", ""))
    STYLEGAN_MIXING = float(ENV.get("STYLEGAN_MIXING", ""))
    STYLEGAN_CKPT = env_get("STYLEGAN_CKPT", None)

    # StyleGAN Regularization
    STYLEGAN_R1 = float(ENV.get("STYLEGAN_R1", ""))
    STYLEGAN_PATH_REG = float(ENV.get("STYLEGAN_PATH_REG", ""))
    STYLEGAN_PATH_BATCH_SHRINK = int(ENV.get("STYLEGAN_PATH_BATCH_SHRINK", ""))
    STYLEGAN_D_REG_INTERVAL = float(ENV.get("STYLEGAN_D_REG_INTERVAL", ""))
    STYLEGAN_G_REG_INTERVAL = float(ENV.get("STYLEGAN_G_REG_INTERVAL", ""))

    # pSp Model
    PSP_ENCODER: str = env_or("PSP_ENCODER", "original", "deep")
    PSP_USE_MEAN: bool = to_bool("PSP_USE_MEAN")

    # pSp Training
    PSP_ITER = int(ENV.get("PSP_ITER", 0))
    PSP_BATCH_SIZE = int(ENV.get("PSP_BATCH_SIZE", 0))
    PSP_OPTIM: str = env_or("PSP_OPTIM", "ranger", "adam")
    PSP_LR = float(ENV.get("PSP_LR", 0))
    PSP_CKPT = Path(ENV.get("PROJECT_DIR", "")) / ENV.get("PSP_CKPT", "")

    PSP_SAMPLE_INTERVAL = int(ENV.get("PSP_SAMPLE_INTERVAL", 0))
    PSP_TEST_INTERVAL = int(ENV.get("PSP_TEST_INTERVAL", 0))
    PSP_CKPT_INTERVAL = int(ENV.get("PSP_CKPT_INTERVAL", 0))

    # pSp Training
    PSP_USE_LOCALIZER = to_bool("PSP_USE_LOCALIZER")
    PSP_LOCALIZER_WEIGHT = float(ENV.get("PSP_LOCALIZER_WEIGHT", 0))
    PSP_LOSS_L2 = float(ENV.get("PSP_LOSS_L2", 0))
    PSP_LOSS_ID = float(ENV.get("PSP_LOSS_ID", 0))
    PSP_LOSS_ID_DISCRIMINATOR = float(ENV.get("PSP_LOSS_ID_DISCRIMINATOR", 0))
    PSP_LOSS_LPIPS = float(ENV.get("PSP_LOSS_LPIPS", 0))
    PSP_LOSS_REG = float(ENV.get("PSP_LOSS_REG", 0))
    PSP_LOSS_DISCRIMINATOR = float(ENV.get("PSP_LOSS_DISCRIMINATOR", 0))


(CONFIG.OUTPUT_DIR / "sample").mkdir(parents=True, exist_ok=True)
(CONFIG.OUTPUT_DIR / "checkpoint").mkdir(parents=True, exist_ok=True)
(CONFIG.OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)

sh.copyfile(str(env_file), str(CONFIG.OUTPUT_DIR / f"{CONFIG.EXPERIMENT_NAME}.env"))


def guard():
    pass
