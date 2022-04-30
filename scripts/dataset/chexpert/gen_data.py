"""
Generates CheXpert dataset.
"""

from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple, cast

import numpy as np
import pandas as pd
from scripts.dataset.chexpert.utils import CheXpertImg
from tqdm import tqdm

from utils import save_img
from utils.cli import OPTIONS, save_options
from utils.cli.dataset import DatasetGen, DatasetOptions
from utils.dataset import LMDBImageWriter

DATASET_OPTIONS = cast(DatasetOptions, OPTIONS.arch)
GEN_OPTIONS = cast(DatasetGen, DATASET_OPTIONS.cmd)


def proc_img(
    img_dir: Path,
    img: CheXpertImg,
) -> Tuple[CheXpertImg, np.ndarray]:
    return img, img.proc_img(GEN_OPTIONS.resolution, img_dir)


def proc_df(df: pd.DataFrame, img_dir: Path, lmdb_dir: Path):
    # Turn dataframe rows into metadata object
    imgs = [
        CheXpertImg(idx, path, sex, age, others)
        for idx, path, sex, age, *others in tqdm(
            df.itertuples(), dynamic_ncols=True, smoothing=0.01
        )
    ]
    print(f"Got {len(imgs)} images.")

    if not GEN_OPTIONS.png:
        writer = LMDBImageWriter(lmdb_dir)
        writer.set_length(len(imgs))

        with Pool(4) as pool:
            for img, val in tqdm(
                pool.imap_unordered(partial(proc_img, img_dir), imgs),
                total=len(imgs),
                dynamic_ncols=True,
                smoothing=0.01,
            ):
                writer.set_img(img.idx, val)
    else:
        (OPTIONS.output_dir / "imgs").mkdir(parents=True, exist_ok=True)

        with Pool(4) as pool:
            for img, val in tqdm(
                pool.imap_unordered(partial(proc_img, img_dir), imgs),
                total=len(imgs),
                dynamic_ncols=True,
                smoothing=0.01,
            ):
                save_img(val, OPTIONS.output_dir / f"imgs/{str(img.idx).zfill(10)}.png")


def gen_chexpert_data():
    save_options()

    # Read train.csv and valid.csv
    train_df = cast(pd.DataFrame, pd.read_csv(DATASET_OPTIONS.path / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATASET_OPTIONS.path / "valid.csv"))
    df = pd.concat([train_df, test_df])

    # Remove root path of extracted folder
    df["Path"] = df["Path"].str.replace("CheXpert-v1.0/", "", regex=False)
    df["Support Devices"] = df["Support Devices"].fillna(-1)

    # Select Frontal AP CXR without Support Devices
    interested_df = df[
        (df["Frontal/Lateral"] == "Frontal")
        & (df["AP/PA"] == "AP")
        & (df["Support Devices"] <= 0)
    ]
    interested_df = interested_df.drop(
        ["Frontal/Lateral", "AP/PA", "No Finding"], axis=1
    )

    # Drop patients without path / sex / age and fill in missing pathlogies as -1
    interested_df = (
        interested_df.dropna(subset=["Path", "Sex", "Age"])
        .fillna(-1)
        .reset_index(drop=True)
    )
    print(f"Total of {interested_df.shape[0]} images left")

    pd.to_pickle(interested_df, str(OPTIONS.output_dir / "metadata.pkl"))

    proc_df(interested_df, DATASET_OPTIONS.path, OPTIONS.output_dir / "lmdb")
