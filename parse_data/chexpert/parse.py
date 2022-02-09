from functools import partial
from multiprocessing import Pool
from pathlib import Path
from pickle import dump
from typing import Tuple, cast

import numpy as np
import pandas as pd
from parse_data.chexpert.utils import CheXpertImg
from tqdm import tqdm

from utils.config import CONFIG
from utils.lmdb import LMDBImageWriter, chexpert_indexer

CONFIG.CHEXPERT_TRAIN_LMDB.mkdir(parents=True, exist_ok=True)
CONFIG.CHEXPERT_TEST_LMDB.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = CONFIG.CHEXPERT_DIR / "train.csv"
TEST_CSV = CONFIG.CHEXPERT_DIR / "valid.csv"


def proc_img(
    img_dir: Path,
    img: CheXpertImg,
) -> Tuple[CheXpertImg, np.ndarray]:
    return img, img.proc_img(CONFIG.RESOLUTION, img_dir)


def proc_df(df: pd.DataFrame, img_dir: Path, lmdb_dir: Path):
    # Turn dataframe rows into metadata object
    imgs = [
        CheXpertImg(idx, path, sex, age, others)
        for idx, path, sex, age, *others in tqdm(
            df.itertuples(), dynamic_ncols=True, smoothing=0.01
        )
    ]
    print(f"Got {len(imgs)} images.")

    writer = LMDBImageWriter(lmdb_dir, chexpert_indexer)

    print(img_dir)

    with Pool(6) as pool:
        for img, val in tqdm(
            pool.imap_unordered(partial(proc_img, img_dir), imgs),
            total=len(imgs),
            dynamic_ncols=True,
            smoothing=0.01,
        ):
            writer.set_idx(img.idx, [val])

    writer.set_int("length", len(imgs))

    dump(imgs, open(lmdb_dir / "meta.pkl", "wb"))


if __name__ == "__main__":
    # Read train.csv and valid.csv
    train_df = cast(pd.DataFrame, pd.read_csv(TRAIN_CSV))
    test_df = cast(pd.DataFrame, pd.read_csv(TEST_CSV))
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
    interested_df = interested_df.dropna(subset=["Path", "Sex", "Age"]).fillna(-1)
    print(f"Total of {interested_df.shape[0]} images left")

    # Shuffle dataframe
    interested_df = (
        interested_df.sample(frac=1, random_state=932)
        .reset_index(drop=True)
        .iloc[:62000, :]
    )

    # Split dataframe into train and test set
    train_df = interested_df.iloc[:55000, :].copy(deep=True).reset_index(drop=True)
    test_df = interested_df.iloc[55000:, :].copy(deep=True).reset_index(drop=True)

    print("Processing training images...")
    proc_df(train_df, CONFIG.CHEXPERT_DIR, CONFIG.CHEXPERT_TRAIN_LMDB)

    print("Processing test images...")
    proc_df(test_df, CONFIG.CHEXPERT_DIR, CONFIG.CHEXPERT_TEST_LMDB)
