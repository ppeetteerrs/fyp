import pickle
from random import Random
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from parse_data.covid_ct.bad_subjects import BAD, MISALIGNED
from utils.config import COVID_19_DIR


def check_unique(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check for columns with only 1 value and remove them
    """
    nunique = df.nunique(dropna=False)
    to_drop = []
    for col in df.columns:
        if nunique[col] == 1:
            print(f"Column {col} has only 1 value, dropping column...")
            to_drop.append(col)
    return df.drop(to_drop, axis=1).reset_index(drop=True)


def check_series_consistency(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    Check that each subject, series pair has only 1 value in selected columns, remove violating entries
    """
    nunique = df.groupby(["Subject", "SeriesNumber"])[cols].nunique(dropna=False)
    pairs: List[Tuple[str, int]] = nunique.index.values.tolist()
    bad_pairs: List[Tuple[str, int]] = []
    for subject, idx in pairs:
        pair_df = nunique.loc[(subject, idx)]
        bad_fields = pair_df[pair_df != 1].index.values.tolist()
        if len(bad_fields) > 0:
            print(
                f"Subject {subject} series {idx} has non-unique values in {bad_fields}"
            )
            bad_pairs.append((subject, idx))
    tmp_df = df.set_index(["Subject", "SeriesNumber"])
    return tmp_df[~tmp_df.index.isin(bad_pairs)].reset_index()


def classify_images(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify images into localizer, lung and mediastinum. Filter out other types
    """
    is_localizer = df["ImageType2"] == "LOCALIZER"
    is_lung = df["SeriesDescription"].str.contains("lung", na=False)
    is_med = df["SeriesDescription"].str.contains("med", na=False)
    df["Type"] = np.where(
        is_localizer,
        "localizer",
        np.where(is_lung, "lung", np.where(is_med, "med", "others")),
    )
    df = df.query("Type != 'others'").reset_index(drop=True)
    return df


def select_min(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each subject, type pair, select series with minimum slice thickness
    """
    min_pairs = df.iloc[df.groupby(["Subject", "Type"])["SliceThickness"].idxmin()][
        ["Subject", "SeriesNumber"]
    ].values.tolist()
    tmp_df = df.set_index(["Subject", "SeriesNumber"])
    return tmp_df[tmp_df.index.isin(min_pairs)].reset_index()


def filter_all_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select subjects with lung and med and localizer views
    """
    has_all_types = df.groupby("Subject").apply(lambda x: x["Type"].nunique() == 3)
    subjects = list(has_all_types[has_all_types].index)
    print(
        f"{len(subjects)}/{df['Subject'].nunique()} subjects has lung + med + localizer"
    )
    df: Any = df[df["Subject"].isin(subjects)]
    return df.reset_index(drop=True)


if __name__ == "__main__":
    # Read data and remove weird entry
    df: Any = pd.read_csv(COVID_19_DIR / "meta.csv")
    df = df.query("Subject != 133")
    df = df[(~df["Subject"].isin(BAD)) & (~df["Subject"].isin(MISALIGNED))]
    df = check_unique(df)
    df = check_series_consistency(
        df,
        [
            "ImagePositionPatient0",
            "ImagePositionPatient1",
            "ImageOrientationPatient",
            "SeriesDescription",
            "PixelSpacing0",
            "PixelSpacing1",
            "ImageType2",
            "ImageType3",
        ],
    )
    df = classify_images(df)
    df = select_min(df)
    df = filter_all_types(df)

    # Save info for each subject
    groups = df.groupby(["Subject"])
    img_types = ["lung", "med", "localizer"]
    paths: List[Dict[str, Union[List[str], str]]] = []
    for i, (subject, group_df) in enumerate(groups):
        sorted_df = group_df.sort_values(by="SliceLocation")
        info = {
            "index": i,
            "subject": subject,
        }
        for img_type in img_types:
            img_paths = sorted_df[sorted_df["Type"] == img_type]["Path"].tolist()
            info[img_type] = img_paths

        paths.append(info)

    # Shuffle paths
    Random(1035).shuffle(paths)
    print(f"Found {len(paths)} valid subjects.")

    # Separate info train and test sets
    train_infos = [{**info, "index": i} for i, info in enumerate(paths[:680])]
    test_infos = [{**info, "index": i} for i, info in enumerate(paths[680:])]
    pickle.dump(train_infos, open(COVID_19_DIR / "train_infos.pkl", "wb"))
    pickle.dump(test_infos, open(COVID_19_DIR / "test_infos.pkl", "wb"))
