"""
Covid-CT dataset utility functions
"""

from typing import Any

import numpy as np
import pandas as pd


def parse_subject(path: str, *_: Any) -> str:
    """
    Extracts subject name from DICOM path.
    """

    return path.split("/")[-2].replace("subject_", "")


def classify_images(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifies CTs into localizer, lung and mediastinum in column "Type".
    Filters out other types.
    """

    is_localizer = df["ScanOptions"] == "SURVIEW"
    is_lung = df["SeriesDescription"].str.lower().str.contains("lung", na=False)
    is_med = df["SeriesDescription"].str.lower().str.contains("med", na=False)
    df["Type"] = np.where(
        is_localizer,
        "localizer",
        np.where(is_lung, "lung", np.where(is_med, "med", "others")),
    )
    df = df.query("Type != 'others'").reset_index(drop=True)
    return df


def filter_all_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selects subjects with lung and med and localizer views.
    """

    has_all_types = df.groupby("Subject").apply(lambda x: x["Type"].nunique() == 3)
    subjects = list(has_all_types[has_all_types].index)
    print(
        f"{len(subjects)}/{df['Subject'].nunique()} subjects has lung + med + localizer"
    )
    df = df[df["Subject"].isin(subjects)]
    return df.reset_index(drop=True)
