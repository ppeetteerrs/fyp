"""
Parses Covid-CT dataset metadata.
"""

from pathlib import Path
from typing import cast

from scripts.dataset.covid_ct.utils import (
    classify_images,
    filter_all_types,
    parse_subject,
)

from utils import load
from utils.cli import OPTIONS, save_options
from utils.cli.dataset import DatasetOptions, DatasetParse
from utils.dicom import DicomMetaParser

DATASET_OPTIONS = cast(DatasetOptions, OPTIONS.arch)
PARSE_OPTIONS = cast(DatasetParse, DATASET_OPTIONS.cmd)


def parse_covid_ct_meta():
    save_options()

    parser = DicomMetaParser(DATASET_OPTIONS.path, parse_subject)

    df = parser.parse()
    df.to_pickle(str(OPTIONS.output_dir / "metadata.pkl"))

    # Uncomment to check dataframe column properties
    parser.check_df(df)

    subject_info = load(
        OPTIONS.project_dir / "scripts/dataset/covid_ct/subject_info.pkl"
    )

    # Remove bad (blurry / movement artifacts) and negative subjects
    df = df[
        (~df["Subject"].isin(subject_info["bad"]))
        & (df["Subject"].isin(subject_info["positive"]))
    ]

    # Classify CTs into lung vs med vs localizer Types
    df = classify_images(df)

    # Select DICOM series (based on SeriesNumber) with minimum slice thickness from each CT Type
    df = parser.select_min(df, sub_category=["Type"])

    # Only select subjects with all 3 (lung, med, localizer) types of CT
    df = filter_all_types(df)

    # Check that specified fields are consistent within each DICOM Series
    df = parser.check_series_consistency(
        df,
        [
            "ImageOrientationPatient",
            "SeriesDescription",
            "PixelSpacing",
            "ImageType",
        ],
    )

    # Extract file paths for each unique DICOM series (based on Subject and Type)
    paths_df = parser.to_path_list(df)
    paths_df.to_pickle(str(OPTIONS.output_dir / "paths.pkl"))
