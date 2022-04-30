"""
Parses LIDC dataset metadata.
"""

from typing import cast

from scripts.dataset.lidc.utils import parse_subject

from utils import load
from utils.cli import OPTIONS, save_options
from utils.cli.dataset import DatasetOptions, DatasetParse
from utils.dicom import DicomMetaParser

DATASET_OPTIONS = cast(DatasetOptions, OPTIONS.arch)
PARSE_OPTIONS = cast(DatasetParse, DATASET_OPTIONS.cmd)


def parse_lidc_meta():
    save_options()

    parser = DicomMetaParser(DATASET_OPTIONS.path, parse_subject, n_workers=4)

    df = parser.parse()
    df.to_pickle(str(OPTIONS.output_dir / "metadata.pkl"))

    # Uncomment to check dataframe column properties
    parser.check_df(df)

    subject_info = load(OPTIONS.project_dir / "scripts/dataset/lidc/subject_info.pkl")

    # Remove bad (blurry / movement artifacts) and negative subjects
    df = df[~df["Subject"].isin(subject_info["bad"])]

    df = df.query(
        "PatientPosition == 'FFS' & PhotometricInterpretation == 'MONOCHROME2'"
    )
    is_primary = df.apply(
        lambda x: x["ImageType"][0] == "ORIGINAL" and x["ImageType"][1] == "PRIMARY",
        axis=1,
    )
    df = df[is_primary].reset_index(drop=True)

    # Select DICOM series (based on SeriesNumber) with minimum slice thickness from each CT Type
    df = parser.select_min(df, sub_category=[])

    # Check that specified fields are consistent within each DICOM Series
    df = parser.check_series_consistency(
        df,
        [
            "ImageOrientationPatient",
            "SeriesDescription",
            "PixelSpacing",
        ],
    )

    df["Type"] = "lung"

    # Extract file paths for each unique DICOM series (based on Subject and Type)
    paths_df = parser.to_path_list(df, sort_by="Path")
    paths_df.to_pickle(str(OPTIONS.output_dir / "paths.pkl"))
