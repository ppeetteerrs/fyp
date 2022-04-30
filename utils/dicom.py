"""
DICOM utility functions
"""

from functools import partial
from glob import glob
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from pydicom.multival import MultiValue
from pydicom.valuerep import IS, DSdecimal, DSfloat
from tqdm.contrib.concurrent import process_map

from utils import Arr32F

unhashable: Dict[str, None] = {}


class DicomMetaParser:
    def __init__(
        self,
        dir_or_paths: Union[str, Path, List[Union[str, Path]]],
        info_to_subject: Callable[[str, Dict[str, Any]], str],
        *,
        n_workers: Optional[int] = None,
    ):
        """
        Initiates a DICOM metadata parser.

        Args:
            dir_or_paths (Union[str, Path, List[Union[str, Path]]]): A directory containing DICOM files or a list of DICOM file paths.
            info_to_subject (Callable[[str, Dict[str, Any]], str]): A function to map (file_path, file_metadata) to a subject name.
            n_workers (Optional[int], optional): Maximum number of parallel workers. Defaults to min(32, n_cpu + 4).
        """

        if isinstance(dir_or_paths, (str, Path)):
            self.dcm_paths = list(glob(f"{dir_or_paths}/**/**.dcm", recursive=True))
        else:
            self.dcm_paths = [str(path) for path in dir_or_paths]

        self.n_workers = n_workers
        self.df = None
        self.info_to_subject = info_to_subject

    def parse(self, *, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Parses the metadata from all DICOM files and returns a dataframe of non-empty attributes.

        Args:
            limit (Optional[int], optional): Number of files to parse, useful for debugging. Defaults to None.

        Returns:
            pd.DataFrame: Metadata
        """

        if limit is not None:
            dcm_paths = self.dcm_paths[:limit]
        else:
            dcm_paths = self.dcm_paths

        df = pd.DataFrame.from_records(
            process_map(
                partial(
                    DicomMetaParser.parse_item, info_to_subject=self.info_to_subject
                ),
                dcm_paths,
                chunksize=1,
                desc="Parsing DICOM metadata",
                max_workers=self.n_workers,
            )
        )
        self.df = df.replace("", np.nan).dropna(axis=1, how="all")
        return self.df

    @staticmethod
    def check_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates a report of DICOM attributes (whether an attribute is unique / what are the different values).
        """

        pd.options.display.max_columns = 999
        pd.options.display.max_rows = 999
        pd.options.display.max_colwidth = None

        report_items: List[Dict[str, Any]] = []

        nunique = df.nunique(dropna=False)
        to_drop = []
        for col in df.columns:
            if nunique[col] == 1:
                report_items.append(
                    {
                        "Category": "Single Value",
                        "Column Name": col,
                        "Value(s)": df[col].unique()[0],
                    }
                )
                to_drop.append(col)
            elif nunique[col] < 10:
                report_items.append(
                    {
                        "Category": "Multiple Values",
                        "Column Name": col,
                        "Value(s)": df[col].unique(),
                    }
                )
            else:
                report_items.append(
                    {
                        "Category": "Many Values (> 10)",
                        "Column Name": col,
                        "Value(s)": "-",
                    }
                )

        return (
            pd.DataFrame.from_records(report_items)
            .groupby(["Category", "Column Name"], sort=True)
            .first()
        )

    @staticmethod
    def parse_item(
        dcm_path: str, info_to_subject: Callable[[str, Dict[str, Any]], str]
    ) -> Dict[str, Any]:
        """
        Parses a DICOM file into a dictionary containing its metadata.

        Args:
            dcm_path (str): Path to DICOM file.
            info_to_subject (Callable[[str, Dict[str, Any]], str]): Mapping from (path, metdata) to subject name.

        Returns:
            Dict[str, Any]: Metadata
        """

        data = pydicom.dcmread(dcm_path)
        info = {
            k: DicomMetaParser.parse_pydicom_field(data, k)
            for k in data.dir()
            if k != "PixelData"
        }
        info["Path"] = dcm_path
        info["Subject"] = info_to_subject(dcm_path, info)
        return info

    @staticmethod
    def parse_pydicom_field(data: pydicom.Dataset, key: str) -> Any:
        """
        Parses pydicom MultiValue (unhashable) into Python primitive types.
        """

        global unhashable

        if key in unhashable:
            return np.nan

        try:
            value = getattr(data, key)
        except:
            return np.nan

        if isinstance(value, MultiValue):
            # Will get represented as strings if not parsed
            if value.type_constructor in [DSfloat, DSdecimal]:
                value = tuple(float(i) for i in value)
            elif value.type_constructor == IS:
                value = tuple(int(i) for i in value)
            else:
                value = tuple(value)
        elif isinstance(value, bytes):
            value = np.nan

        try:
            hash(value)
        except:
            unhashable[key] = None
            print(f"Field: {key} is unhashable, setting to nan...")
            return np.nan

        return value

    @staticmethod
    def check_series_consistency(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """
        Checks that each subject, series pair has only 1 value in selected columns, remove violating entries.
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

    @staticmethod
    def select_min(
        df: pd.DataFrame,
        *,
        sub_category: List[str] = [],
        field: str = "SliceThickness",
    ) -> pd.DataFrame:
        """
        For each (subject, sub_category) pair, select series with minimum "field" value.
        """

        min_df: Any = df.iloc[df.groupby(["Subject"] + sub_category)[field].idxmin()]
        min_pairs = min_df[["Subject", "SeriesNumber"]].values.tolist()
        tmp_df = df.set_index(["Subject", "SeriesNumber"])
        return tmp_df[tmp_df.index.isin(min_pairs)].reset_index()

    @staticmethod
    def to_path_list(
        df: pd.DataFrame, *, field: str = "Type", sort_by: str = "SliceLocation"
    ) -> pd.DataFrame:
        """
        For each (subject, sub_category) pair, gets the paths to all DICOM files.
        """

        # Save info for each subject
        groups = df.groupby(["Subject"])
        field_values: List[str] = df[field].unique().tolist()
        paths: List[Dict[str, Union[List[str], str]]] = []
        for _, (subject, group_df) in enumerate(groups):
            sorted_df = group_df.sort_values(by=sort_by)
            info: Dict[str, Any] = {
                "Subject": subject,
            }
            for field_value in field_values:
                img_paths = sorted_df[sorted_df["Type"] == field_value]["Path"].tolist()
                info[field_value] = tuple(img_paths)

            paths.append(info)
        return pd.DataFrame.from_records(paths)


def read_dcm(
    paths: List[str],
    spacing: Optional[Tuple[int, int, int]] = (1, 1, 1),
    clip: bool = False,
) -> Tuple[sitk.Image, Arr32F]:
    """
    Reads a DICOM series and returns a resampled output.
    Input DICOM must have Hounsfield range [-1024, 3071].
    Numpy output is converted into [0, 4095].

    Args:
        paths (List[str]): File paths to slices in DICOM series.
        spacing (Optional[Tuple[int, int, int]], optional): X, Y, Z pixel spacing in mm. Defaults to (1, 1, 1).
        clip (bool, optional): Clip numpy data range to [-1024, 3071]. Defaults to False.

    Returns:
        Tuple[sitk.Image, Arr32F]: SITK image and numpy image.
    """

    # Read image using sitk
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(paths)
    image: sitk.Image = reader.Execute()

    # Resample image to new pixel spacing
    if spacing is not None:
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()
        new_size = [
            int(i * j / k) for i, j, k in zip(original_size, original_spacing, spacing)
        ]
        image = sitk.Resample(
            image1=image,
            size=new_size,
            transform=sitk.Transform(),
            interpolator=sitk.sitkLinear,
            outputOrigin=image.GetOrigin(),
            outputSpacing=spacing,
            outputDirection=image.GetDirection(),
            defaultPixelValue=0,
            outputPixelType=image.GetPixelID(),
        )
    np_img = sitk.GetArrayFromImage(image)

    if clip:
        np_img[np_img > 3071] = 3071
        np_img[np_img < -1024] = -1024

    assert np.all(np_img >= -1024) and np.all(
        np_img < 3072
    ), f"DICOM series has Hounsfield values beyond [-1024, 3071], Max: {np_img.max()}, Min: {np_img.min()}. Paths: {paths}"

    np_img = np.true_divide(np_img + 1024, 4095).astype(np.float32)

    return image, np_img
