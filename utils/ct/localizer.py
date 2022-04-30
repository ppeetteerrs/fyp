"""
Localizer functions.
"""

from typing import Tuple

import cusignal as cs
import cv2 as cv
import numpy as np
from utils import Arr32F
from utils.cv import project


def get_feature(img_2d: Arr32F, percentage: int) -> Arr32F:
    """
    Extracts the brightest N percent of pixels from image

    Args:
        img_2d (Arr32F): 2D Image
        percentage (int): Percentage of pixels to extract

    Returns:
        Arr32F: Extracted pixels
    """

    return np.where(img_2d < np.percentile(img_2d, 100 - percentage), 0, img_2d)


def align_features(
    sub_feature_2db: Arr32F,
    base_feature_2db: Arr32F,
) -> Tuple[int, int, int, int]:
    """
    Calculates the corner coordinates of sub_feature_2db in
    base_feature_2db image by maximizing cross-correlation

    Args:
        sub_feature_2db (Arr32F): Feature to be found
        base_feature_2db (Arr32F): Image containing feature to be found

    Returns:
        Tuple[int, int, int, int]: row_start, row_end, col_start, col_end
    """

    # Calculate cross correlation
    sub_feature_2db = sub_feature_2db.astype(float)
    h, w = sub_feature_2db.shape
    sub_feature_2db[: h // 2, :] *= 3

    corr = cs.correlate2d(
        base_feature_2db.astype(float), sub_feature_2db, mode="same"
    ).get()

    # Get maximum cross correlation coordinates (sub_feature_2db relative to base_iamge)
    indices = np.unravel_index(corr.argmax(), base_feature_2db.shape)
    row_start, col_start = int(indices[0]), int(indices[1])
    row_start = max(row_start - sub_feature_2db.shape[0] // 2, 0)
    col_start = max(col_start - sub_feature_2db.shape[1] // 2, 0)
    row_end = row_start + sub_feature_2db.shape[0]
    col_end = col_start + sub_feature_2db.shape[1]
    return row_start, row_end, col_start, col_end


def crop_localizer(
    localizer: Arr32F, crop: Tuple[int, int, int, int]
) -> Tuple[Arr32F, Arr32F]:
    """
    Crops the localizer according to the corner coordinates and returns
    an RGB image showing the bounding box

    Args:
        localizer (Arr32F): Localizer image
        crop (Tuple[int, int, int, int]): Corner coordinates

    Returns:
        Tuple[Arr32F, Arr32F]: Cropped localizer, localizer with bounding box
    """

    row_start, row_end, col_start, col_end = crop
    cropped = localizer[row_start:row_end, col_start:col_end]
    rgb = cv.rectangle(
        cv.cvtColor(localizer, cv.COLOR_GRAY2RGB),
        (col_start, row_start),
        (col_end, row_end),
        (0, 1, 0),
        3,
    )
    return cropped, rgb


def get_localizer(
    lung_3d: Arr32F, loc_2d: Arr32F
) -> Tuple[Arr32F, Arr32F, Arr32F, Arr32F, Arr32F]:
    """
    Align localizer scan with CT projection

    Args:
        ct_projection (Arr32F): CT volume naiive projection
        localizer (Arr32F): Localizer scan

    Returns:
        Tuple[Arr32F, Arr32F, Arr32F, Arr32F, Arr32F]:
        cropped localizer, localizer with bounding box, CT feature, localizer feature
    """
    lung_2d = project(lung_3d)
    ct_feature = get_feature(lung_2d, 40)
    localizer_feature = get_feature(loc_2d, 20)
    indices = align_features(ct_feature, localizer_feature)
    cropped, rgb = crop_localizer(loc_2d, indices)
    return lung_2d, ct_feature, localizer_feature, rgb, cropped
