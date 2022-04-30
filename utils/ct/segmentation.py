"""
CT segmentation functions.
"""

from functools import partial
from multiprocessing import Pool
from typing import Any, Tuple

import cv2 as cv
import numpy as np
import SimpleITK as sitk
from lungmask import mask, utils
from utils import Arr8U, Arr32F, ArrB, nop
from utils.cv import CONTOUR_IDX, denoise

# Make lungmask tqdm quiet
mask.tqdm = nop
utils.tqdm = nop


def get_body_mask(img_2d: Arr8U, body_threshold: int) -> ArrB:
    """
    Extracts body mask (excludes bed) from CT slice.

    Args:
        img (Arr8U): CT slice
        body_threshold (int): Lower threshold for body

    Returns:
        ArrB: body_mask, threshold, biggest contour
    """

    # Threshold and remove small connections to bed
    threshold = cv.threshold(img_2d, body_threshold, 255, cv.THRESH_BINARY)[1]
    img_2d = denoise(img_2d, erode_iter=3, dilate_iter=3)

    # Find largest contour and fill
    contours = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[
        CONTOUR_IDX
    ]
    contour = None
    if len(contours) > 0:
        contour = max(contours, key=cv.contourArea)
    contour_img = np.zeros_like(img_2d)
    if contour is not None:
        cv.drawContours(contour_img, [contour], 0, 255, cv.FILLED)

    # Clean contour
    mask = denoise(contour_img, erode_iter=3, dilate_iter=3)

    return mask.astype(np.bool8)


def get_bone_mask(img_2d: Arr8U, bone_threshold: int) -> ArrB:
    """
    Extracts bones from CT slice.

    Args:
        img_2d (Arr8U): CT slice
        bone_threshold (int): Lower threshold for bones

    Returns:
        ArrB: bones, threshold, filled contours
    """

    # Apply threshold to extract bone-like structures
    threshold = np.where(img_2d < bone_threshold, 0, 255).astype(np.uint8)

    # Fill all contours (i.e. fill in bone marrow?!)
    contours = cv.findContours(threshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[
        CONTOUR_IDX
    ]
    contour_img = np.zeros_like(threshold)

    if len(contours) > 0:
        cv.drawContours(contour_img, contours, -1, 255, cv.FILLED)

    # Denoise the image (remove small dots)
    bones = denoise(contour_img, kernel_size=2)

    return bones.astype(np.bool8)


def segment(
    sitk_img: sitk.Image,
    img: Arr32F,
    *,
    lung: bool = True,
    body: bool = True,
    bones: bool = True,
    body_threshold: int = 35,
    bone_threshold: int = 70,
    seg_model: Any = None,
) -> Tuple[ArrB, ArrB, ArrB]:
    """
    Segments a CT image.

    Args:
        sitk_img (sitk.Image): SITK CT image.
        img (Arr32F): Equivalent 3D numpy array obtained from `read_dcm`.
        lung (bool, optional): Extracts lung mask (U-Net). Defaults to True.
        body (bool, optional): Extracts body mask (bed removal through thresholding). Defaults to True.
        bones (bool, optional): Extracts bones mask (thresholding). Defaults to True.
        body_threshold (int, optional): Minimum 8-bit threshold (not Hounsfield) for human tissues. Defaults to 35.
        bone_threshold (int, optional): Minimum 8-bit threshold (not Hounsfield) for boness. Defaults to 70.
        seg_model (Any, optional): Lung segmentation model to use. Defaults to None.

    Returns:
        Tuple[ArrB, ArrB, ArrB]: Extracted lung, body and bones masks.
    """

    img_8bits = (img * 255).astype(np.uint8)

    if lung:
        lung_mask = mask.apply(sitk_img, model=seg_model, batch_size=1) > 0
    else:
        lung_mask = np.ones_like(img_8bits).astype(bool)

    get_body = partial(get_body_mask, body_threshold=body_threshold)
    get_bones = partial(get_bone_mask, bone_threshold=bone_threshold)

    with Pool(4) as pool:
        if body:
            body_mask = np.asarray(pool.map(get_body, img_8bits))
        else:
            body_mask = np.ones_like(img_8bits).astype(bool)
        if bones:
            bones_mask = np.asarray(pool.map(get_bones, img_8bits * body_mask))
        else:
            bones_mask = np.ones_like(img_8bits).astype(bool)
    return lung_mask, body_mask, bones_mask
