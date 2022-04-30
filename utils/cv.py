"""
Computer Vision functions.
"""

from typing import Collection, List, Optional, Tuple, Union

import cv2 as cv
import numpy as np

from utils import AnyArr, Arr8U, Arr32F

CONTOUR_IDX = 0 if int(cv.__version__[0]) >= 4 else 1


def remove_border(
    img: AnyArr,
    axes: Optional[Collection[int]] = None,
    tol: float = 0,
    reference: Optional[Arr32F] = None,
) -> AnyArr:
    """
    Removes border from an nd-array.
    """

    if reference is None:
        mask = img > tol
    else:
        mask = reference > tol

    # Perform cropping along each axis
    for idx, dim in enumerate(img.shape):
        if axes is not None and idx in axes:
            continue
        other_dims = tuple(i for i in range(len(img.shape)) if i != idx)
        dim_mask = mask.any(other_dims)
        img = np.take(img, range(dim_mask.argmax(), dim - dim_mask[::-1].argmax()), idx)
    return img


def crop(img: AnyArr, size: Optional[Union[Tuple[int, int], int]] = None) -> AnyArr:
    """
    Center-crops and resizes an image

    Args:
        img (AnyArr): Input image
        size (Union[Tuple[int, int], int], optional): Height, width. Defaults to None.

    Returns:
        AnyArr: Output image
    """
    curr_h, curr_w = new_h, new_w = img.shape

    # Calculate new height and width
    if size is not None:
        if isinstance(size, int):
            new_h = new_w = size
        else:
            new_h, new_w = size

    # Height and width to crop to
    crop_h = int(min(curr_h, curr_w / new_w * new_h))
    crop_w = int(crop_h / new_h * new_w)

    row_start = (curr_h - crop_h) // 2
    col_start = (curr_w - crop_w) // 2

    img = img.astype(np.float32)
    return cv.resize(
        img[row_start : row_start + crop_h, col_start : col_start + crop_w],
        (new_w, new_h),
    )


def denoise(
    img: Arr8U,
    kernel_size: int = 3,
    erode_iter: int = 1,
    dilate_iter: int = 1,
    total_iter: int = 1,
) -> Arr8U:
    """
    Denoise image through erosion-dilation cycles. Input and output are expected to be of type np.uint8.
    """

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    for _ in range(total_iter):
        img = cv.erode(img, kernel, iterations=erode_iter)
        img = cv.dilate(img, kernel, iterations=dilate_iter)
    return img


def min_max_normalize(
    img: Arr32F,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
) -> Arr32F:
    """
    Performs min-max normalization.
    """

    if lower is None:
        lower = float(img.min())

    if upper is None:
        upper = float(img.max())

    if upper - lower > 0:
        res = np.true_divide((img - lower), (upper - lower))
    else:
        return np.zeros_like(img)

    return res.astype(img.dtype)


def project(
    img_3d: Union[List[Arr32F], Arr32F],
    *,
    remove_border_axes: Optional[List[int]] = [1],
    remove_border_ref: Optional[Arr32F] = None,
    crop_size: Optional[Union[int, Tuple[int, int]]] = None,
    normalize: bool = True,
    axis: int = 1,
) -> Arr32F:
    """
    Simple projection of 3D volume to 2D view along an axis.
    """

    if not isinstance(img_3d, np.ndarray):
        img_3d = np.array(img_3d)

    img_3d = remove_border(img_3d, remove_border_axes, reference=remove_border_ref)

    img_2d = np.mean(img_3d, axis=axis)

    if crop_size is not None:
        img_2d = crop(img_2d, size=crop_size)

    if normalize:
        img_2d = min_max_normalize(img_2d)

    return img_2d
