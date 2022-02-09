import pickle
from io import BytesIO
from typing import Dict, List, Optional, Tuple, Union

import cusignal as cs
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from utils.config import CONFIG
from utils.img import center_crop
from utils.lmdb import LMDBImageWriter, covid_ct_indexer_lung


# Add normalization param, remove blank stacks, use colour channels to pass info
def get_img(paths: List[str], stack_axis=0) -> Tuple[np.ndarray, np.ndarray]:
    # Read image using sitk
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(paths)
    image: sitk.Image = reader.Execute()

    # Resample image to new pixel spacing
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    new_spacing = [1, 1, 1]
    new_size = [
        int(i * j / k) for i, j, k in zip(original_size, original_spacing, new_spacing)
    ]
    image = sitk.Resample(
        image1=image,
        size=new_size,
        transform=sitk.Transform(),
        interpolator=sitk.sitkLinear,
        outputOrigin=image.GetOrigin(),
        outputSpacing=new_spacing,
        outputDirection=image.GetDirection(),
        defaultPixelValue=0,
        outputPixelType=image.GetPixelID(),
    )
    np_img = sitk.GetArrayFromImage(image)

    # Stack image and min-max normalize
    np_img = np.sum(np_img, axis=stack_axis)
    np_min = np_img.min()
    np_max = np_img.max()

    np_img = ((np_img - np_min) / (np_max - np_min) * 255).astype(np.uint8)

    # Equalize histogram + thresholding to get bone feature map
    feature_map = cv.equalizeHist(np_img)
    feature_map[feature_map < 170] = 0
    feature_map[feature_map > 0] = 255
    return np_img, feature_map.astype(int)


def align(
    sub_feature: np.ndarray,
    base_feature: np.ndarray,
) -> Tuple[int, int, int, int]:
    # Calculate cross correlation
    corr = cs.correlate2d(base_feature, sub_feature, mode="same").get()

    # Get maximum cross correlation coordinates (sub_feature relative to base_iamge)
    row_start, col_start = np.unravel_index(corr.argmax(), base_feature.shape)
    row_start = max(row_start - sub_feature.shape[0] // 2, 0)
    col_start = max(col_start - sub_feature.shape[1] // 2, 0)
    row_end = row_start + sub_feature.shape[0]
    col_end = col_start + sub_feature.shape[1]
    return row_start, row_end, col_start, col_end


def process_img(
    img: np.ndarray, size: int, crop: Optional[Tuple[int, int, int, int]] = None
) -> np.ndarray:
    if crop is not None:
        row_start, row_end, col_start, col_end = crop
        img = img[row_start:row_end, col_start:col_end]
    img = center_crop(img)
    img = cv.equalizeHist(img)
    img = cv.resize(img, (size, size))
    return img


def get_plt_img(
    img: np.ndarray, box: Optional[Tuple[int, int, int, int]] = None
) -> None:
    img = img.astype(np.uint8)
    rgb_img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    if box is not None:
        row_start, row_end, col_start, col_end = box
        rgb_img = cv.rectangle(
            rgb_img,
            (col_start, row_start),
            (col_end, row_end),
            (255, 0, 0),
            3,
        )
    return rgb_img


def process_subject(
    info: Dict[str, Union[str, List[str]]], size: int = 1024
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    localizer_img, localizer_feature = get_img(info["localizer"], 0)
    lung_img, lung_feature = get_img(info["lung"], 1)
    lung_box = align(lung_feature, localizer_feature)
    lung_raw = process_img(lung_img, size)
    lung_target = process_img(localizer_img, size, lung_box)
    med_img, med_feature = get_img(info["med"], 1)
    med_box = align(med_feature, localizer_feature)
    med_raw = process_img(med_img, size)
    med_target = process_img(localizer_img, size, med_box)
    return lung_raw, lung_target, med_raw, med_target


def process_subject_expand(
    info: Dict[str, Union[str, List[str]]], size: int = 1024
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Tuple[int, int, int, int],
]:
    localizer_img, localizer_feature = get_img(info["localizer"], 0)
    lung_img, lung_feature = get_img(info["lung"], 1)
    lung_box = align(lung_feature, localizer_feature)
    lung_raw = process_img(lung_img, size)
    lung_target = process_img(localizer_img, size, lung_box)
    return (
        localizer_img,
        localizer_feature,
        lung_img,
        lung_feature,
        lung_raw,
        lung_target,
        lung_box,
    )


def plot_lungs(infos: List[Dict[str, Union[str, List[str]]]]) -> None:
    _, ax = plt.subplots(len(infos), 6, figsize=(3 * 6, 3 * len(infos)))
    for i, info in tqdm(list(enumerate(infos)), dynamic_ncols=True, smoothing=0.01):

        (
            localizer_img,
            localizer_feature,
            lung_img,
            lung_feature,
            lung_raw,
            lung_target,
            lung_box,
        ) = process_subject_expand(info)
        ax[i][0].set_title("lung")
        ax[i][0].imshow(get_plt_img(lung_img))
        ax[i][1].set_title("localizer")
        ax[i][1].imshow(get_plt_img(localizer_img, lung_box))
        ax[i][2].set_title("lung feature")
        ax[i][2].imshow(get_plt_img(lung_feature))
        ax[i][3].set_title("localizer feature")
        ax[i][3].imshow(get_plt_img(localizer_feature, lung_box))
        ax[i][4].set_title("lung raw")
        ax[i][4].imshow(get_plt_img(lung_raw))
        ax[i][5].set_title("lung target")
        ax[i][5].imshow(get_plt_img(lung_target))


def plot_subjects(infos: List[Dict[str, Union[str, List[str]]]]) -> None:
    _, ax = plt.subplots(len(infos), 4, figsize=(3 * len(infos), 12))
    for i, info in tqdm(list(enumerate(infos)), dynamic_ncols=True, smoothing=0.01):
        lung_raw, lung_target, med_raw, med_target = process_subject(info)
        ax[i][0].set_title("lung raw")
        ax[i][0].imshow(get_plt_img(lung_raw))
        ax[i][1].set_title("lung target")
        ax[i][1].imshow(get_plt_img(lung_target))
        ax[i][2].set_title("med raw")
        ax[i][2].imshow(get_plt_img(med_raw))
        ax[i][3].set_title("med target")
        ax[i][3].imshow(get_plt_img(med_target))


def img_to_bytes(np_img: np.ndarray):
    _, buffer = cv.imencode(".png", np_img)
    io_bytes = BytesIO(buffer).getvalue()
    return io_bytes


def prog_infos(writer: LMDBImageWriter, infos: List[Dict[str, Union[str, List[str]]]]):
    for info in tqdm(infos, dynamic_ncols=True, smoothing=0.01):
        # lung_raw, lung_target, med_raw, med_target = process_subject(
        #     info, size=CONFIG.RESOLUTION
        # )
        # writer.set_idx(info["index"], (lung_raw, lung_target, med_raw, med_target))
        lung_raw, lung_target, _, _ = process_subject(info, size=CONFIG.RESOLUTION)
        writer.set_idx(info["index"], (lung_raw, lung_target))
    writer.set_int("length", len(infos))


if __name__ == "__main__":
    train_writer = LMDBImageWriter(CONFIG.COVID_19_TRAIN_LMDB, covid_ct_indexer_lung)
    train_infos = pickle.load(open(CONFIG.COVID_19_DIR / "train_infos.pkl", "rb"))
    print(f"There are {len(train_infos)} training images.")
    prog_infos(train_writer, train_infos)

    test_writer = LMDBImageWriter(CONFIG.COVID_19_TEST_LMDB, covid_ct_indexer_lung)
    test_infos = pickle.load(open(CONFIG.COVID_19_DIR / "test_infos.pkl", "rb"))
    print(f"There are {len(test_infos)} testing images.")
    prog_infos(test_writer, test_infos)
