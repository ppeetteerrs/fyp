import pickle
from typing import Any, List

import cv2 as cv
import numpy as np
import SimpleITK as sitk
from tqdm.contrib.concurrent import process_map
from utils.config import CONFIG
from utils.img import center_crop
from utils.lmdb import LMDBImageWriter, covid_ct_indexer_bone


def process(img: np.ndarray) -> np.ndarray:
    # img = cv.equalizeHist(img.astype(np.uint8))
    legal = np.count_nonzero(img) > 0
    if legal:
        upper_thres = np.percentile(img[img > 0], 90)
    else:
        upper_thres = 1
    img = np.where(img < upper_thres, 0, 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(2):
        img = cv.dilate(img, kernel, iterations=1)
        img = cv.erode(img, kernel, iterations=1)
    kernel = np.ones((2, 2), np.uint8)
    img = cv.erode(img, kernel, iterations=1)
    img = cv.dilate(img, kernel, iterations=1)
    # img = cv.dilate(img, kernel, iterations=1)
    # img = cv.dilate(img, kernel, iterations=1)
    # upper_thres = np.percentile(np_img[np_img > 0], 95)
    # np_img = np.where(np_img < upper_thres, 0, 255).astype(np.uint8)
    return img

def renoise(img: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)

    for _ in range(5):
        img = cv.dilate(img, kernel, iterations=1)
        img = cv.erode(img, kernel, iterations=1)
    return img


def get_img(paths: List[str]):
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
    # upper_thres = np.percentile(np_img[np_img > 0], 95)
    # np_img = np.where(np_img < upper_thres, 0, 255).astype(np.uint8)
    imgs = [process(img) for img in np_img]

    np_thres = cv.resize(
        center_crop(np.mean(imgs, axis=1).astype(np.uint8)), (256, 256)
    )
    renoised = renoise(np_thres)
    return cv.equalizeHist(renoised)
    # return renoised



if __name__ == "__main__":
    train_infos = pickle.load(open(CONFIG.COVID_19_DIR / "train_infos.pkl", "rb"))

    writer = LMDBImageWriter(CONFIG.COVID_19_TRAIN_LMDB, covid_ct_indexer_bone)

    imgs = process_map(get_img, [info["lung"] for info in train_infos], chunksize=1)

    for info, img in zip(train_infos, imgs):
        writer.set_idx(info["index"], [img])

    test_infos = pickle.load(open(CONFIG.COVID_19_DIR / "test_infos.pkl", "rb"))

    writer = LMDBImageWriter(CONFIG.COVID_19_TEST_LMDB, covid_ct_indexer_bone)

    imgs = process_map(get_img, [info["lung"] for info in test_infos], chunksize=1)

    for info, img in zip(test_infos, imgs):
        writer.set_idx(info["index"], [img])
