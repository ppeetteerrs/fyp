import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple, cast

warnings.filterwarnings("ignore")

import cv2 as cv
import numpy as np
import pydicom
import SimpleITK as sitk
from tqdm import tqdm
from utils import save_img
from utils.cli import OPTIONS, save_options
from utils.cli.dataset import DatasetGen, DatasetOptions
from utils.cv import crop, min_max_normalize
from utils.dataset import LMDBImageWriter

DATASET_OPTIONS = cast(DatasetOptions, OPTIONS.arch)
GEN_OPTIONS = cast(DatasetGen, DATASET_OPTIONS.cmd)


def proc_item(arg: Tuple[int, Path]) -> Tuple[int, np.ndarray]:
    idx, dcm_path = arg
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(dcm_path))
    img = reader.Execute()
    img = sitk.RescaleIntensity(img, 0, 255)
    np_img = sitk.GetArrayFromImage(img)[0].astype(np.uint8)
    np_img = crop(np_img, size=GEN_OPTIONS.resolution)
    if GEN_OPTIONS.equalize:
        np_img = cv.equalizeHist(np_img)
    np_img = np_img.astype(np.float32) / 255
    np_img = min_max_normalize(np_img)
    return idx, np_img


def gen_brixia_data():
    save_options()

    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    dcm_paths = list(DATASET_OPTIONS.path.glob("*.dcm"))
    clean_dcm_path = [
        dcm_path
        for dcm_path in tqdm(
            dcm_paths,
            desc="Filtering out scans with high bit != 11...",
            dynamic_ncols=True,
            smoothing=0.01,
        )
        if pydicom.dcmread(dcm_path).HighBit == 11
    ]

    if not GEN_OPTIONS.png:
        writer = LMDBImageWriter(OPTIONS.output_dir / "lmdb")
        writer.set_length(len(clean_dcm_path))

        with Pool(4) as pool:
            for idx, val in tqdm(
                pool.imap_unordered(proc_item, enumerate(clean_dcm_path)),
                total=len(clean_dcm_path),
                dynamic_ncols=True,
                smoothing=0.01,
            ):
                writer.set_img(idx, val)
    else:
        (OPTIONS.output_dir / "imgs").mkdir(parents=True, exist_ok=True)

        with Pool(4) as pool:
            for idx, val in tqdm(
                pool.imap_unordered(proc_item, enumerate(clean_dcm_path)),
                total=len(clean_dcm_path),
                dynamic_ncols=True,
                smoothing=0.01,
            ):
                save_img(val, OPTIONS.output_dir / f"imgs/{str(idx).zfill(10)}.png")
