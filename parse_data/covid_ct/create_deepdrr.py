import pickle
import warnings
from functools import partial
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, List, Tuple

import cv2 as cv
import numpy as np
import SimpleITK as sitk
from deepdrr import MobileCArm, Volume
from deepdrr.projector import Projector
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from utils.config import CONFIG
from utils.img import center_crop
from utils.lmdb import LMDBImageWriter, covid_ct_indexer_drr

warnings.filterwarnings("ignore")


def to_nifti(info: Any, nifti_dir: Path):
    out_file = nifti_dir / f"{info['subject']}.nii.gz"

    # Read image using sitk
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(info["lung"])
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
    sitk.WriteImage(image, str(out_file))


def get_drr(filename: Path, tmp_dir: str) -> np.ndarray:
    volume = Volume.from_nifti(filename, cache_dir=Path(tmp_dir))
    volume.facedown()
    carm = MobileCArm()
    carm.reposition(volume.center_in_world)
    carm.move_to(alpha=0, beta=-180, degrees=True)
    carm.move_by([0, 0, volume.center_in_world[0]])
    with Projector(volume, carm=carm) as projector:
        img = projector()
    return (img * 255).astype(np.uint8)


def proc_img(
    info: Any, nifti_dir: Path, tmp_dir: str, size: int = 1024
) -> Tuple[int, np.ndarray]:
    out_file = nifti_dir / f"{info['subject']}.nii.gz"
    to_nifti(out_file, info["lung"])
    img = get_drr(out_file, tmp_dir)
    img = center_crop(img)
    img = cv.equalizeHist(img)
    img = cv.resize(img, (size, size))
    return info["index"], img


if __name__ == "__main__":
    nifti_dir = Path(CONFIG.OUTPUT_DIR / "nifti")
    nifti_dir.mkdir(parents=True, exist_ok=True)

    lmdb_dir = Path(CONFIG.COVID_19_TEST_LMDB)

    infos = pickle.load(open(CONFIG.COVID_19_DIR / "test_infos.pkl", "rb"))
    tmp_dir = mkdtemp()

    proc_img_fn = partial(proc_img, nifti_dir=nifti_dir, tmp_dir=tmp_dir)

    writer = LMDBImageWriter(lmdb_dir, covid_ct_indexer_drr)

    print("Converting to nifti")
    process_map(partial(to_nifti, nifti_dir=nifti_dir), infos, chunksize=1)

    print("Converting to DRR")
    for info in tqdm(infos, dynamic_ncols=True, smoothing=0.01):
        out_file = nifti_dir / f"{info['subject']}.nii.gz"
        img = get_drr(out_file, tmp_dir)
        img = center_crop(img)
        img = cv.equalizeHist(img)
        img = cv.resize(img, (CONFIG.RESOLUTION, CONFIG.RESOLUTION))
        writer.set_idx(info["index"], [img])

    writer.set_int("length", len(infos))
