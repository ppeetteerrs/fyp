"""
Generates Covid-CT dataset.
"""

import logging
import warnings
from typing import List, Tuple, cast

# Make DeepDRR quiet
warnings.filterwarnings("ignore")
logging.disable()

from lungmask import mask
from tqdm import tqdm

from utils import load, save_img
from utils.cli import OPTIONS, save_options
from utils.cli.dataset import DatasetGen, DatasetOptions
from utils.ct.drr import get_drr
from utils.ct.localizer import get_localizer
from utils.ct.segmentation import segment
from utils.cv import crop, project, remove_border
from utils.dataset import LMDBImageWriter
from utils.dicom import read_dcm

DATASET_OPTIONS = cast(DatasetOptions, OPTIONS.arch)
GEN_OPTIONS = cast(DatasetGen, DATASET_OPTIONS.cmd)


def gen_covid_ct_data():
    save_options()

    ct_paths = load(OPTIONS.output_dir / "paths.pkl")
    items: List[Tuple[str, Tuple[str, ...], Tuple[str, ...]]] = ct_paths[
        ["Subject", "med", "localizer"]
    ].values.tolist()
    seg_model = mask.get_model("unet", "R231")

    if not GEN_OPTIONS.png:
        writer = LMDBImageWriter(OPTIONS.output_dir / "lmdb")
        writer.set_length(len(items))
    else:
        (OPTIONS.output_dir / "body").mkdir(parents=True, exist_ok=True)
        (OPTIONS.output_dir / "lung").mkdir(parents=True, exist_ok=True)
        (OPTIONS.output_dir / "bones").mkdir(parents=True, exist_ok=True)
        (OPTIONS.output_dir / "soft").mkdir(parents=True, exist_ok=True)
        (OPTIONS.output_dir / "outer").mkdir(parents=True, exist_ok=True)
        (OPTIONS.output_dir / "localizer").mkdir(parents=True, exist_ok=True)
        (OPTIONS.output_dir / "drr").mkdir(parents=True, exist_ok=True)
        writer = None

    for idx, (subject, med_paths, loc_paths) in tqdm(
        list(enumerate(items)), desc="Generating body, lung, bones, soft, outer"
    ):
        raw_sitk, raw_np = read_dcm(list(med_paths))

        lung_mask, body_mask, bones_mask = segment(
            raw_sitk, raw_np, seg_model=seg_model
        )

        body_3d = raw_np * body_mask
        body = project(raw_np * body_mask, crop_size=GEN_OPTIONS.resolution)
        lung = project(
            raw_np * lung_mask, crop_size=GEN_OPTIONS.resolution, remove_border_ref=body_3d
        )
        bones = project(
            raw_np * bones_mask,
            crop_size=GEN_OPTIONS.resolution,
            remove_border_ref=body_3d,
        )
        soft = project(
            raw_np * body_mask * (~bones_mask),
            crop_size=GEN_OPTIONS.resolution,
            remove_border_ref=body_3d,
        )
        outer = project(
            raw_np * body_mask * (~lung_mask),
            crop_size=GEN_OPTIONS.resolution,
            remove_border_ref=body_3d,
        )

        # Save images to different places
        if not GEN_OPTIONS.png:
            assert writer is not None
            writer.set_meta((idx, "subject"), subject)
            writer.set_img((idx, "body"), body)
            writer.set_img((idx, "lung"), lung)
            writer.set_img((idx, "bones"), bones)
            writer.set_img((idx, "soft"), soft)
            writer.set_img((idx, "outer"), outer)
        else:
            save_img(body, OPTIONS.output_dir / f"body/{str(idx).zfill(10)}.png")
            save_img(lung, OPTIONS.output_dir / f"lung/{str(idx).zfill(10)}.png")
            save_img(bones, OPTIONS.output_dir / f"bones/{str(idx).zfill(10)}.png")
            save_img(soft, OPTIONS.output_dir / f"soft/{str(idx).zfill(10)}.png")
            save_img(outer, OPTIONS.output_dir / f"outer/{str(idx).zfill(10)}.png")

    for idx, (subject, med_paths, loc_paths) in tqdm(
        list(enumerate(items)), desc="Generating localizer, drr"
    ):
        raw_sitk, raw_np = read_dcm(list(med_paths))
        loc_np = read_dcm(loc_paths)[1][0]

        loc = crop(get_localizer(raw_np, loc_np)[-1], size=GEN_OPTIONS.resolution)
        drr = crop(remove_border(get_drr(raw_sitk, move_by=750), tol=0.3), size=256)

        # Save images to different places
        if not GEN_OPTIONS.png:
            assert writer is not None
            # writer.set_meta((idx, "subject"), subject)
            writer.set_img((idx, "localizer"), loc)
            writer.set_img((idx, "drr"), drr)
        else:
            save_img(loc, OPTIONS.output_dir / f"localizer/{str(idx).zfill(10)}.png")
            save_img(drr, OPTIONS.output_dir / f"drr/{str(idx).zfill(10)}.png")
