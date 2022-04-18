"""
Custom image dataset classes
"""

import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image, ImageOps
from torch import Tensor
from torch.functional import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import Compose
from utils import PathLike, clean_pathlikes, clean_strs

from dataset.lmdb import Indexer, LMDBImageReader


def load_PIL(path: Path) -> Image.Image:
    """
    Loads a grayscale PIL image.
    """

    with open(path, "rb") as f:
        img = Image.open(f)
        return ImageOps.grayscale(img)


transform = transforms.Compose(
    [
        # Converts 8-bit PIL to (0, 1)
        transforms.ToTensor(),
        # (x - mean) / std hence range is now (-1, 1)
        transforms.Normalize(0.5, 0.5, inplace=True),
    ]
)
"""
Standard `torchvision` transform to normalize PIL.Image.
"""


class MultiImageDataset(Dataset[Dict[str, Tensor]]):
    def __init__(
        self,
        folders: Union[PathLike, Sequence[PathLike]],
        subfolders: Optional[Union[str, Sequence[str]]] = None,
        extensions: Sequence[str] = IMG_EXTENSIONS,
        length: Optional[int] = None,
        shuffle: bool = False,
    ) -> None:
        """
        In-memory image dataset that contains images from multiple `folders` and
        multiple `subfolders` (classes) under each `folder`. Directory structure should be
        `folder/subfolder/<img_name>.<img_extension>`.

        Args:
            folders (Union[PathLike, Sequence[PathLike]]): Folder(s) that contain images.
            subfolders (Optional[Union[str, Sequence[str]]], optional): Subfolder(s) that contain images.
            extensions (Sequence[str], optional): Accepted image extensions.
            length (Optional[int], optional): Length limit of dataset.
            shuffle (bool): Whether to shuffle images when length is specified.
        """

        # Clean function arguments
        folders = clean_pathlikes(folders)
        subfolders = clean_strs(subfolders)

        # Append image paths from each (folder, subfolder) pair
        self.data: Dict[str, Tensor] = {}

        for subfolder in subfolders:
            img_paths: List[Path] = []
            for folder in folders:
                img_paths.extend(
                    img_path
                    for img_path in (folder / subfolder).iterdir()
                    if img_path.suffix in extensions
                )
            if length is not None:
                if shuffle:
                    random.shuffle(img_paths)
                img_paths = img_paths[:length]
            self.data[subfolder] = torch.stack(
                [transform(load_PIL(img_path)) for img_path in img_paths]
            )

        # Validate tensor shapes
        val_lengths = [val.shape[0] for val in self.data.values()]
        self.length = length if length is not None else max(*val_lengths)
        assert all(
            [val_length == self.length for val_length in val_lengths]
        ), "Image folder length mismatch."

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        """
        Returns `Dataset[idx]`, which is a dict of tensors on the CPU.
        The keys are the subfolder names.
        Cannot run `to("cuda")` here because there are multiple dataloader workers.
        """
        return {key: val[idx] for key, val in self.data.items()}

    def __len__(self) -> int:
        """
        Returns dataset length.
        """
        return self.length


class LMDBImageDataset(Dataset[Tuple[Tensor, ...]]):
    def __init__(
        self,
        path: PathLike,
        indexer: Indexer[int] = lambda idx: tuple([str(idx).zfill(6).encode()]),
        transform: Compose = transform,
        length: Optional[int] = None,
    ):
        """
        A dataset of images stored in lmdb.

        Args:
            path (Path): Path to lmdb folder. Folder should contain `data.mdb` and `lock.mdb`.
            indexer (Indexer[int], optional): Function that maps integer index to tuple of `byte` keys.
            transform (Compose, optional): Transform applied to fetched images. Defaults to transform.
            length (Optional[int], optional): Length limit for the dataset. Defaults to None.
        """

        self.lmdb = LMDBImageReader(path, indexer)
        self.transform = transform

        self.length = self.lmdb.get_int("length") if length is None else length

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        """
        Returns `Dataset[idx]`, which is a tuple of tensors on the CPU.
        Cannot run `to("cuda")` here because there are multiple dataloader workers.
        """

        return tuple(self.transform(img) for img in self.lmdb.get_idx(idx))

    def __len__(self) -> int:
        """
        Returns dataset length.
        """
        return self.length
