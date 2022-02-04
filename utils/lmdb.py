import os
from io import BytesIO
from pathlib import Path
from typing import (
    Any,
    Callable,
    Collection,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import cv2 as cv
import numpy as np
from PIL import Image
from torch.functional import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose

import lmdb
from utils.utils import PathLike

T = TypeVar("T")
Indexer = Callable[[T], Collection[bytes]]


def chexpert_indexer(idx: int) -> Tuple[bytes]:
    return tuple([str(idx).zfill(6).encode()])


def covid_ct_indexer(idx: int) -> Tuple[bytes]:
    return tuple(
        [
            f"{str(idx).zfill(6)}_{img_type}".encode()
            for img_type in ["lung_raw", "lung_targ", "med_raw", "med_targ"]
        ]
    )


def covid_ct_indexer_lung(idx: int) -> Tuple[bytes]:
    return tuple(
        [
            f"{str(idx).zfill(6)}_{img_type}".encode()
            for img_type in ["lung_raw", "lung_targ"]
        ]
    )


class LMDBWriter(Generic[T]):
    def __init__(self, path: PathLike, indexer: Indexer[T]) -> None:
        self.path = path
        self.indexer = indexer
        self.env = lmdb.open(
            str(path),
            readonly=False,
            readahead=False,
            meminit=False,
            map_size=1024 ** 4,
        )

    def set(self, key: str, value: bytes):
        with self.env.begin(write=True) as txn:
            txn.put(
                key.encode(),
                value,
            )

    def set_int(self, key: str, value: int):
        with self.env.begin(write=True) as txn:
            txn.put(
                key.encode(),
                str(value).encode(),
            )

    def set_idx(self, idx: T, values: Collection[bytes]):
        with self.env.begin(write=True) as txn:
            keys = self.indexer(idx)
            assert len(keys) == len(
                values
            ), f"Keys and values have different lengths: {keys} {values}"
            for key, value in zip(keys, values):
                txn.put(
                    key,
                    value,
                )


class LMDBReader(Generic[T]):
    def __init__(self, path: PathLike, indexer: Indexer[T]) -> None:
        self.path = path
        self.indexer = indexer
        self.env = lmdb.open(
            str(path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def get(self, key: str) -> bytes:
        with self.env.begin(write=False) as txn:
            value: Any = txn.get(key.encode())
            return value

    def get_int(self, key: str) -> int:
        with self.env.begin(write=False) as txn:
            value: Any = txn.get(key.encode())
            return int(value.decode())

    def get_idx(self, idx: T) -> Tuple[bytes, ...]:
        with self.env.begin(write=False) as txn:
            keys = self.indexer(idx)
            values: Tuple[Any, ...] = tuple(txn.get(key) for key in keys)
            return values


class LMDBImageWriter(LMDBWriter[T]):
    def set_idx(self, idx: T, imgs: Collection[np.ndarray]):
        values = [BytesIO(cv.imencode(".png", img)[1]).getvalue() for img in imgs]
        super().set_idx(idx, values)


class LMDBImageReader(LMDBReader[T]):
    def get_idx(self, idx: T) -> Tuple[Image.Image, ...]:
        values = super().get_idx(idx)

        return tuple(Image.open(BytesIO(value)) for value in values)


class LMDBImageDataset(Dataset[int]):
    def __init__(
        self,
        path: Path,
        indexer: Indexer[int],
        transform: Compose,
        length: Optional[int] = None,
    ):
        self.lmdb = LMDBImageReader(path, indexer)
        self.transform = transform

        self.length = self.lmdb.get_int("length") if length is None else length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        values: Tuple[Any, ...] = tuple(
            self.transform(img) for img in self.lmdb.get_idx(idx)
        )
        return values
