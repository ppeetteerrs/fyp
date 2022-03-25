"""lmdb helper classes"""

from io import BytesIO
from typing import Any, Callable, Collection, Generic, Tuple, TypeVar

import cv2 as cv
import numpy as np
from PIL import Image
from utils import PathLike

import lmdb

T = TypeVar("T")
Indexer = Callable[[T], Collection[bytes]]


class LMDBWriter(Generic[T]):
    def __init__(self, path: PathLike, indexer: Indexer[T]) -> None:
        """
        A wrapper class that writes to lmdb.
        """

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
        """
        Sets `db[key]` to bytes `value`.
        """

        with self.env.begin(write=True) as txn:
            txn.put(
                key.encode(),
                value,
            )

    def set_int(self, key: str, value: int):
        """
        Sets `db[key]` to integer `value`. Useful for setting db `length`.
        """

        with self.env.begin(write=True) as txn:
            txn.put(
                key.encode(),
                str(value).encode(),
            )

    def set_idx(self, idx: T, values: Collection[bytes]):
        """
        Maps `idx` to a tuple of string (`keys`) through `self.indexer`.
        Sets `db[key]` to `value` for each `(key, value)` pair.
        """

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
        """
        A wrapper class that reads from lmdb.
        """

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
        """
        Gets bytes from `db[key]`.
        """

        with self.env.begin(write=False) as txn:
            value: Any = txn.get(key.encode())
            return value

    def get_int(self, key: str) -> int:
        """
        Gets integer from `db[key]`. Useful for getting db `length`.
        """

        with self.env.begin(write=False) as txn:
            value: Any = txn.get(key.encode())
            return int(value.decode())

    def get_idx(self, idx: T) -> Tuple[bytes, ...]:
        """
        Maps `idx` to a tuple of string (`keys`) through `self.indexer`.
        Gets `db[key]` for each `key` in `keys`.
        """

        with self.env.begin(write=False) as txn:
            keys = self.indexer(idx)
            values: Tuple[Any, ...] = tuple(txn.get(key) for key in keys)
            return values


class LMDBImageWriter(LMDBWriter[T]):
    """
    `LMDBWriter` that stores `np.ndarray`.
    """

    def set_idx(self, idx: T, imgs: Collection[np.ndarray]):
        """
        Maps `idx` to a tuple of string (`keys`) through `self.indexer`.
        Sets `db[key]` to `img` for each `(key, img)` pair.
        """

        values = [BytesIO(cv.imencode(".png", img)[1]).getvalue() for img in imgs]
        super().set_idx(idx, values)


class LMDBImageReader(LMDBReader[T]):
    """
    `LMDBWriter` that reads `PIL.Image`.
    """

    def get_idx(self, idx: T) -> Tuple[Image.Image, ...]:
        """
        Maps `idx` to a tuple of string (`keys`) through `self.indexer`.
        Gets `db[key]` for each `key` in `keys`.
        """

        values = super().get_idx(idx)

        return tuple(Image.open(BytesIO(value)) for value in values)
