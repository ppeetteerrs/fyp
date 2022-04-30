from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2 as cv
import lmdb
import numpy as np
from PIL import Image, ImageOps
from torch.functional import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

LMDBIndex = Union[int, Tuple[int, str]]


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


class LMDBIndexed:
    """
    An LMDB database wrapper indexed by either a single int or a tuple (int, str).
    """

    @staticmethod
    def idx_to_key(idx: LMDBIndex) -> str:
        if isinstance(idx, int):
            return f"{str(idx).zfill(10)}"
        else:
            return f"{str(idx[0]).zfill(10)}_{idx[1]}"


class LMDBWriter(LMDBIndexed):
    def __init__(self, path: Union[str, Path]) -> None:
        """
        Initiates a writer to a LMDB dataset.

        Args:
            path (Union[str, Path]): Path to LMDB folder
        """

        self.path = path
        self.env = lmdb.open(
            str(path),
            readonly=False,
            readahead=False,
            meminit=False,
            map_size=1024**4,
        )

    def set(self, key: str, value: bytes):
        """
        Sets the byte value of the given key.
        """

        with self.env.begin(write=True) as txn:
            txn.put(
                key.encode(),
                value,
            )

    def set_int(self, key: str, value: int):
        """
        Sets the int value of the given key.
        """

        with self.env.begin(write=True) as txn:
            txn.put(
                key.encode(),
                str(value).encode(),
            )

    def set_str(self, key: str, value: str):
        """
        Sets the string value of the given key.
        """

        with self.env.begin(write=True) as txn:
            txn.put(
                key.encode(),
                value.encode(),
            )


class LMDBReader(LMDBIndexed):
    def __init__(self, path: Union[str, Path]) -> None:
        """
        Initiates a reader from a LMDB dataset.

        Args:
            path (Union[str, Path]): Path to LMDB folder
        """

        self.path = path
        self.env = lmdb.open(
            str(path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def get(self, key: str) -> bytes:
        """
        Gets the byte value of the given key.
        """

        with self.env.begin(write=False) as txn:
            value: Any = txn.get(key.encode())
            return value

    def get_int(self, key: str) -> int:
        """
        Gets the int value of the given key.
        """

        with self.env.begin(write=False) as txn:
            value: Any = txn.get(key.encode())
            return int(value.decode())

    def get_str(self, key: str) -> str:
        """
        Gets the string value of the given key.
        """

        with self.env.begin(write=False) as txn:
            value: Any = txn.get(key.encode())
            return value.decode()


class LMDBImageWriter(LMDBWriter):
    def set_length(self, length: int):
        """
        Sets image dataset length.
        """

        self.set_int("length", length)

    def set_meta(self, idx: LMDBIndex, value: str):
        """
        Sets a metadata field for a certain integer index.
        """

        self.set_str(self.idx_to_key(idx), value)

    def set_img(self, idx: LMDBIndex, img: Union[np.ndarray, Image.Image]):
        """
        Sets the image data for a certain integer index.
        """

        if isinstance(img, Image.Image):
            value = BytesIO()
            img.save(value, format="TIFF")
        else:
            value = BytesIO(cv.imencode(".tiff", img)[1])
        self.set(self.idx_to_key(idx), value.getvalue())


class LMDBImageReader(LMDBReader):
    def get_length(self) -> int:
        """
        Gets image dataset length.
        """

        return self.get_int("length")

    def get_meta(self, idx: LMDBIndex) -> str:
        """
        Gets metadata field for a certain integer index.
        """

        return self.get_str(self.idx_to_key(idx))

    def get_img(self, idx: LMDBIndex) -> Image.Image:
        """
        Gets the image data for a certain integer index.
        """

        return Image.open(BytesIO(self.get(self.idx_to_key(idx))))


class LMDBImageDataset(Dataset[int]):
    def __init__(
        self,
        path: Union[str, Path],
        names: List[str] = [],
        length: Optional[int] = None,
        transform: transforms.Compose = transform,
    ):
        """
        An LMDB image dataset. Each integer index contains multiple images (given by `names`).
        E.g. dataset[i] returns a dictionary with keys `names`. Each value is an image Tensor.
        If `names` is empty, dataset[i] returns a single Tensor.

        `length` truncates the dataset to specified length. Useful for getting a constant subset as samples.
        Supporting (deterministic) shuffle before length truncation is a trivial task, implement if necessary.

        `transform` is applied to each image before conversion into a PyTorch Tensor.
        """

        self.lmdb = LMDBImageReader(path)
        self.transform = transform
        self.names = names
        self.length = self.lmdb.get_int("length") if length is None else length

    def __len__(self) -> int:
        """
        Dataset length.
        """

        return self.length

    def __getitem__(self, idx: int) -> Union[Dict[str, Tensor], Tensor]:
        """
        Gets dataset item. Returns an image Tensor or dictionary of image Tensors.
        """

        if len(self.names) > 0:
            return {
                name: self.transform(self.lmdb.get_img((idx, name)))
                for name in self.names
            }
        else:
            return self.transform(self.lmdb.get_img(idx))
