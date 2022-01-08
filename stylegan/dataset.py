from io import BytesIO
from typing import Any, Optional

import lmdb
from lmdb import Environment
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class MultiResolutionDataset(Dataset[int]):
    """
    A Pytorch Dataset with a lmdb backend
    """

    def __init__(self, path: str, transform: Compose, resolution: int):
        env: Optional[Environment] = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not env:
            print("Cannot open lmdb dataset", path)
            exit(1)

        self.env = env
        self.resolution = resolution
        self.transform = transform

        # Get dataset length
        with self.env.begin(write=False) as txn:
            length: Any = txn.get("length".encode("utf-8"))
            self.length = int(length.decode("utf-8"))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        """
        Get raw bytes from lmdb, turn it into image and transform it
        """
        with self.env.begin(write=False) as txn:
            key = f"{self.resolution}-{str(index).zfill(5)}".encode("utf-8")
            img_bytes: Any = txn.get(key)
        img = self.transform(Image.open(BytesIO(img_bytes)))

        return img
