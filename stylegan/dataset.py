from io import BytesIO
from typing import Any, Generator, Optional

import lmdb
from lmdb import Environment
from PIL import Image
from torch.functional import Tensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
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
            length: Any = txn.get("length".encode())
            self.length = int(length.decode())

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        """
        Get raw bytes from lmdb, turn it into image and transform it
        """
        with self.env.begin(write=False) as txn:
            key = str(index).zfill(6).encode()
            img_bytes: Any = txn.get(key)
        img = self.transform(Image.open(BytesIO(img_bytes)))

        return img


def repeat(loader: DataLoader) -> Generator[Tensor, None, None]:
    while True:
        for batch in loader:
            yield batch
