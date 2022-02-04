import os
from typing import Any, Generator, Literal, Union

from torch.utils.data.dataloader import DataLoader

PathLike = Union[str, bytes, os.PathLike]

Resolution = Literal[4, 8, 16, 32, 64, 128, 256, 512, 1024]


def repeat(loader: DataLoader) -> Generator[Any, None, None]:
    while True:
        for batch in loader:
            yield batch
