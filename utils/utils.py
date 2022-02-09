import os
from typing import Any, Generator, Literal, Union

from torch.utils.data.dataloader import DataLoader

PathLike = Union[str, bytes, os.PathLike]


def repeat(loader: DataLoader) -> Generator[Any, None, None]:
    while True:
        for batch in loader:
            yield batch
