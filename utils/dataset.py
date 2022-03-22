from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from numpy import dtype
from torch import Tensor
from torch.functional import Tensor
from torch.utils.data import Dataset, TensorDataset
from torchvision.transforms import Compose
from tqdm import tqdm

from utils.img import load_PIL, transform
from utils.lmdb import Indexer, LMDBImageReader
from utils.utils import PathLike, check_exist


class BinaryImageDataset(TensorDataset):
    def __init__(
        self,
        positives: Sequence[PathLike],
        negatives: Sequence[PathLike],
        pattern: str = "*.png",
    ):
        self.positive_paths: List[Path] = list(
            chain.from_iterable(Path(positive).glob(pattern) for positive in positives)
        )
        self.negative_paths: List[Path] = list(
            chain.from_iterable(Path(negative).glob(pattern) for negative in negatives)
        )

        images = torch.stack(
            list(
                chain(
                    [
                        transform(load_PIL(path))
                        for path in tqdm(
                            self.positive_paths, desc="Loading positive images"
                        )
                    ],
                    [
                        transform(load_PIL(path))
                        for path in tqdm(
                            self.negative_paths, desc="Loading negative images"
                        )
                    ],
                )
            )
        )

        labels = torch.Tensor(
            list(
                chain(
                    [1 for _ in self.positive_paths],
                    [0 for _ in self.negative_paths],
                )
            ),
        ).long()
        super().__init__(images, labels)

    def __getitem__(self, index: int):
        return tuple(tensor[index] for tensor in self.tensors)


class MulticlassImageDataset(TensorDataset):
    def __init__(
        self,
        folders: Sequence[PathLike],
        classes: Sequence[str],
        pattern: str = "*.png",
        length: int = None,
    ) -> None:

        self.classes = classes
        self.length = length
        data: List[Tensor] = []

        for img_class in tqdm(classes, desc="Loading images from each class"):
            class_folders = check_exist(
                [Path(folder) / img_class for folder in folders]
            )
            class_paths = chain.from_iterable(
                class_folder.glob(pattern) for class_folder in class_folders
            )
            self.class_paths = class_paths
            class_imgs = torch.stack(
                [transform(load_PIL(path)) for path in class_paths]
            )
            data.append(class_imgs)

        super().__init__(*data)

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return {
            img_class: tensor[index]
            for img_class, tensor in zip(self.classes, self.tensors)
        }

    def __len__(self):
        if self.length is not None:
            return self.length
        return super().__len__()


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
