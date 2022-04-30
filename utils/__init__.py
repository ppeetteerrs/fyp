import pickle
from pathlib import Path
from typing import Any, Dict, Generator, TypeVar, Union

import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.utils.data.dataloader import DataLoader

# numpy array types
ArrB = NDArray[np.bool8]
Arr8U = NDArray[np.uint8]
Arr16U = NDArray[np.uint16]
Arr32F = NDArray[np.float32]
AnyArr = TypeVar("AnyArr", bound=np.ndarray)

def load(path: Union[str, Path]) -> Any:
    """
    Loads a pickle file.
    """

    return pickle.load(open(path, "rb"))


def save_img(img: Arr32F, path: Union[str, Path]):
    """
    Saves an image using OpenCV.
    """

    cv.imwrite(str(path), (img * 255).astype(np.uint8))


def nop(item: Any, *_: Any, **__: Any) -> Any:
    """
    NOP function overload to mute imported libraries.
    """

    return item


def accumulate(
    model1: nn.Module,
    model2: nn.Module,
    decay: float = 0.5 ** (32 / (10 * 1000)),
) -> None:
    """
    Accumulates parameters of model2 onto model1 using EMA.
    """

    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def repeat(loader: DataLoader) -> Generator[Any, None, None]:
    """
    Repeats a PyTorch Dataloader indefinitely.
    """

    while True:
        for batch in loader:
            yield batch


def to_device(
    tensor_dict: Dict[str, Tensor], device: str = "cuda"
) -> Dict[str, Tensor]:
    """
    Move all tensors in a dictionary to specified device.
    """

    return {k: v.to(device) for k, v in tensor_dict.items()}
