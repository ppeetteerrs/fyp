from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Sequence, Union

from torch import Tensor, nn
from torch.utils.data.dataloader import DataLoader

PathLike = Union[str, Path]


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


def clean_pathlikes(arg: Optional[Union[PathLike, Sequence[PathLike]]]) -> List[Path]:
    arg = Path() if arg is None else arg
    arg = [arg] if isinstance(arg, (str, Path)) else list(arg)
    return [Path(item).resolve() for item in arg]


def clean_strs(arg: Optional[Union[str, Sequence[str]]]) -> List[str]:
    arg = "" if arg is None else arg
    return [arg] if isinstance(arg, str) else list(arg)


def repeat(loader: DataLoader) -> Generator[Any, None, None]:
    while True:
        for batch in loader:
            yield batch


def check_exist(paths: Sequence[Path]) -> Sequence[Path]:

    for path in paths:
        if not path.is_dir():
            raise FileNotFoundError(f"{path} does not exist!")
    return paths


def to_device(
    tensor_dict: Dict[str, Tensor], device: str = "cuda"
) -> Dict[str, Tensor]:
    return {k: v.to(device) for k, v in tensor_dict.items()}
