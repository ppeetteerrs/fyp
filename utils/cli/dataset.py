"""Dataset CLI Options"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Tuple, Union

from simple_parsing import Serializable, choice, subparsers
from stylegan2_torch import Resolution

LossSpec = List[Tuple[str, str, float]]


@dataclass
class DatasetParse(Serializable):
    pass


@dataclass
class DatasetGen(Serializable):
    png: bool = False
    """Outputs png pictures instead of lmdb database."""
    resolution: Resolution = 256
    """Output image resolution."""


@dataclass
class DatasetOptions(Serializable):
    """
    Dataset CLI Options
    """

    cmd: Union[DatasetParse, DatasetGen] = subparsers(
        {"parse_meta": DatasetParse, "gen_data": DatasetGen}, default=DatasetGen()
    )
    """Options for active command."""
    dataset: Literal["covid_ct", "lidc", "brixia", "chexpert"] = choice(
        "covid_ct", "lidc", "brixia", "chexpert", default="covid_ct"
    )
    """Dataset to process."""

    @property
    def path(self) -> Path:
        return Path("/data") / self.dataset
