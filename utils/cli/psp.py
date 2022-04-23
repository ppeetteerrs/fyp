"""pSp CLI Options"""

from dataclasses import dataclass
from typing import List, Literal, Tuple

from simple_parsing import Serializable, choice, list_field, subparsers
from stylegan2_torch import Resolution
from utils.cli.stylegan import StyleGANArchOptions

LossSpec = List[Tuple[str, str, float]]


def decode_loss_spec(items: str) -> LossSpec:
    """
    Decodes CLI string from `{input1:truth1:weight1,input2:truth2:weight2}` to
    list of `(input, truth, weight)` tuples.
    """

    if items == "":
        return []

    output: LossSpec = []
    for item in items.split(","):
        components = item.split(":")
        assert len(components) == 3, f"{item} in {items} should have 3 components."
        output.append((components[0], components[1], float(components[2])))
    return output


@dataclass
class PSPTrain(Serializable):
    """
    pSp Training Options
    """

    # Training
    iterations: int = 200000
    """Total number of training iterations."""
    batch_size: int = 2
    """Training batch size."""
    lr: float = 0.0001
    """Learning rate."""
    mixing: float = 0
    """Style mixing probability (currently unused)."""
    dataset: str = "input/data/covid_ct_lmdb"
    """Paths to image LMDB dataset."""

    # Loss function
    l2: str = "out:soft:3"
    """Specs of l2 pixel-wise loss. Format: {input1:truth1:weight1,input2:truth2:weight2}."""
    id: str = ""
    """Specs of ID (identity) loss. Format: {input1:truth1:weight1,input2:truth2:weight2}."""
    id_ckpt: str = "input/pretrained/arcface.pt"
    """ArcFace model checkpoint for ID loss."""
    lpips: str = "out:body:0.1"
    """Specs of LPIPS loss. Format: {input1:truth1:weight1,input2:truth2:weight2}."""
    reg: float = 0.01
    """Weight of regularization loss."""

    # Logging
    sample_size: int = 4
    """Number of samples to generate."""
    sample_interval: int = 200
    """Interval to generate samples."""
    ckpt_interval: int = 50000
    """Interval to generate checkpoints."""

    @property
    def l2_spec(self) -> LossSpec:
        """Parses `self.l2` into list of (input, truth, weight) tuples."""
        return decode_loss_spec(self.l2)

    @property
    def id_spec(self) -> LossSpec:
        """Parses `self.id` into list of (input, truth, weight) tuples."""
        return decode_loss_spec(self.id)

    @property
    def lpips_spec(self) -> LossSpec:
        """Parses `self.lpips` into list of (input, truth, weight) tuples."""
        return decode_loss_spec(self.lpips)


@dataclass
class PSPGenerate(Serializable):
    """
    pSp Generate Options
    """

    dataset: str = ""
    """Paths to image LMDB dataset."""


@dataclass
class PSPMix(Serializable):
    """
    pSp Mixing Options
    """

    dataset: str = ""
    """Paths to image LMDB dataset."""
    mix_mode: Literal["alt", "half", "mean"] = choice(
        "alt", "half", "mean", default="mean"
    )
    """Mixing mode: alt, half or mean."""
    n_images: int = 0
    """Number of images to generate via mixing."""


# TODO: Add StyleGAN Options


@dataclass
class PSPArch(StyleGANArchOptions, Serializable):
    """
    pSp Architecture Options
    """

    cmd: PSPTrain = subparsers(
        {"train": PSPTrain, "generate": PSPGenerate, "mix": PSPMix}, default=PSPTrain()
    )
    """Options for active command."""
    classes: List[str] = list_field("body", "lung", "soft")
    """
    Image classes needed for dataset.
    (Used as `subfolders` for `MultiImageDataset`, provided as dict of Tensors.)
    """
    inputs: List[str] = list_field("body")
    """Image classes used as input."""
    use_mean: bool = True
    """Add latent average to encoded style vectors."""
    input_resolution: Resolution = 256
    """Input image resolution."""
    ckpt: str = "stylegan.pt"
    """Path to pretrained StyleGAN / pSp."""
