"""StyleGAN CLI Options"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from simple_parsing import Serializable, list_field, subparsers
from stylegan2_torch import Resolution


@dataclass
class StyleGANTrain(Serializable):
    """
    StyleGAN Training Options
    """

    # Training
    iterations: int = 800000
    """Total number of training iterations."""
    batch_size: int = 8
    """Training batch size (per node)."""
    lr: float = 0.002
    """Learning rate."""
    mixing: float = 0.9
    """Style mixing probability."""

    # Regularization
    r1: float = 10
    """Weight of discriminator R1 regularization loss."""
    path_reg: float = 2
    """Weight of generator path regularization loss."""
    path_batch_shrink: int = 2
    """Ratio to shrink batch_size for generator regularization."""
    d_reg_interval: int = 16
    """Interval to regularize discriminator."""
    g_reg_interval: int = 4
    """Interval to regularize generator."""

    # Logging
    sample_size: int = 25
    """Number of samples to generate."""
    sample_interval: int = 500
    """Interval to generate samples."""
    ckpt_interval: int = 100000
    """Interval to generate checkpoints."""
    dataset: str = "output/chexpert/lmdb"
    """Paths to image LMDB dataset."""

    # Re-training
    restart: bool = False
    """Restart training, i.e. resets checkpoint current step and optimizer state."""


@dataclass
class StyleGANGenerate(Serializable):
    """
    StyleGAN Training Options
    """

    n_images: int = 2000
    """Number of images to generate from noise."""
    mixing: float = 0
    """Style mixing probability."""
    trunc: float = 1
    """Truncation ratio. If 0.7 means 0.7 taken from noise, 0.3 from mean."""


@dataclass
class StyleGANArchOptions(Serializable):
    """
    StyleGAN Architecture Options
    """

    output_resolution: Resolution = 256
    """Output resolution."""
    latent_dim: int = 512
    """Latent vector dimensions."""
    n_mlp: int = 8
    """Number of layers in mapping network."""
    lr_mlp_mult: float = 0.01
    """Learning rate multiplier for mapping network."""
    blur_kernel: List[int] = list_field(1, 3, 3, 1)
    """Blurring kernel."""
    channels_mult: int = 2
    """Convolution channels multiplier."""

    @property
    def channels_map(self) -> Dict[Resolution, int]:
        return {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * self.channels_mult,
            128: 128 * self.channels_mult,
            256: 64 * self.channels_mult,
            512: 32 * self.channels_mult,
            1024: 16 * self.channels_mult,
        }


@dataclass
class StyleGANArch(StyleGANArchOptions, Serializable):
    """
    StyleGAN Architecture Options
    """

    cmd: Union[StyleGANTrain, StyleGANGenerate] = subparsers(
        {"train": StyleGANTrain, "generate": StyleGANGenerate}, default=StyleGANTrain()
    )
    ckpt: Optional[str] = None
    """Path to pretrained StyleGAN."""
