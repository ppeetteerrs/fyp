from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from torch import Tensor
from torchvision import transforms


def load_PIL(path: Path) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return ImageOps.grayscale(img)


transform: Callable[..., Tensor] = transforms.Compose(
    [
        # Converts 8-bit PIL to (0, 1)
        transforms.ToTensor(),
        # (x - mean) / std hence range is now (-1, 1)
        transforms.Normalize(0.5, 0.5, inplace=True),
    ]
)
