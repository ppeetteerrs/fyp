from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms


def remove_border(img: np.ndarray, tol=50) -> np.ndarray:
    mask = img > tol
    m, n = img.shape
    mask0, mask1 = mask.any(0), mask.any(1)
    col_start, col_end = mask0.argmax(), n - mask0[::-1].argmax()
    row_start, row_end = mask1.argmax(), m - mask1[::-1].argmax()
    return img[row_start:row_end, col_start:col_end]


def center_crop(img: np.ndarray) -> np.ndarray:
    x, y = img.shape
    dim = min(x, y)

    x_start = (x - dim) // 2
    y_start = (y - dim) // 2

    return img[x_start : x_start + dim, y_start : y_start + dim]


def plot_imgs(imgs: List[np.ndarray], shape: Tuple[int, int]):
    total = np.prod(shape)
    assert total == len(imgs), "Incompatible shape"
    imgs = [
        np.concatenate(imgs[i * shape[0] : (i + 1) * shape[0]], axis=0)
        for i in range(shape[1])
    ]
    _, ax = plt.subplots(figsize=(shape[0] * 5, shape[1] * 5))
    ax.imshow(np.concatenate(imgs, axis=1), cmap="gray")


transform = transforms.Compose(
    [
        # Converts 8-bit PIL to (0, 1)
        transforms.ToTensor(),
        # (x - mean) / std hence range is now (-1, 1)
        transforms.Normalize(0.5, 0.5, inplace=True),
    ]
)
