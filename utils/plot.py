"""
2D and 3D image plotting functions.
"""

from typing import List, Optional, Sequence, Tuple, Union, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import IntSlider, interact


def use_widget():
    """
    Change VSCode Matplotlib backend to use widgets.
    """

    matplotlib.use("module://ipympl.backend_nbagg", force=True)


def plot(
    imgs: Union[Sequence[np.ndarray], Sequence[Sequence[np.ndarray]]],
    titles: Optional[Sequence[str]] = None,
):
    """
    Plots a 1D or 2D list of images with optional per-column titles.
    """

    img_grid: List[List[np.ndarray]]

    if not isinstance(imgs[0], List):
        img_grid = cast(List[List[np.ndarray]], [imgs])
    else:
        img_grid = cast(List[List[np.ndarray]], imgs)

    n_rows = len(img_grid)
    n_cols = len(img_grid[0])

    plot_w = max(6, 5 * n_cols)
    plot_h = max(6, 5 * n_rows)

    titles = ["" for _ in range(n_cols)] if titles is None else titles
    _, axes = plt.subplots(n_rows, n_cols, figsize=(plot_w, plot_h))

    for i in range(n_rows):
        for j in range(n_cols):
            if n_rows > 1:
                ax = axes[i][j]
            elif n_cols > 1:
                ax = axes[j]
            else:
                ax = axes
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            if i == 0:
                ax.set_title(titles[j])
            img = img_grid[i][j]
            cmap = "gray" if len(img.shape) <= 2 else None
            ax.imshow(img, cmap=cmap)


def plot_volume(
    img: np.ndarray,
    mask: Optional[np.ndarray] = None,
    *,
    axis: int = 0,
    value_range: Optional[Tuple[int, int]] = None,
    size: int = 8,
    initial_val: int = 0,
):
    """
    Plots a 3D image using a series of 2D image with a slider.

    Args:
        img (np.ndarray): 3D image array.
        mask (Optional[np.ndarray], optional): Segmentation masks to overlay. Defaults to None.
        axis (int, optional): Image axis represented by slider. Defaults to 0.
        value_range (Optional[Tuple[int, int]], optional): Value range of image. Defaults to (img.min, img.max).
        size (int, optional): Figure size. Defaults to 8.
        initial_val (int, optional): Initial slice index. Defaults to 0.
    """

    if mask is not None:
        assert (
            img.shape == mask.shape
        ), "CT volume shape does not match segmentation mask shape."

    use_widget()

    if isinstance(img, List):
        img = np.array(img)

    # Set figure pixel range
    if value_range is None:
        value_range = (img.min(), img.max())

    # Set figure size
    first_slice = img.take(initial_val, axis=axis)
    rows, cols = first_slice.shape
    fig, ax = plt.subplots(1, 1, figsize=(size, int(size / cols * rows)))

    # Set figure rubbish to False
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False

    # Show first slice
    img_ax = ax.imshow(
        first_slice,
        vmin=value_range[0],
        vmax=value_range[1],
        cmap="gray",
    )

    if mask is not None:
        curr_mask = mask.take(initial_val, axis=axis)
        mask_ax = ax.imshow(
            curr_mask,
            cmap="rainbow",
            alpha=(curr_mask > 0) * 0.3,
            vmin=0,
            vmax=mask.max(),
        )

    # Update
    def update(val):
        img_ax.set_data(img.take(int(val), axis=axis))
        if mask is not None:
            curr_mask = mask.take(val, axis=axis)
            mask_ax.set_data(curr_mask)
            mask_ax.set_alpha((curr_mask > 0) * 0.3)
        fig.canvas.draw_idle()

    interact(update, val=IntSlider(min=0, max=img.shape[axis] - 1, value=initial_val))
