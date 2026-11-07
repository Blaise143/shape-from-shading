import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from typing import Union, Optional
from pathlib import Path

TensorLike = Union[np.ndarray, torch.Tensor]
PathLike = Union[str, Path]


def to_numpy(x: TensorLike) -> np.ndarray:
    """Convert to a numpy array, detaching/moving to CPU if it's a tensor.

    Args:
        x: A numpy array or torch tensor.

    Returns:
        np.ndarray: `x` unchanged if already a numpy array, otherwise
        ``x.detach().cpu().numpy()``.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return x


def apply_mask(img: np.ndarray, mask: TensorLike):
    """
    Given an image, apply a boolean mask. everything outside will be set to np.nan

    Args:
        img: Image array, shape ``(H, W)`` or ``(H, W, C)``.
        mask: Boolean (or tensor convertible to boolean) mask of shape
            ``(H, W)``; pixels where `mask` is falsy are set to NaN.

    Returns:
        np.ndarray: Copy of `img` with pixels outside `mask` set to NaN.
    """
    mask_np = to_numpy(mask).astype(bool)
    out = img.copy()
    if out.ndim == 3:
        out[~mask_np, :] = np.nan
    else:
        out[~mask_np] = np.nan
    return out


def save_or_show(
    fig: Figure,
    save_path: Optional[PathLike] = None,
    show: bool = False,
    pad_inches: float = 0.0,
) -> None:
    """
    Handle saving and/or showing a matplotlib figure.
    if show is true, it shows the plot. When save plath is available, it saves it to path

    Args:
        fig: The matplotlib figure to save/show.
        save_path: If given, the figure is saved to this path.
        show: If True, displays the figure via ``plt.show()``; otherwise it
            is closed (to free memory) after any save.
        pad_inches: Padding passed to ``fig.savefig``.
    """
    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, bbox_inches="tight", pad_inches=pad_inches)
    if show:
        plt.show()
    else:
        plt.close(fig)


def normals_to_rgb(normals: TensorLike) -> np.ndarray:
    """
    Convert raw normals to range [0,1] for visualization.

    Maps each component from ``[-1, 1]`` to ``[0, 1]`` (``(n + 1) / 2``) so
    the normal map can be displayed as an RGB image.

    Args:
        normals: Normal map with a trailing channel dim of size 3.

    Returns:
        np.ndarray: Normals rescaled to ``[0, 1]`` and clipped.

    Raises:
        AssertionError: If the last dimension of `normals` isn't 3.
    """
    assert normals.shape[-1] == 3, (
        f"Expected normals with 3 channels but got {normals.shape}"
    )
    np_normals = (to_numpy(normals).astype(np.float32) + 1) / 2
    return np_normals.clip(0.0, 1.0)
