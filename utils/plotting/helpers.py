import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from typing import Union, Optional
from pathlib import Path

TensorLike = Union[np.ndarray, torch.Tensor]
PathLike = Union[str, Path]


def to_numpy(x: TensorLike) -> np.ndarray:
    """Convert to numpy array if it is a tensor"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        return x


def apply_mask(img: np.ndarray, mask: TensorLike):
    """
    Given an image, apply a boolean mask. everything outside will be set to np.nan
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
    """
    assert normals.shape[-1] == 3, (
        f"Expected normals with 3 channels but got {normals.shape}"
    )
    np_normals = (to_numpy(normals).astype(np.float32) + 1) / 2
    return np_normals.clip(0.0, 1.0)
