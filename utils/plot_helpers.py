import numpy as np
from pathlib import Path
from typing import Optional, Union
from skimage.restoration import denoise_tv_chambolle
import matplotlib.pyplot as plt
import torch
import wandb
import io
import plotly.graph_objects as go


def plot_2d(img, title: str, save_path: str, show: bool = False):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    plt.imshow(img)
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()


def adjusted_depth(depth, mask):
    """
    Create a complete surface where the background is a fitted plane and
    valid points are kept from the original depth map.

    Parameters
    ----------
    depth : ndarray of shape (H, W)
        Depth map with some invalid values.
    mask : ndarray of shape (H, W)
        Boolean mask of valid (True) depth values.

    Returns
    -------
    surface : ndarray of shape (H, W)
        Combined surface: plane as background, valid depth where available.
    plane_depth : ndarray of shape (H, W)
        The fitted and shifted background plane.
    normal : ndarray of shape (3,)
        Unit normal vector of the plane orientation.
    """

    H, W = depth.shape
    y, x = np.indices((H, W))
    x_valid = x[mask]
    y_valid = y[mask]
    z_valid = depth[mask]

    # Fit plane: z = a*x + b*y + c
    A = np.stack((x_valid, y_valid, np.ones_like(x_valid)), axis=1)
    coeff, _, _, _ = np.linalg.lstsq(A, z_valid, rcond=None)
    a, b, c = coeff

    # Compute plane depth map
    plane_depth = a * x + b * y + c
    # ---
    y_min, x_min = np.unravel_index(np.argmin(depth * mask), depth.shape)
    # evaluate plane at that pixel
    # plane_val = a * x_min + b * y_min + c
    # shift = depth[y_min, x_min] - plane_val
    # plane_depth += shift
    # ---
    # Shift so min(plane_depth[mask]) = min(z_valid)
    shift = z_valid.min() - plane_depth[mask].min()
    plane_depth += shift

    # Compute normal
    normal = np.array([a, b, -1.0])
    normal /= np.linalg.norm(normal)

    # Combine plane background and valid depth points
    surface = plane_depth.copy()
    surface[mask] = depth[mask]

    return surface, plane_depth, normal


def plot_depth_3d_plotly(
    depth: Union[torch.Tensor, np.ndarray],
    save_path: Optional[str] = None,
    show: bool = False,
    log: bool = False,
    normalize_xyz: bool = False,
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    hide_invalid: bool = True,
    adjust_depth_map: bool = False,
):
    """
    - if show is true, shows the 3d plot
    - If save_path is specified, saves to path
    - if log is true, it logs to wandb
    """

    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        if adjust_depth_map:
            depth, _, _ = adjusted_depth(depth=depth, mask=mask)
    step: int = 8
    # downsampling
    if hide_invalid:
        assert mask is not None, "Provide a boolean mask"
        depth = np.where(mask, depth, np.nan)

    depth = depth[::step, ::step].astype(np.float32)
    # if decimals is not None:
    depth = np.round(depth, 3)

    n, m = depth.shape
    # X, Y = np.meshgrid(np.arange(m), np.arange(n))
    # X, Y = np.meshgrid(np.linspace(-1, 1, m), np.linspace(-1, 1, n))

    X, Y = np.meshgrid(np.linspace(-1, 1, m), np.linspace(-1, 1, n), indexing="xy")
    # X, Y = X / (m - 1), Y / (n - 1)  # Trial, might delete later

    # if hide_invalid and mask is not None:
    #     mask = mask[::step, ::step].astype(bool)
    #     depth = np.where(mask, depth, np.nan)
    #
    if normalize_xyz:
        X = X / (m - 1)
        Y = Y / (n - 1)
        z_min, z_max = np.nanmin(depth), np.nanmax(depth)
        if z_max > z_min:
            depth = (depth - z_min) / (z_max - z_min)
        else:
            depth = np.zeros_like(depth)

    # if mask is not None:  # ADDED for plotting
    #     mask = mask[::step, ::step].astype(np.float32)
    #     flat_plane_value = np.nanmin(depth)  # might experimennt with other stuffs
    #     depth = np.where(mask, depth, flat_plane_value)
    #
    fig = go.Figure(
        data=[go.Surface(z=depth, x=X, y=Y, colorscale="Viridis", showscale=True)]
    )

    fig.update_layout(
        title="3D Depth Surface",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Depth",
            aspectmode="data",
            # aspectmode="manual",
            aspectratio=dict(x=2, y=2, z=1),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    # fig.show()
    if save_path:
        save_path = Path(save_path)
        fig.write_html(str(save_path), include_plotlyjs="cdn")

    if show:
        fig.show()

    if log:
        img_bytes = fig.to_image(format="png")
        img = io.BytesIO(img_bytes)
        wandb.log({"plotly_depth": wandb.Image(img, caption="3D Depth Surface")})
