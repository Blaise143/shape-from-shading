import torch
import wandb
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from typing import Union, Optional
from pathlib import Path
import io
import imageio
from utils.plotting.helpers import to_numpy, apply_mask, save_or_show, normals_to_rgb

TensorLike = Union[np.ndarray, torch.Tensor]
PathLike = Union[str, Path]


def plot_image(
    img: TensorLike,
    title: Optional[str] = None,
    save_path: Optional[PathLike] = None,
    show: bool = False,
    cmap: Optional[str] = None,
    hide_axis: bool = False,
    pad_inches: float = 0.0,
) -> None:
    """Plot an image with matplotlib, optionally saving and/or showing it.

    Args:
        img: Image to plot (numpy array or torch tensor).
        title: Optional axes title.
        save_path: If given, the figure is saved to this path.
        show: If True, displays the figure.
        cmap: Optional matplotlib colormap name.
        hide_axis: If True, hides the axis ticks/frame.
        pad_inches: Padding passed to ``fig.savefig``.
    """
    img_np = to_numpy(img)
    fig, ax = plt.subplots()
    ax.imshow(img_np, cmap=cmap)
    if title is not None:
        ax.set_title(title)
    if hide_axis:
        ax.axis("off")

    save_or_show(fig, save_path=save_path, show=show, pad_inches=pad_inches)


def plot_2d(
    img: TensorLike, title: str, save_path: Optional[PathLike], show: bool = False
) -> None:
    """Plot a 2D image with a title and visible axis; optionally save/show it.

    A thin wrapper around :func:`plot_image` with ``hide_axis=False``.

    Args:
        img: Image to plot.
        title: Axes title.
        save_path: If given, the figure is saved to this path.
        show: If True, displays the figure.
    """
    plot_image(
        img=img,
        title=title,
        save_path=save_path,
        show=show,
        hide_axis=False,
    )


def save_init(img: TensorLike, save_path: PathLike, title: str = "input") -> None:
    """Save a snapshot of the input image, axis hidden (used for the run's `input.png`).

    A thin wrapper around :func:`plot_image` with ``hide_axis=True`` and no
    on-screen display.

    Args:
        img: Image to save.
        save_path: Destination path for the saved figure.
        title: Axes title (not visible since the axis is hidden, but kept
            for consistency with the other plotting helpers).
    """
    plot_image(
        img=img,
        title=title,
        save_path=save_path,
        show=False,
        hide_axis=True,
        pad_inches=0.0,
    )


def plot_normals(
    normals: TensorLike,
    save_path: Optional[PathLike] = None,
    show: bool = False,
    mask: Optional[TensorLike] = None,
    title: Optional[str] = None,
) -> None:
    """
    Plot a normal map, visualized as an RGB image.

    Converts `normals` to the ``[0, 1]`` RGB visualization range (via
    :func:`utils.plotting.helpers.normals_to_rgb`) and, if `mask` is given,
    sets pixels outside the mask to NaN so they render as blank. For a
    per-component (X/Y/Z) view instead, see :func:`plot_component_normals`.

    Args:
        normals: Normal map with a trailing channel dim of size 3.
        save_path: If given, the figure is saved to this path.
        show: If True, displays the figure.
        mask: Optional boolean mask of valid pixels.
        title: Optional axes title.
    """
    rgb = normals_to_rgb(normals)

    if mask is not None:
        mask_np = to_numpy(mask).astype(bool)
        rgb = apply_mask(rgb, mask_np)
    plot_image(
        img=rgb,
        title=title,
        save_path=save_path,
        show=show,
        hide_axis=True,
        pad_inches=0.1,
    )


def save_depth_png(depth: np.ndarray, mask: np.ndarray, out_path: str) -> None:
    """
    Save a depth map as a masked, normalized PNG.

    Depth values are normalized to ``[0, 255]`` and inverted (closer =
    brighter), with pixels outside the mask forced to white (255).

    Args:
        depth: Depth map, shape ``(H, W)``.
        mask: Boolean mask of valid pixels, shape ``(H, W)``.
        out_path: Destination path for the PNG.
    """
    depth_masked = np.where(mask, depth, np.nan)

    vmin = np.nanmin(depth_masked)
    vmax = np.nanmax(depth_masked)
    denom = (vmax - vmin) if vmax > vmin else 1.0

    depth_norm = (depth_masked - vmin) / denom
    depth_norm = 1.0 - depth_norm

    depth_img = np.nan_to_num(depth_norm, nan=0.0)
    depth_img = (depth_img * 255).astype(np.uint8)
    depth_img[~mask] = 255

    imageio.imwrite(out_path, depth_img)


def plot_depth_3d_plotly(
    depth: Union[torch.Tensor, np.ndarray],
    save_path: Optional[PathLike] = None,
    show: bool = False,
    log: bool = False,
    normalize_xyz: bool = False,
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    hide_invalid: bool = True,
):
    """Render a depth map as an interactive 3D surface with Plotly.

    The depth map is downsampled (every 8th pixel) for a lighter-weight
    plot. If `hide_invalid` is set, `mask` is required and invalid (masked
    out) pixels are set to NaN so they render as gaps in the surface.

    Args:
        depth: Depth map, shape ``(H, W)``.
        save_path: If given, saves the plot as an interactive HTML file at
            this path.
        show: If True, opens the plot in a browser via ``fig.show()``.
        log: If True, renders the figure to a PNG and logs it to wandb.
        normalize_xyz: If True, rescales X, Y, and depth into ``[0, 1]``.
        mask: Boolean mask of valid pixels; required when `hide_invalid`
            is True.
        hide_invalid: If True, masks out invalid pixels as NaN (gaps) in
            the surface. Requires `mask`.

    Raises:
        AssertionError: If `hide_invalid` is True but `mask` is None.
    """

    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().numpy()
    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
    step: int = 8
    if hide_invalid:
        assert mask is not None, "Provide a boolean mask"
        depth = np.where(mask, depth, np.nan)

    depth = depth[::step, ::step].astype(np.float32)
    depth = np.round(depth, 3)

    n, m = depth.shape

    X, Y = np.meshgrid(np.linspace(-1, 1, m), np.linspace(-1, 1, n), indexing="xy")
    if normalize_xyz:
        X = X / (m - 1)
        Y = Y / (n - 1)
        z_min, z_max = np.nanmin(depth), np.nanmax(depth)
        if z_max > z_min:
            depth = (depth - z_min) / (z_max - z_min)
        else:
            depth = np.zeros_like(depth)

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
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    if save_path:
        save_path = Path(save_path)
        fig.write_html(str(save_path), include_plotlyjs="cdn")

    if show:
        fig.show()

    if log:
        img_bytes = fig.to_image(format="png")
        img = io.BytesIO(img_bytes)
        wandb.log({"plotly_depth": wandb.Image(img, caption="3D Depth Surface")})


def plot_component_normals(
    normals: TensorLike,
    mask: TensorLike,
    save_dir: PathLike | None = None,
    name: str = "component_normals",
    show: bool = False,
) -> None:
    """
    Plot per-component normal maps in a single row of subplots.
    Expects normals in shape (3, H, W) or (H, W, 3).
    Assumes mask is ALWAYS provided and is (H, W).

    Args:
        normals: Normal map, shape ``(3, H, W)`` or ``(H, W, 3)``.
        mask: Boolean mask of valid pixels, shape ``(H, W)``; pixels outside
            it are set to NaN in each component plot.
        save_dir: If given (along with `name`), saves the figure to
            ``{save_dir}/{name}.png``.
        name: Filename (without extension) to save as, when `save_dir` is
            given.
        show: If True, displays the figure; otherwise it is closed after
            any save.

    Raises:
        ValueError: If `normals` isn't shape ``(3, H, W)`` or ``(H, W, 3)``.
    """
    n = to_numpy(normals)

    if n.ndim == 3 and n.shape[0] == 3:
        comps = [n[i] for i in range(3)]  # (H, W) each
    elif n.ndim == 3 and n.shape[-1] == 3:
        comps = [n[..., i] for i in range(3)]
    else:
        raise ValueError(
            f"Expected normals shape (3, H, W) or (H, W, 3), got {n.shape}"
        )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    labels = ["N_x", "N_y", "N_z"]

    for comp, ax, label in zip(comps, axes, labels):
        comp_masked = apply_mask(comp, mask)
        ax.imshow(comp_masked, cmap="jet")
        ax.set_title(label)
        ax.axis("off")

    if save_dir is not None and name:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{name}.png"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.0)

    if show:
        plt.show()
    else:
        plt.close(fig)
