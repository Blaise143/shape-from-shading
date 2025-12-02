import torch
import wandb
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from typing import Union, Optional
from pathlib import Path
import io
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
    """Plots image, optionally saves it"""
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
    """Save image, optionally show it"""
    plot_image(
        img=img,
        title=title,
        save_path=save_path,
        show=show,
        hide_axis=False,
    )


def save_init(img: TensorLike, save_path: PathLike, title: str = "input") -> None:
    """
    save initialization
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
    title: Optional[str] = "Normal Map",
) -> None:
    """
    Plot a normal map (RGB) and, optionally, its individual XYZ components.
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


def plot_depth_3d_plotly(
    depth: Union[torch.Tensor, np.ndarray],
    save_path: Optional[str] = None,
    show: bool = False,
    log: bool = False,
    normalize_xyz: bool = False,
    mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    hide_invalid: bool = True,
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
