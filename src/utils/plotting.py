import torch
import wandb
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import Union, Optional
from pathlib import Path
import io


def plot_component_normals(normals, mask=None, save_dir=None, name="component_normals"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, ax in enumerate(axes):
        normal_img = normals[i].numpy()
        if mask is not None:
            normal_img[~mask.numpy()] = np.nan
        ax.imshow(normal_img, cmap="jet")
        ax.set_title(f"N_{['x', 'y', 'z'][i]}")
        ax.axis("off")
    if save_dir and name:
        pth = save_dir + f"/{name}.png"
        plt.savefig(pth)
        # wandb.log({"component_normals": wandb.Image(plt.imread(pth))})
    plt.show()


def plot_2d(img, title: str, save_path: str, show: bool = False):
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()

    plt.imshow(img)
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()


def save_init(img: np.ndarray, save_path: str, title="input") -> None:
    """
    TODO: just change to plot?
    """
    plt.imshow(img)
    plt.axis("off")
    plt.title(title)
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_normals(n, save_path=None):
    """
    TODO: find a way to combine this wih the save_init.py function. reuse stuff.
    """
    normals = np.asarray(n, dtype=np.float32)
    assert normals.shape[-1] == 3, "Expected normals with 3 channels (x, y, z)"

    rgb = (normals * 0.5 + 0.5).clip(0, 1)

    fig, ax = plt.subplots()
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title("Normal Map")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
    else:
        plt.show()

    # Individual components
    labels = ["X", "Y", "Z"]
    for i, label in enumerate(labels):
        comp = (normals[..., i] * 0.5 + 0.5).clip(0, 1)
        fig, ax = plt.subplots()
        ax.imshow(comp, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"{label}-component")

        if save_path:
            path = save_path.replace(".png", f"_{label}.png")
            fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
        else:
            plt.show()


# def plot_normals(normals: torch.Tensor, title="Normal Map", save_img=None, log=False):
#     """
#     Visualizes normals as a color image.
#     Expects normals to be a (H, W, 3) tensor with values in [-1, 1].
#     """
#     normals_vis = (normals + 1) / 2
#     normals_vis = normals_vis.clamp(0, 1).detach().cpu().numpy()
#
#     plt.figure(figsize=(6, 6))
#     plt.imshow(normals_vis)
#     # plt.title(title)
#     plt.axis("off")
#     if save_img is not None:
#         pth = f"../out/{save_img}.png"
#         plt.savefig(pth)
#     # if log:
#     # w
#     # wandb.log({"normals_vis": wandb.Image(pth)})
#     plt.show()
#


def plot_surface(Z, elev=30, azim=45, name="surface_3d", show=False, log_wandb=False):
    H, W = Z.shape
    X, Y = torch.meshgrid(
        torch.arange(W, device=Z.device),
        torch.arange(H, device=Z.device),
        indexing="xy",
    )

    Z_cpu = Z.detach().cpu().numpy()
    X = X.cpu().numpy()
    Y = Y.cpu().numpy()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z_cpu, cmap="viridis", edgecolor="none")

    ax.view_init(elev=elev, azim=azim)
    ax.set_title(name)
    ax.set_aspect("equal")  # just added
    # ax.set_box_aspect([1,1,1])
    # plotly stuffs
    plotly_fig = go.Figure(
        data=[go.Surface(z=Z_cpu, colorscale="Viridis")],
    )
    plotly_fig.update_layout(
        title="3D Deph map",
        width=800,
        margin=dict(l=10, r=10, b=10, t=30),
    )

    if name:
        pth = f"out/{name}.png"
        plt.savefig(pth)
    if show:
        plt.show()
    if log_wandb:
        wandb.log({"surface_3d": wandb.Plotly(plotly_fig)})
        # plt.show()


def animate_surface_snapshots(z_snapshots, interval=200, title="3D Surface Evolution"):
    """
    Animates a list of Z snapshots (torch.Tensor or np.ndarray) as 3D surfaces using Plotly.

    Args:
        z_snapshots: list of (H, W) torch.Tensor or np.ndarray
        interval: time between frames in ms
        title: plot title
    """
    # Convert to numpy if needed
    z_snapshots = [
        z.detach().cpu().numpy() if isinstance(z, torch.Tensor) else z
        for z in z_snapshots
    ]
    H, W = z_snapshots[0].shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    frames = [
        go.Frame(data=[go.Surface(z=z, x=X, y=Y, colorscale="Viridis")], name=str(i))
        for i, z in enumerate(z_snapshots)
    ]

    fig = go.Figure(
        data=[go.Surface(z=z_snapshots[0], x=X, y=Y, colorscale="Viridis")],
        frames=frames,
    )

    fig.update_layout(
        title=title,
        updatemenus=[
            {
                "type": "buttons",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": interval, "redraw": True},
                                "fromcurrent": True,
                            },
                        ],
                    }
                ],
            }
        ],
    )

    fig.show()


def animate_surface_snapshots_matplotlib(
    z_snapshots, interval=200, title="3D Surface Evolution"
):
    """
    Animates a list of Z snapshots (torch.Tensor or np.ndarray) as 3D surfaces using Matplotlib.

    Args:
        z_snapshots: list of (H, W) torch.Tensor or np.ndarray
        interval: time between frames in ms
        title: plot title
    """
    z_snapshots = [
        z.detach().cpu().numpy() if isinstance(z, torch.Tensor) else z
        for z in z_snapshots
    ]
    H, W = z_snapshots[0].shape
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = [ax.plot_surface(X, Y, z_snapshots[0], cmap="viridis", edgecolor="none")]

    def update(frame):
        ax.clear()
        surf[0] = ax.plot_surface(
            X, Y, z_snapshots[frame], cmap="viridis", edgecolor="none"
        )
        ax.set_title(f"{title} (Frame {frame})")
        ax.set_zlim(np.min(z_snapshots[0]), np.max(z_snapshots[0]))

    # pyrefly: ignore [bad-argument-type]
    ani = FuncAnimation(fig, update, frames=len(z_snapshots), interval=interval)
    plt.show()


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
    step: int = 8
    # downsampling
    if hide_invalid:
        assert mask is not None, "Provide a boolean mask"
        depth = np.where(mask, depth, np.nan)

    depth = depth[::step, ::step].astype(np.float32)
    # if decimals is not None:
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

    # fig.show()
    if save_path:
        # pyrefly: ignore [bad-assignment]
        save_path = Path(save_path)
        fig.write_html(str(save_path), include_plotlyjs="cdn")

    if show:
        fig.show()

    if log:
        img_bytes = fig.to_image(format="png")
        img = io.BytesIO(img_bytes)
        wandb.log({"plotly_depth": wandb.Image(img, caption="3D Depth Surface")})
