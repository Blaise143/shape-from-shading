import os
import torch
import numpy as np
from src.config import SFSConfig
from typing import Optional
from src.data_load import ShapeFromShadingDataset
from src.sfs import shape_from_shading
from typing import Literal
from utils import (
    plot_normals,
    integrate_normals,
    plot_depth_3d_plotly,
    plot_2d,
    save_depth_png,
)


def save_results(
    folder: str,
    normals: torch.Tensor,
    reflectance: torch.Tensor,
    mask: np.ndarray,
) -> None:
    """
    Save normals (png/npy/pt), integrated depth (pt/png/html), and reflectance (png)
    for a single object's result into `folder`.

    Also integrates the normals into a depth map (via
    :func:`utils.integrator.integrate_normals`) before saving it.

    Args:
        folder: Output directory for this object's result files.
        normals: Final normal map, shape ``(H, W, 3)``.
        reflectance: Final reflectance map, shape ``(H, W)``.
        mask: Boolean mask of valid pixels, shape ``(H, W)``.
    """
    plot_normals(normals, save_path=os.path.join(folder, "normals.png"))
    np.save(os.path.join(folder, "normals.npy"), normals.numpy())
    torch.save(normals, os.path.join(folder, "normal.pt"))

    z = integrate_normals(normals.numpy() * mask[..., None])
    depth = torch.from_numpy(z)
    torch.save(depth, os.path.join(folder, "depth.pt"))
    save_depth_png(z, mask, os.path.join(folder, "depth.png"))
    plot_depth_3d_plotly(
        depth=z,
        save_path=os.path.join(folder, "depth.html"),
        show=False,
        normalize_xyz=False,
        mask=mask,
        hide_invalid=True,
    )

    plot_2d(
        reflectance,
        title="reflectance",
        save_path=os.path.join(folder, "reflectance.png"),
    )


def run(
    dataset: ShapeFromShadingDataset,
    out_path: str = "out",
    target: Optional[str] = None,
    num_iters: int = 2000,
    lr: float = 1e-2,
    lambda_smooth: float = 1.0,
    optimize_light: bool = True,
    light_direction: torch.Tensor = torch.tensor([0.0, 0.0, 1]),
    optimize_albedo: bool = False,
    init_kind: Literal["init_grads", "zeroes", "ones", "sphere"] = "init_grads",
    parametrization_type: Literal["cartesian", "stereographic"] = "cartesian",
    use_wandb: bool = False,
    log_intermediates: bool = False,
):
    """Run Shape-from-Shading over one or all objects in `dataset`.

    For each selected object, builds an :class:`~src.config.SFSConfig`,
    runs :func:`src.sfs.shape_from_shading`, and saves the results via
    :func:`save_results` into ``{out_path}/{object_name}/``.

    Args:
        dataset: Indexed image/mask pairs to run on.
        out_path: Base output directory; a subfolder is created per object.
        target: If given, only this object is run; otherwise every object in
            `dataset` is run.
        num_iters: Number of optimizer iterations per object.
        lr: Learning rate for the Adam optimizer.
        lambda_smooth: Weight of the normal-smoothness regularizer.
        optimize_light: Whether to jointly optimize the light direction.
        light_direction: Initial light direction, shape ``(3,)``.
        optimize_albedo: Whether to jointly optimize albedo.
        init_kind: Strategy used to initialize the normal map.
        parametrization_type: Whether normals are optimized directly
            (``"cartesian"``) or via a stereographic projection
            (``"stereographic"``).
        use_wandb: Whether to log losses to Weights & Biases.
        log_intermediates: Whether to periodically save intermediate normal
            maps during optimization.
    """
    os.makedirs(out_path, exist_ok=True)

    if target is not None:
        objects = [target]
    else:
        objects = dataset.list_objects()

    for obj in objects:
        print(f"\n Running Shape-from-Shading for '{obj}'...")

        try:
            paths = dataset.get(obj)
        except ValueError as e:
            print(f"Skipping '{obj}': {e}")
            continue

        img_pth, msk_pth = paths["image"], paths["mask"]

        if not os.path.exists(img_pth) or not os.path.exists(msk_pth):
            print(f"Skipping '{obj}' (missing image or mask)")
            continue

        folder_name = os.path.join(out_path, obj)
        os.makedirs(folder_name, exist_ok=True)

        config = SFSConfig(
            image_path=img_pth,
            num_iters=num_iters,
            lambda_smooth=lambda_smooth,
            lr=lr,
            optimize_light=optimize_light,
            light_direction=light_direction,
            save_to=folder_name,
            optimize_albedo=optimize_albedo,
            use_mask=True,
            init_kind=init_kind,
            parametrization_type=parametrization_type,
        )
        config.save_params(path=os.path.join(folder_name, "params.json"))

        normals, reflectance, mask = shape_from_shading(
            config,
            mask_pth=msk_pth,
            use_wandb=use_wandb,
            log_intermediates=log_intermediates,
        )

        save_results(folder_name, normals, reflectance, mask)

        print(f" Finished '{obj}'. Results saved to {folder_name}")
