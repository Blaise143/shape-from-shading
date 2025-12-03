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
):
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
        config.save_params(path=f"{folder_name}/params.json")

        normals, reflectance, mask = shape_from_shading(
            config, mask_pth=msk_pth, kind="custom_mask"
        )

        plot_normals(normals, save_path=f"{folder_name}/normals.png")
        np.save(f"{folder_name}/normals.npy", normals.numpy())
        torch.save(normals, f"{folder_name}/normal.pt")

        z = integrate_normals(normals.numpy() * mask[..., None])
        depth = torch.from_numpy(z)
        torch.save(depth, f"{folder_name}/depth.pt")
        save_depth_png(z, mask, os.path.join(folder_name, "depth.png"))
        plot_depth_3d_plotly(
            depth=z,
            save_path=f"{folder_name}/depth.html",
            show=False,
            normalize_xyz=False,
            mask=mask,
            hide_invalid=True,
        )

        plot_2d(
            reflectance,
            title="reflectance",
            save_path=f"{folder_name}/reflectance.png",
        )

        print("final inspection: normals, mask, z")
        print(f" Finished '{obj}'. Results saved to {folder_name}")
