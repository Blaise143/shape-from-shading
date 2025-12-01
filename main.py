from sfs import shape_from_shading
import torch
import numpy as np
from utils import *  # .grad_helpers import *
from config import SFSConfig
from typing import Optional
from data_load import ShapeFromShadingDataset
import os


def run(
    dataset,
    base_out_dir: str = "out",
    focus_object: Optional[str] = None,
    num_iters: int = 2000,
    lr: float = 1e-2,
    lambda_smooth: float = 1.0,
    optimize_light: bool = True,
    light_direction: torch.Tensor = torch.tensor([0.0, 0.0, 1]),
    optimize_albedo: bool = False,
    init_kind: str = "init_grads",
    parametrization_type: str = "cartesian",
):
    for cue in dataset.cues:
        objects = [focus_object] if focus_object else dataset.list_objects(cue)

        for obj in objects:
            print(f"\n Running Shape-from-Shading for {cue}/{obj}...")

            paths = dataset.get(cue=cue, obj=obj)
            img_pth, msk_pth = paths["image"], paths["mask"]

            if not os.path.exists(img_pth) or msk_pth is None:
                print(f"Skipping {cue}/{obj} (missing image or mask)")
                continue

            folder_name = make_output_folder(base_dir=base_out_dir, cue=cue, obj=obj)

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
                # pyrefly: ignore [bad-argument-type]
                parametrization_type=parametrization_type,
            )
            config.save_params(path=f"{folder_name}/params.json")

            normals, reflectance, mask = shape_from_shading(
                config, mask_pth=msk_pth, kind="custom_mask"
            )

            plot_normals(normals, save_path=f"{folder_name}/normals.png")
            np.save(f"{folder_name}/normals.npy", normals.numpy())
            z = integrate_normals(normals.numpy() * mask[..., None])
            depth = torch.from_numpy(z)
            torch.save(depth, f"{folder_name}/depth.pt")
            torch.save(normals, f"{folder_name}/normal.pt")

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

            print(f" Finished {cue}/{obj}. Results saved to {folder_name}")


if __name__ == "__main__":
    dataset = ShapeFromShadingDataset(root_dir="GT_data4")
    run(
        dataset,
        base_out_dir="stereo_outputs",
        focus_object=None,  # sphere, pyramid, None, cube, ball
        num_iters=2000,
        lr=0.01,
        lambda_smooth=1,
        optimize_light=True,
        light_direction=torch.tensor([0, 0, 1.0]),
        optimize_albedo=False,
        init_kind="init_grads",
        parametrization_type="stereographic",
    )
