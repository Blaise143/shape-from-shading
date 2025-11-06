# ruff: noqa: F405, E742
import torch
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import wandb
import numpy as np
from utils.grad_helpers import *
from datetime import datetime
from config import SFSConfig
from typing import Optional


def get_img_and_mask(img_path: str, kind: str, mask_path: Optional[str] = None):
    assert kind in ["sphere", "cube", "face", "ones"], "kind is invalid"

    img = Image.open(img_path).convert("L")
    img_np = np.array(img)

    if kind == "cube":
        if mask_path is None:
            raise RuntimeError("Provide a valid mask path for a cube")
        else:
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask) > 0
    elif kind == "face":
        if mask_path is None:
            raise RuntimeError("mask invalid")
        else:
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask) > 0
    else:
        mask = img_np > 0

    if kind == "ones":
        mask = np.ones_like(img_np)

    return img_np, mask


def shape_from_shading(
    cfg: SFSConfig, mask_pth: Optional[str] = None, kind: str = "face"
):
    """
    When mask None -> sphere. when it isnt: cube
    """

    device = get_device()
    img_np, mask = get_img_and_mask(cfg.image_path, mask_path=mask_pth, kind=kind)
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask).to(device)
    cfg.save_params(f"{cfg.save_to}/params.json")
    # img = Image.open(cfg.image_path).convert("L")
    # img_np = np.array(img)
    if cfg.save_to:
        save_init(img=img_np, save_path=cfg.save_to + "/input.png")

    img_tensor = T.ToTensor()(img_np).squeeze(0)
    print("Img_tensor, img ")
    img_tensor = img_tensor / img_tensor.max()
    I = img_tensor.to(device)
    H, W = I.shape
    print(f"I info {I.min(), I.max()}")
    # init params:
    init_kind = cfg.init_kind
    raw = get_init(
        shape=(H, W),
        image_np=img_np,
        kind=init_kind,  # "init_grads"
    )
    wandb.init(project="sfs")
    if cfg.light_direction is None:
        light_dir_ = torch.tensor([0.3, 0.3, 0.9], device=device)
    else:
        light_dir_ = cfg.light_direction.to(device)
    rho = 1.0  # 0.5
    albedo = torch.nn.Parameter(torch.tensor(rho, device=device))
    if cfg.optimize_light:
        light_dir = torch.nn.Parameter(light_dir_.clone().detach())
        optimizables = (
            [raw, light_dir, albedo] if cfg.optimize_albedo else [raw, light_dir]
        )
        opt = torch.optim.Adam(optimizables, lr=cfg.lr)
    else:
        light_dir = light_dir_
        optimizables = [raw, albedo] if cfg.optimize_albedo else [raw]
        opt = torch.optim.Adam(optimizables, lr=cfg.lr)

    def l2(x, mask=None):
        if mask is not None:
            x = x * mask[..., None]
        grads = torch.gradient(x, dim=(0, 1))  # grad_x , grad_y
        return sum((g**2).mean() for g in grads)

    #    mask = (img_tensor > 0).to(device)
    plot_normals(
        raw.detach().cpu().clone(), save_path=f"{cfg.save_to}/initialization.png"
    )
    for _ in tqdm(range(cfg.num_iters)):
        opt.zero_grad()

        n = raw / (raw.norm(dim=-1, keepdim=True) + 1e-8)
        L = light_dir / light_dir.norm()
        if _ in [1, 10, 15, 20, 25, 100, 200, 1000, 2000, 3000, 4000, 5000]:
            plot_normals(
                n.detach().cpu().clone(), save_path=f"{cfg.save_to}/normals{_}.png"
            )
        R = (n * L.view(1, 1, 3)).sum(dim=-1).relu()
        if cfg.use_mask:
            data_loss = (R - (I * mask)).pow(2).mean()
        else:
            data_loss = (R - I).pow(2).mean()
        smooth_loss = cfg.lambda_smooth * l2(n, mask=mask)  # TODO: DELETE AFTER
        loss = data_loss + smooth_loss
        wandb.log(
            {
                "data_loss": data_loss.item(),
                "smooth_loss": smooth_loss,  # .item(),
                "total_loss": loss.item(),
            }
        )

        loss.backward()
        opt.step()

    normals = (raw / (raw.norm(dim=-1, keepdim=True) + 1e-8)).detach().cpu()
    L = light_dir / light_dir.norm()
    reflectance = albedo.cpu() * (normals.cpu() * L.view(1, 1, 3).cpu()).sum(dim=-1)
    if cfg.optimize_light:
        print(f"Optimized light direction: {L}")
    if cfg.optimize_albedo:
        print(f"optimized albedo: {albedo}")
    return normals, reflectance, mask.cpu().numpy()


if __name__ == "__main__":
    from utils.helpers import integrate_normals
    from utils.plot_helpers import plot_2d, plot_depth_3d_plotly

    import os
    import numpy as np

    DATA = {
        "sphere": "GT_data4/Cue1/sphere.png",
        "cylinder": "GT_data4/Cue1/cylinder.png",
        "cube": "GT_data4/Cue1/cube.png",
        "pyramid": "GT_data4/Cue1/pyramid.png",
        "face": "images/face.png",
    }
    MASKS = {
        "sphere": "GT_data4/Masks/sphere.png",
        "cylinder": "GT_data4/Masks/cylinder.png",
        "pyramid": "GT_data4/Masks/pyramid.png",
        "cube": "GT_data4/Masks/cube.png",
        "face": "images/image_mask.png",
    }
    ### dirs: out/normals/face, out/normals/sphere
    # img_pth = "images/sphere.png"  # range between 0 and about 0.5
    # msk_pth = "images/image_mask.png"
    target = "face"
    img_pth = DATA[target]
    msk_pth = MASKS[target]
    img, mask = get_img_and_mask(img_path=img_pth, kind="face", mask_path=msk_pth)
    folder_name = folder_name = (
        "out/"
        + img_pth.split("/")[-1].split(".")[0].replace("/", "-")
        + "/"
        + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )
    os.makedirs(folder_name, exist_ok=True)
    config = SFSConfig(
        image_path=img_pth,
        num_iters=2_000,
        lambda_smooth=1.0,
        lr=1e-2,
        optimize_light=True,
        light_direction=torch.tensor([0, 0, 1.0]),  # [0.045, 0.659, 0.715]),
        save_to=folder_name,
        optimize_albedo=False,
        use_mask=True,
        init_kind="init_grads",  # init_grads, zeroes, sphere, ones
    )
    config.save_params(path=f"{folder_name}/params.json")
    normals, reflectance, mask = shape_from_shading(
        config, mask_pth=msk_pth, kind="face"
    )
    plot_normals(normals, save_path=f"{folder_name}/final_normals.png")
    z = integrate_normals(normals.numpy() * mask[..., None])
    print("Inspect Z!")
    breakpoint()
    plot_depth_3d_plotly(
        depth=z,
        save_path=f"{folder_name}/depth.html",
        show=True,
        normalize_xyz=False,
        mask=mask,
        hide_invalid=True,
    )
    plot_2d(
        reflectance, title="reflectance", save_path=f"{folder_name}/reflectance.png"
    )
