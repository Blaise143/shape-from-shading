# ruff: noqa: F404, E742
# noqa F405
import torch
import torchvision.transforms as T
from tqdm import tqdm
import wandb
import numpy as np
from src.config import SFSConfig

from utils import (
    get_device,
    get_img_and_mask,
    save_init,
    get_init,
    normals_to_fg,
    fg_to_normals,
    plot_normals,
)
from src.core_utils import (
    l2_smoothness,
    normalize_normals,
    build_optimizer,
    get_light_direction,
)


def shape_from_shading(
    cfg: SFSConfig,
    mask_pth: str,
    use_wandb: bool = False,
    log_intermediates: bool = False,
):
    """
    Main SFS optimization loop.

    Loads the image/mask, initializes a normal map, and optimizes it (plus
    optionally light direction and albedo) by gradient descent on a
    Lambertian reflectance loss with a smoothness regularizer:
    ``loss = (R - I)^2 + lambda_smooth * l2_smoothness(n)``, where
    ``R = max(0, n . L)`` is the predicted reflectance.

    Args:
        cfg: Run configuration; see :class:`~src.config.SFSConfig`.
        mask_pth: Path to the binary mask image for `cfg.image_path`.
        use_wandb: If True, logs per-iteration losses to Weights & Biases.
        log_intermediates: If True, saves a normal-map snapshot every 200
            iterations to ``{cfg.save_to}/normals{it}.png``.

    Returns:
        tuple: ``(normals, reflectance, mask)`` where

        - ``normals`` is the optimized, mask-zeroed normal map, shape
          ``(H, W, 3)``, on CPU.
        - ``reflectance`` is the predicted reflectance map, shape ``(H, W)``,
          on CPU.
        - ``mask`` is the boolean mask as a numpy array, shape ``(H, W)``.
    """
    device = get_device()

    img_np, mask = get_img_and_mask(cfg.image_path, mask_path=mask_pth)
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask).to(device)

    cfg.save_params(f"{cfg.save_to}/params.json")

    if cfg.save_to:
        save_init(img=img_np, save_path=f"{cfg.save_to}/input.png")

    img_tensor = T.ToTensor()(img_np).squeeze(0)
    img_tensor = img_tensor / img_tensor.max()
    I = img_tensor.to(device)
    H, W = I.shape

    raw = get_init(
        shape=(H, W),
        image_np=img_np,
        kind=cfg.init_kind,
        device=device,
    )

    if cfg.parametrization_type == "stereographic":
        fg = normals_to_fg(raw)
        optim_var = torch.nn.Parameter(fg)
    else:
        optim_var = torch.nn.Parameter(raw)

    light_dir_init = get_light_direction(cfg, device=device)
    opt, light_dir, albedo = build_optimizer(cfg, optim_var, light_dir_init, device)

    if use_wandb:
        if wandb.run is None:
            wandb.init(project="sfs")

    plot_normals(
        raw.detach().cpu().clone(),
        save_path=f"{cfg.save_to}/initialization.png",
    )
    print(f"Using parametrization_type: {cfg.parametrization_type}")

    for it in tqdm(range(cfg.num_iters)):
        opt.zero_grad()

        if cfg.parametrization_type == "stereographic":
            n = fg_to_normals(optim_var)
        else:
            n = normalize_normals(optim_var)

        L = light_dir / light_dir.norm()

        if log_intermediates and it % 200 == 0:
            plot_normals(
                n.detach().cpu().clone(),
                save_path=f"{cfg.save_to}/normals{it}.png",
            )

        R = (n * L.view(1, 1, 3)).sum(dim=-1).relu()

        if cfg.use_mask:
            data_loss = (R - (I * mask)).pow(2).mean()
        else:
            data_loss = (R - I).pow(2).mean()

        smooth_loss = cfg.lambda_smooth * l2_smoothness(n, mask=mask)
        loss = data_loss + smooth_loss

        if use_wandb:
            wandb.log(
                {
                    "data_loss": data_loss.item(),
                    "smooth_loss": smooth_loss.item(),
                    "total_loss": loss.item(),
                }
            )

        loss.backward()
        opt.step()

    if cfg.parametrization_type == "stereographic":
        normals = fg_to_normals(optim_var.detach()).cpu()
    else:
        normals = normalize_normals(optim_var.detach()).cpu()

    normals = normals * mask.detach().cpu()[..., None]

    L = light_dir / light_dir.norm()
    assert albedo is not None
    reflectance = albedo.cpu() * (normals * L.view(1, 1, 3).cpu()).sum(dim=-1)

    if cfg.optimize_light:
        print(f"Optimized light direction: {L}")
    if cfg.optimize_albedo:
        print(f"Optimized albedo: {albedo}")

    return normals, reflectance, mask.cpu().numpy()
