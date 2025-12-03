import torch
from typing import Tuple, Literal, Optional
from src.config import SFSConfig


def l2_smoothness(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculate the smoothness loss, the L2 norm of the partial derivatives of x.
    The domain of where to take the loss is specified by mask
    """
    if mask is not None:
        x = x * mask[..., None]
    grads = torch.gradient(x, dim=(0, 1))  # grad_x , grad_y
    loss = torch.zeros((), dtype=x.dtype, device=x.device)
    for g in grads:
        loss = loss + (g**2).mean()
    return loss


def normalize_normals(n: torch.Tensor) -> torch.Tensor:
    """
    Normalize per-pixel normals along the last dimension.
    """
    return n / (n.norm(dim=-1, keepdim=True) + 1e-8)


def get_light_direction(
    cfg: SFSConfig, device: Literal["cuda", "mps", "cpu"]
) -> torch.Tensor:
    """
    Returns a default light direction of [0.3,0.3,0.9] if the cfg doesnt have one already.
    """
    if cfg.light_direction is None:
        return torch.tensor([0.3, 0.3, 0.9], device=device)
    return cfg.light_direction.to(device)


def build_optimizer(
    cfg: SFSConfig,
    optim_var: torch.nn.Parameter,
    light_dir_: torch.Tensor,
    device: Literal["cuda", "mps", "cpu"],
) -> Tuple[torch.optim.Optimizer, torch.Tensor, Optional[torch.nn.Parameter]]:
    """
    Build optimizer and (optionally) learnable light + albedo.
    """
    rho = 1.0
    albedo: torch.nn.Parameter = torch.nn.Parameter(torch.tensor(rho, device=device))

    if cfg.optimize_light:
        light_dir = torch.nn.Parameter(light_dir_.clone().detach())
        optimizables = [optim_var, light_dir]
        if cfg.optimize_albedo:
            optimizables.append(albedo)
    else:
        light_dir = light_dir_
        if cfg.optimize_albedo:
            optimizables = [optim_var, albedo]
        else:
            optimizables = [optim_var]

    opt = torch.optim.Adam(optimizables, lr=cfg.lr)
    return opt, light_dir, albedo
