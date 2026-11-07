import torch
from typing import Tuple, Literal, Optional
from src.config import SFSConfig


def l2_smoothness(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute a smoothness regularizer: the mean squared spatial gradient of `x`.

    Args:
        x: Tensor of shape ``(H, W, ...)`` whose spatial smoothness is
            penalized (e.g. a normal map).
        mask: Optional boolean/float mask of shape ``(H, W)``. When given,
            `x` is zeroed outside the mask before taking gradients, so the
            loss is only (approximately) computed within the masked region.

    Returns:
        torch.Tensor: Scalar loss, the mean of ``grad_x**2`` and ``grad_y**2``.
    """
    if mask is not None:
        x = x * mask[..., None]
    grads = torch.gradient(x, dim=(0, 1))  # grad_x , grad_y
    loss = torch.zeros((), dtype=x.dtype, device=x.device)
    for g in grads:
        loss = loss + (g**2).mean()
    return loss


def normalize_normals(n: torch.Tensor) -> torch.Tensor:
    """Normalize per-pixel normals to unit length along the last dimension.

    Args:
        n: Normal map of shape ``(H, W, 3)`` (unnormalized).

    Returns:
        torch.Tensor: Normal map of the same shape, with each pixel's normal
        vector rescaled to unit norm (a small epsilon avoids division by zero).
    """
    return n / (n.norm(dim=-1, keepdim=True) + 1e-8)


def get_light_direction(
    cfg: SFSConfig, device: Literal["cuda", "mps", "cpu"]
) -> torch.Tensor:
    """Resolve the initial light direction to use for a run.

    Args:
        cfg: The run's config. If ``cfg.light_direction`` is set, that value
            is used; otherwise a default is returned.
        device: Device the returned tensor should live on.

    Returns:
        torch.Tensor: A 3-vector light direction on `device`, either
        ``cfg.light_direction`` or the default ``[0.0, 0.0, 1.0]``.
    """
    if cfg.light_direction is None:
        return torch.tensor([0.0, 0.0, 1.0], device=device)
    return cfg.light_direction.to(device)


def build_optimizer(
    cfg: SFSConfig,
    optim_var: torch.nn.Parameter,
    light_dir_: torch.Tensor,
    device: Literal["cuda", "mps", "cpu"],
) -> Tuple[torch.optim.Optimizer, torch.Tensor, Optional[torch.nn.Parameter]]:
    """Build the Adam optimizer, selecting which quantities are learnable.

    `optim_var` (the normal-map parameter being optimized) is always
    included. Whether `light_dir_` and/or albedo are also added as learnable
    parameters depends on `cfg.optimize_light` and `cfg.optimize_albedo`.

    Args:
        cfg: The run's config; reads ``cfg.optimize_light``,
            ``cfg.optimize_albedo``, and ``cfg.lr``.
        optim_var: The primary tensor being optimized (raw normals or
            stereographic ``(f, g)`` parameters), already wrapped as a
            ``torch.nn.Parameter``.
        light_dir_: Initial light direction, e.g. from
            :func:`get_light_direction`.
        device: Device to place the albedo parameter on.

    Returns:
        tuple: ``(opt, light_dir, albedo)`` where

        - ``opt`` is the configured ``torch.optim.Adam`` instance.
        - ``light_dir`` is a learnable ``torch.nn.Parameter`` if
          ``cfg.optimize_light`` is True, otherwise the original
          `light_dir_` tensor (not optimized).
        - ``albedo`` is a ``torch.nn.Parameter`` initialized to 1.0 (always
          created, but only included in `opt` when ``cfg.optimize_albedo``
          is True).
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
