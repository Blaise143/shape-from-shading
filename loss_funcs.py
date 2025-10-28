import torch

def photometric_loss(R: torch.Tensor,
                     I: torch.Tensor,
                     weight: torch.Tensor) -> torch.Tensor:
    # weight -> valid mask
    wsum = weight.sum().clamp_min(1.0)
    return (((R - I).pow(2) * weight) / wsum).sum()

def edge_aware_smooth(n, I, k=2.0, eps=1e-6):
    # I in [H,W], n in [H,W,3]
    gx, gy = torch.gradient(I, dim=(0,1))
    gmag = torch.sqrt(gx*gx + gy*gy + eps)          # edges/high-frequency get large gmag
    w = torch.exp(-k * gmag)                         # w≈1 in flat regions (more smoothing), w≈0 near edges

    loss = 0.0
    for c in range(3):
        nx, ny = torch.gradient(n[..., c], dim=(0,1))
        loss = loss + (w * (nx*nx + ny*ny)).mean()
    return loss

def smoothness_loss(n: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    ny, nx = torch.gradient(n, dim=(0,1))
    sm = ny.pow(2).sum(dim=-1) + nx.pow(2).sum(dim=-1)
    wsum = weight.sum().clamp_min(1.0)
    # breakpoint()
    return ((sm * weight) / wsum).sum()

def view_perp_loss(n: torch.Tensor, v: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
    nv = (n * v).sum(dim=-1)
    wsum = edge_mask.sum().clamp_min(1.0)
    return ((nv * nv) * edge_mask).sum() / wsum

def boundary_align_loss(n: torch.Tensor,
                        n_bdry: torch.Tensor,
                        edge_mask: torch.Tensor) -> torch.Tensor:
    cos = (n * n_bdry).sum(dim=-1).abs()
    wsum = edge_mask.sum().clamp_min(1.0)
    return ((1.0 - cos) * edge_mask).sum() / wsum

def hemi_loss(n: torch.Tensor, v: torch.Tensor, weight: torch.Tensor, margin: float = 0.0):
    cos = (n * v).sum(dim=-1)
    hinge = (margin - cos).clamp_min(0)
    wsum = weight.sum().clamp_min(1.0)
    return (hinge * weight).sum() / wsum


def integrability_loss_curl(n: torch.Tensor,
                            weight: torch.Tensor,
                            clamp_nz: float = 1e-4) -> torch.Tensor:
    """
    n:      [H,W,3] (not necessarily normalized)
    weight: [H,W]   mask/weights (>0 inside)
    """
    # normalize to unit normals (safer for stability)
    n = n / (n.norm(dim=-1, keepdim=True).clamp_min(1e-8))
    nx, ny, nz = n[...,0], n[...,1], n[...,2]
    nz = torch.where(nz.abs() < clamp_nz, nz.sign() * clamp_nz, nz)

    # p = -dz/dx, q = -dz/dy from normals
    p = -nx / nz
    q = -ny / nz

    # cross-derivative consistency: py == qx
    py, _  = torch.gradient(p, dim=(0,1))
    _,  qx = torch.gradient(q, dim=(0,1))
    curl   = py - qx

    w = (weight > 0).to(p.dtype)
    wsum = w.sum().clamp_min(1.0)
    return ((curl**2) * w).sum() / wsum
