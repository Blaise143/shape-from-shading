import torch


def fg_to_normals(fg: torch.Tensor) -> torch.Tensor:
    """Convert stereographic parameters ``(f, g)`` back to unit normals.

    Inverts :func:`normals_to_fg` via the standard stereographic projection
    formula, so normals can be optimized in an unconstrained 2D space (`f`,
    `g`) while always mapping back to valid unit vectors.

    Args:
        fg: Tensor of shape ``(..., 2)`` holding stereographic parameters
            ``(f, g)`` in its last dimension.

    Returns:
        torch.Tensor: Unit normal map of shape ``(..., 3)``.
    """
    f, g = fg[..., 0], fg[..., 1]
    r2 = f**2 + g**2

    denom = 4 + r2
    nx = 4 * f / denom
    ny = 4 * g / denom
    nz = (4 - r2) / denom

    return torch.stack((nx, ny, nz), dim=-1)


def normals_to_fg(normals: torch.Tensor) -> torch.Tensor:
    """Convert unit normals to stereographic parameters ``(f, g)``.

    Normals are first renormalized to unit length, then projected via
    ``f = 2*nx / (1 + nz)``, ``g = 2*ny / (1 + nz)``. This is the inverse of
    :func:`fg_to_normals`.

    Args:
        normals: Normal map of shape ``(..., 3)``.

    Returns:
        torch.Tensor: Stereographic parameters of shape ``(..., 2)``.
    """
    normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
    nx, ny, nz = normals[..., 0], normals[..., 1], normals[..., 2]

    denom = 1 + nz + 1e-8

    f = 2 * nx / denom
    g = 2 * ny / denom

    return torch.stack((f, g), dim=-1)


if __name__ == "__main__":
    normals = torch.randn((5, 5, 3))
    normals[..., 2] = normals[..., 2].abs()  # torch.tensor([[0.99, 0.2, 0.97]])
    normals = normals / normals.norm(dim=-1, keepdim=True)
    fg = normals_to_fg(normals)
    n_rec = fg_to_normals(fg)
    print(normals / normals.norm(dim=-1, keepdim=True))
    print(n_rec)
