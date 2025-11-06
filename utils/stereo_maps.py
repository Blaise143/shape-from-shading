import torch


def normals_to_fg(normals: torch.Tensor):
    normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-8)
    nx, ny, nz = normals[..., 0], normals[..., 1], normals[..., 2]
    nz = torch.clamp(nz, min=1e-8)
    p = nx / nz
    q = ny / nz
    denom = 1 + torch.sqrt(1 + p**2 + q**2)
    f = 2 * p / denom
    g = 2 * q / denom
    return torch.stack((f, g), dim=-1)


def fg_to_normals(fg: torch.Tensor):
    f, g = fg[..., 0], fg[..., 1]
    r2 = f**2 + g**2
    denom = 1 - 0.25 * r2
    p = f / denom
    q = g / denom
    n = torch.stack((p, q, torch.ones_like(p)), dim=-1)
    n = n / n.norm(dim=-1, keepdim=True)
    return n


if __name__ == "__main__":
    normals = torch.randn((5, 5, 3))
    normals[..., 2] = normals[..., 2].abs()  # torch.tensor([[0.99, 0.2, 0.97]])
    normals = normals / normals.norm(dim=-1, keepdim=True)
    fg = normals_to_fg(normals)
    n_rec = fg_to_normals(fg)
    breakpoint()
    print(normals / normals.norm(dim=-1, keepdim=True))
    print(n_rec)
