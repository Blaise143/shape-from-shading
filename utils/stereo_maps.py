import torch


def fg_to_normals(fg: torch.Tensor):
    f, g = fg[..., 0], fg[..., 1]
    r2 = f**2 + g**2

    denom = 4 + r2
    nx = 4 * f / denom
    ny = 4 * g / denom
    nz = (4 - r2) / denom

    return torch.stack((nx, ny, nz), dim=-1)


def normals_to_fg(normals: torch.Tensor):
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
    breakpoint()
    print(normals / normals.norm(dim=-1, keepdim=True))
    print(n_rec)
