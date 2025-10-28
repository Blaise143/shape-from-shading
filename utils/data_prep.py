import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from .plotting import *
import json
from PIL import Image
import numpy as np
# wandb.login()
# wandb.init("something")


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_to_json(path, data):
    with open(path, "r") as f:
        json.dump(data, f)


def load_img(path, normals=False):
    if not normals:
        img = Image.open(path).convert("L")
    else:
        img = Image.open(path).convert("RGB")
    img = np.array(img)
    max_val = np.nanmax(img)
    return img / max_val if max_val > 0 else img  # img.max()


def normalize(v: torch.Tensor, eps=1e-8):
    assert len(v.shape) in [1, 3]
    if len(v.shape) == 1:
        if torch.norm(v) == 0:
            return v / (torch.norm(v) + eps)
        return v / torch.norm(v)
    else:
        return v / (torch.norm(v, dim=2, keepdim=True) + eps)


def get_boundary(img=None, device=None, mask=None):
    if device is None:
        device = get_device()

    if img is not None:
        H, W = img.shape
    if mask is None:
        boundary_mask = torch.zeros_like(img, dtype=torch.bool, device=device)
        boundary_mask[0, :] = True
        boundary_mask[-1, :] = True
        boundary_mask[:, 0] = True
        boundary_mask[:, -1] = True
        return boundary_mask
    m = torch.as_tensor(mask, device=device).bool().float()  # (H,W)→float
    # (1,1,H,W)
    m = m.unsqueeze(0).unsqueeze(0)

    neigh_has_1 = F.max_pool2d(m, kernel_size=3, stride=1, padding=1) > 0

    neigh_has_0 = F.max_pool2d(1.0 - m, kernel_size=3, stride=1, padding=1) > 0

    edge = neigh_has_0 & neigh_has_1

    return edge.squeeze(0).squeeze(0)


def initialize_sphere(I):
    H, W = I.shape
    Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    cx, cy = W // 2, H // 2
    r = min(H, W) // 2 - 1
    Z = torch.zeros((H, W))
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r**2
    Z[mask] = torch.sqrt(r**2 - (X[mask] - cx) ** 2 - (Y[mask] - cy) ** 2)
    Z = Z / Z.max()  # * 0.01
    return Z


if __name__ == "__main__":
    title = "da_sphere"
    Z = initialize_sphere(torch.randn(500, 500))

    plot_surface(Z, show=True, name=title)
    # breakpoint()
    # exit()
    plt.figure(figsize=(6, 5))
    plt.imshow(Z.cpu().numpy(), cmap="viridis")
    plt.title(title)
    plt.colorbar()
    plt.show()

    # 3D surface plot
    H, W = Z.shape
    X, Y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing="xy")
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X.numpy(), Y.numpy(), Z.cpu().numpy(), cmap="viridis")
    ax.set_title(title + " (3D)")
    plt.show()
