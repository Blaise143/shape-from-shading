import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from enum import StrEnum
import numpy as np
import matplotlib.pyplot as plt


@torch.no_grad()
def get_mask_boundary(mask: torch.Tensor) -> torch.Tensor:
    m = mask.float().unsqueeze(0).unsqueeze(0)
    ones = torch.ones((1, 1, 3, 3), device=m.device, dtype=m.dtype)
    neigh = F.conv2d(m, ones, padding=1)
    edge = ((m == 1) & (neigh < 9)).squeeze(0).squeeze(0)
    return edge.bool()


def _gaussian_kernel1d(sigma: float, radius: int, device, dtype):
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-(x**2) / (2 * sigma**2))
    k = k / k.sum()
    return k.view(1, 1, -1, 1), k.view(1, 1, 1, -1)


@torch.no_grad()
def outward_boundary_normals(mask: torch.Tensor, smooth=True, sigma=1.0):
    assert mask.ndim == 2
    device = mask.device
    dtype = torch.float32

    edge = get_mask_boundary(mask)
    edge_f = edge.float()

    m = mask.float().unsqueeze(0).unsqueeze(0)
    if smooth:
        radius = max(1, int(3 * sigma))
        kv, kh = _gaussian_kernel1d(sigma, radius, device, dtype)
        m = F.conv2d(F.conv2d(m, kv, padding=(radius, 0)), kh, padding=(0, radius))

    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(-1, -2)
    gx = F.conv2d(m, sobel_x, padding=1)[0, 0]
    gy = F.conv2d(m, sobel_y, padding=1)[0, 0]

    g = torch.stack([gx, gy], dim=-1)
    g = g / (g.norm(dim=-1, keepdim=True) + 1e-12)
    n2d_out = -g

    n3 = torch.zeros((*mask.shape, 3), device=device, dtype=dtype)
    n3[..., 0] = n2d_out[..., 0] * edge_f
    n3[..., 1] = n2d_out[..., 1] * edge_f
    n3[..., 2] = 0.0

    return edge, n3


# def get_boundary_normals(mask_path: str = "images/sphere_mask.png"):
# boundary_mask = get_boolean_mask(mask_path)
# edge, boundary_normal = outward_boundary_normals(
# boundary_mask, smooth=True, sigma=2
# )
# return edge, boundary_mask.float(), boundary_normal


def pad(tensor, target_shape=(1024, 1024), pad_value=0):
    current_shape = tensor.shape
    if len(current_shape) != 2:
        raise ValueError("Only 2D tensors are supported")

    pad_height = target_shape[0] - current_shape[0]
    pad_width = target_shape[1] - current_shape[1]
    if pad_height < 0 or pad_width < 0:
        raise ValueError("Target shape must be greater than or equal to tensor shape")

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Padding format: (left, right, top, bottom)
    padding = (pad_left, pad_right, pad_top, pad_bottom)

    return F.pad(tensor, padding, mode="constant", value=pad_value)


def compute_smoothness_map(n: torch.Tensor) -> torch.Tensor:
    """
    n: (H, W, 3), assumed L2-normalized per-pixel
    returns: (H, W) smoothness magnitude
    """
    H, W, _ = n.shape
    # forward differences with zero-padding on last row/col
    dx = torch.zeros_like(n)
    dy = torch.zeros_like(n)
    dx[:-1, :, :] = n[1:, :, :] - n[:-1, :, :]
    dy[:, :-1, :] = n[:, 1:, :] - n[:, :-1, :]

    # per-pixel isotropic TV-like magnitude across channels
    mag = torch.sqrt((dx**2 + dy**2).sum(dim=-1) + 1e-12)
    return mag  # (H, W)


#
# def show_smoothness(
#     n: torch.Tensor, mask: torch.Tensor = None, title="Smoothness magnitude"
# ):
#     """
#     n: (H, W, 3)
#     mask: optional (H, W) boolean tensor to restrict display inside object
#     """
#     sm = compute_smoothness_map(n)
#
#     if mask is not None:
#         sm = sm.masked_fill(~mask.bool(), float("nan"))
#
#     arr = sm.detach().cpu().numpy()
#     plt.imshow(arr, cmap="magma")
#     plt.colorbar(fraction=0.046, pad=0.04)
#     plt.title(title)
#     plt.axis("off")
#     plt.show()


@torch.no_grad()
def show_boundary_angle_map(
    n: torch.Tensor,
    n_bdry: torch.Tensor,
    edge_mask: torch.Tensor,
    title: str = "Boundary angle (deg)",
    orientation_invariant: bool = True,
):
    """
    n:       [H,W,3] predicted normals (unit)
    n_bdry:  [H,W,3] target boundary normals (unit)
    edge_mask: [H,W] bool or {0,1}
    """
    # ensure CPU + unit
    n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    n_bdry = n_bdry / n_bdry.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    dot = (n * n_bdry).sum(dim=-1).clamp(-1, 1)
    if orientation_invariant:
        dot = dot.abs()

    ang = torch.acos(dot) * (180.0 / np.pi)  # [H,W] in degrees
    vis = ang.clone()
    E = edge_mask.bool()
    vis[~E] = float("nan")  # hide non-boundary

    arr = vis.detach().cpu().numpy()
    plt.figure()
    plt.imshow(arr, cmap="magma")
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.set_label("degrees")
    plt.title(title)
    plt.axis("off")
    plt.show()

    # Optional: histogram over boundary pixels
    # vals = ang[E].detach().cpu().numpy().ravel()
    # plt.figure()
    # plt.hist(vals, bins=50)
    # plt.xlabel("angle (deg)")
    # plt.ylabel("count")
    # plt.title("Boundary angle histogram")
    # plt.show()

    # return ang  # return tensor in case you want to log stats


import numpy as np
import cv2


def _robustify_pq(p, q, grad_clip_pct=99.0, pq_smooth_sigma=1.0):
    for arr in (p, q):
        lo, hi = np.percentile(arr, [100 - grad_clip_pct, grad_clip_pct])
        np.clip(arr, lo, hi, out=arr)
    k = max(1, int(2 * round(pq_smooth_sigma) + 1))
    p = cv2.GaussianBlur(p, (k, k), pq_smooth_sigma)
    q = cv2.GaussianBlur(q, (k, k), pq_smooth_sigma)
    return p, q


def _poisson_integration(p, q):
    H, W = p.shape
    h = 1  # / H
    lap = np.zeros((H, W), np.float32)
    lap[1:-1, 1:-1] = (
        (p[1:-1, 2:] - p[1:-1, :-2]) / h + (q[2:, 1:-1] - q[:-2, 1:-1]) / h
    ) * 0.5
    F = cv2.dct(lap.astype(np.float32))
    i, j = np.meshgrid(np.arange(W), np.arange(H))
    denom = 2 * (np.cos(np.pi * i / W) + np.cos(np.pi * j / H) - 2)
    denom[0, 0] = 1.0
    Z = F / denom
    Z[0, 0] = 0.0
    return cv2.idct(Z)


def integrate_normals(normals, grad_clip_pct=99.0, pq_smooth_sigma=1.0):
    print("integrating")
    p = normals[:, :, 0] / (normals[:, :, 2] + 1e-3)
    q = normals[:, :, 1] / (normals[:, :, 2] + 1e-3)
    p, q = _robustify_pq(p, q, grad_clip_pct, pq_smooth_sigma)
    z = _poisson_integration(p, q)
    z = z - z.min()
    # z = z / z.max()
    z /= z.max() - z.min() + 1e-8
    z = z * 0.8
    # z = z - np.nanmin(z)
    return z  # _poisson_integration(p, q)


if __name__ == "__main__":
    ...
    # b_mask = get_boolean_mask("images/mask_latest_again.png")
    # edge, n_bdry = outward_boundary_normals(b_mask, smooth=True, sigma=2.0)
    #
    # # ...
    # # from test_4 import get_image
    #
    # t = get_image("images/image.png")  # torch.randn(818, 608)
    # print(t.shape)
    # plt.imshow(t, cmap="gray")
    # plt.show()
    # t_padded = pad(t)
    # # print(t_padded.shape)
    # plt.imshow(t_padded, cmap="gray")  # ; plt.show()
    # # plt.savefig("images/padded_face.png")
    # # plt.imsave("images/padded_face.png", t_padded)
    # # plt.show()
    # # exit()
    # t_mask = get_image("images/image_mask.png")
    #
    # plt.imshow(t_mask, cmap="gray"); plt.show()
    # t_mask_padded = pad(t_mask)
    # plt.imshow(t_mask_padded, cmap="gray"); plt.imsave("images/padded_face_mask.png", t_mask_padded)#plt.savefig("images/padded_face_mask.png")
    # plt.show()
    # # exit()
    # edge, b_mask, n_bdry = get_boundary_normals()#mask_path="images/padded_face_mask.png")
    # visualize_normals_boundary(n_bdry.cpu(), mask=edge.cpu())
    # plot_boundary_normals_arrows(
    # edge, n_bdry, mask=b_mask, stride=20, len_px=30.0, figsize=(6, 6))
    # # print(f"boundary shape: {n_bdry.shape}")
    # breakpoint()
