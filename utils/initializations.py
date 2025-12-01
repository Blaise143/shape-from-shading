import torch

# from enum import StrEnum
# from synthesis import Sphere
import cv2
import numpy as np
from typing import Tuple
from utils.utils import get_device


def compute_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    im = image.astype(np.float32)  # / 255.0
    img = cv2.copyMakeBorder(im, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    grad_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)[1:-1, 1:-1]
    grad_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)[1:-1, 1:-1]
    return grad_x, grad_y


def _robustify_pq(p: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    grad_clip_pct = 99
    pq_smooth_sigma = 1.0
    for arr in (p, q):
        lo, hi = np.percentile(arr, [100 - grad_clip_pct, grad_clip_pct])
        np.clip(arr, lo, hi, out=arr)
    k = max(1, int(2 * round(pq_smooth_sigma) + 1))
    p = cv2.GaussianBlur(p, (k, k), pq_smooth_sigma)
    q = cv2.GaussianBlur(q, (k, k), pq_smooth_sigma)
    print("robustified pq")
    return p, q


def compute_normals_from_gradients(
    grad_x: np.ndarray, grad_y: np.ndarray
) -> np.ndarray:
    """
    n = [-p, -q, 1]
    """
    p, q = grad_x, grad_y
    nz = np.ones_like(p)
    normals = np.stack([-p, -q, nz], axis=-1)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)

    return normals / np.maximum(norms, 1e-8)


# ---
def get_init(shape, image_np=None, kind="ones", device=get_device()):
    """
    TODO: make the kind argument an enum
    """
    assert kind in ["ones", "zeroes", "sphere", "init_grads"]
    H, W = shape  # I.shape
    normals_shape = (H, W, 3)
    if kind == "ones":
        return torch.ones(normals_shape, device=device, requires_grad=True)
    elif kind == "zeroes":
        zeroes = torch.zeros(normals_shape, device=device)  # , requires_grad=True)
        zeroes[..., 2] = 0.99
        zeroes = zeroes + torch.randn_like(zeroes) * 0.01
        zeroes.requires_grad = True
        return zeroes
    # elif kind == "init_grads":
    elif kind == "sphere":
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )

        radius = 0.8  # relative radius
        z = torch.sqrt(torch.clamp(radius**2 - xs**2 - ys**2, min=0.0))

        normals = torch.stack([xs, ys, z], dim=-1)
        normals = 0.5 * normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)

        normals.requires_grad = True
        return normals
    else:
        # pyrefly: ignore [bad-argument-type]
        grad_x, grad_y = compute_gradients(image_np)
        grad_x, grad_y = _robustify_pq(grad_x, grad_y)
        normals = compute_normals_from_gradients(grad_x, grad_y)  # (p, q)
        normals = normals.astype(np.float32)
        normals_torch = torch.from_numpy(normals).to(device)
        normals_torch.requires_grad = True
        return normals_torch


#
# class Initializers(StrEnum):
#     SPHERE = "sphere"
#     ZEROS = "zeros"
#     ONES = "ones"
#
#
# def initialize_normals(kind: Initializers = Initializers.SPHERE, image_size: int=1024, device="mps"):
#     if kind not in Initializers:
#         message = "kind should be one of ['sphere', 'zeros', 'ones']"
#         raise Exception(message)
#
#     if kind == Initializers.SPHERE:
#         return _get_sphere_normals(image_size, device=device)
#     elif kind == Initializers.ONES:
#         return torch.ones(size=(image_size, image_size, 3), device=device)
#     else:
#         zeros = torch.zeros(size=(image_size, image_size, 3), device=device)
#         zeros[..., 2] = 1.0
#         return zeros
#
#
# def _get_sphere_normals(img_size: int = 1024, device: str= "mps"):
#     radius = (img_size // 2) - 10
#     grid_range = torch.linspace(-radius, radius, img_size)
#     y, x = torch.meshgrid(grid_range, grid_range, indexing='ij')
#
#     sphere = Sphere(center=[0.0, 0.0, 0.0], radius=radius,
#                     light_direction=torch.tensor([0.0, 0.0, 1.0]))
#
#     radiance, mask, normals = sphere.run(x, y)
#     normals = normals.to(device)
#     if normals.shape[0]==3:
#         return normals.permute(1,2,0)
#     return normals
#
#
# if __name__ == "__main__":
#     spherical = initialize_normals()
#     ones = initialize_normals(Initializers.ONES)
#     zeros = initialize_normals(Initializers.ZEROS)
#     print(spherical)
#     print(ones)
#     print(zeros)
#     print(spherical.min(), spherical.max())
