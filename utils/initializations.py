import torch
from typing import Optional, Literal
import cv2
import numpy as np
from typing import Tuple
from utils.utils import get_device


def compute_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Use Scharr filters to get image gradients.

    Args:
        image: Grayscale image array of shape ``(H, W)``.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ``(grad_x, grad_y)``, each of shape
        ``(H, W)``, the Scharr-filtered gradients in x and y.
    """
    im = image.astype(np.float32)  # / 255.0
    img = cv2.copyMakeBorder(im, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    grad_x = cv2.Scharr(img, cv2.CV_64F, 1, 0)[1:-1, 1:-1]
    grad_y = cv2.Scharr(img, cv2.CV_64F, 0, 1)[1:-1, 1:-1]
    return grad_x, grad_y


def _robustify_pq(p: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Clip outliers and Gaussian-blur a pair of gradient fields.

    Clips each of `p`, `q` to their 1st/99th percentile (in-place) to limit
    outlier influence, then applies a small Gaussian blur to smooth them.

    Args:
        p: Gradient field in x (or depth-gradient ``p``), shape ``(H, W)``.
            Modified in-place by the percentile clip.
        q: Gradient field in y (or depth-gradient ``q``), shape ``(H, W)``.
            Modified in-place by the percentile clip.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The blurred ``(p, q)``.
    """
    grad_clip_pct = 99
    pq_smooth_sigma = 1.0
    for arr in (p, q):
        lo, hi = np.percentile(arr, [100 - grad_clip_pct, grad_clip_pct])
        np.clip(arr, lo, hi, out=arr)
    k = max(1, int(2 * round(pq_smooth_sigma) + 1))
    p = cv2.GaussianBlur(p, (k, k), pq_smooth_sigma)
    q = cv2.GaussianBlur(q, (k, k), pq_smooth_sigma)
    return p, q


def compute_normals_from_gradients(
    grad_x: np.ndarray, grad_y: np.ndarray
) -> np.ndarray:
    """Convert surface gradients (p, q) to unit surface normals.

    Uses the standard shape-from-shading convention
    ``n = normalize([-p, -q, 1])``.

    Args:
        grad_x: Gradient in x (``p``), shape ``(H, W)``.
        grad_y: Gradient in y (``q``), shape ``(H, W)``.

    Returns:
        np.ndarray: Unit normal map of shape ``(H, W, 3)``.
    """
    p, q = grad_x, grad_y
    nz = np.ones_like(p)
    normals = np.stack([-p, -q, nz], axis=-1)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)

    return normals / np.maximum(norms, 1e-8)


def get_init(
    shape: Tuple[int, int],
    image_np: Optional[np.ndarray] = None,
    kind: Literal["init_grads", "zeroes", "sphere", "ones"] = "init_grads",
    device: Optional[str] = None,
):
    """Initialize a learnable normal map for the optimization loop.

    Args:
        shape: ``(H, W)`` spatial shape of the normal map to create.
        image_np: Grayscale input image, required when ``kind="init_grads"``
            (used to compute an image-gradient-based initialization).
        kind: Initialization strategy:

            - ``"init_grads"``: Estimate normals from the image's Scharr
              gradients (via :func:`compute_gradients` and
              :func:`compute_normals_from_gradients`); usually the best
              starting point.
            - ``"zeroes"``: Near-flat normals pointing at the viewer
              (``z ~= 0.99``) plus a little noise.
            - ``"ones"``: All-ones tensor (not unit-normalized).
            - ``"sphere"``: A synthetic hemisphere bulging toward the
              viewer.
        device: Device to create the tensor on; defaults to
            :func:`utils.utils.get_device`.

    Returns:
        torch.Tensor: A ``(H, W, 3)`` tensor with ``requires_grad=True``,
        ready to be wrapped in (or used directly as) an optimizer parameter.

    Raises:
        ValueError: If `kind` is not one of the supported strategies.
    """
    if kind not in ("ones", "zeroes", "sphere", "init_grads"):
        raise ValueError(f"Unknown init kind: {kind!r}")
    H, W = shape
    normals_shape = (H, W, 3)
    if device is None:
        device = get_device()
    if kind == "ones":
        return torch.ones(normals_shape, device=device, requires_grad=True)
    elif kind == "zeroes":
        zeroes = torch.zeros(normals_shape, device=device)  # , requires_grad=True)
        zeroes[..., 2] = 0.99
        zeroes = zeroes + torch.randn_like(zeroes) * 0.01
        zeroes.requires_grad = True
        return zeroes
    elif kind == "sphere":
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )

        radius = 0.8
        z = torch.sqrt(torch.clamp(radius**2 - xs**2 - ys**2, min=0.0))

        normals = torch.stack([xs, ys, z], dim=-1)
        normals = 0.5 * normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-8)

        normals.requires_grad = True
        return normals
    else:
        assert image_np is not None, "image_np must be valid for kind=='init_grads'"
        grad_x, grad_y = compute_gradients(image_np)
        grad_x, grad_y = _robustify_pq(grad_x, grad_y)
        normals = compute_normals_from_gradients(grad_x, grad_y)  # (p, q)
        normals = normals.astype(np.float32)
        normals_torch = torch.from_numpy(normals).to(device)
        normals_torch.requires_grad = True
        return normals_torch


# if __name__ == "__main__":
#     spherical = initialize_normals()
#     ones = initialize_normals(Initializers.ONES)
#     zeros = initialize_normals(Initializers.ZEROS)
#     print(spherical)
#     print(ones)
#     print(zeros)
#     print(spherical.min(), spherical.max())
