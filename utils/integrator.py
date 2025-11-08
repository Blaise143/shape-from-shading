import cv2
import numpy as np


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
    w = 1  # / W
    lap = np.zeros((H, W), np.float32)
    lap[1:-1, 1:-1] = (
        (p[1:-1, 2:] - p[1:-1, :-2]) * h + (q[2:, 1:-1] - q[:-2, 1:-1]) * w
    ) * 0.5
    F = cv2.dct(lap.astype(np.float32))
    i, j = np.meshgrid(np.arange(W), np.arange(H))
    denom = 2 * (np.cos(np.pi * i / W) + np.cos(np.pi * j / H) - 2)
    denom[0, 0] = 1.0
    Z = F / denom
    Z[0, 0] = 0.0
    return cv2.idct(Z)


def integrate_normals(
    normals,
    grad_clip_pct=99.0,
    pq_smooth_sigma=1.0,
    x_range=(-1.0, 1.0),
    y_range=(-1.0, 1.0),
):
    """
    normals: (H, W, 3) unit normals.
    """
    print("integrating")
    H, W, _ = normals.shape

    nz = np.clip(normals[:, :, 2], 1e-6, None)

    p_world = -normals[:, :, 0] / nz
    q_world = -normals[:, :, 1] / nz

    x_min, x_max = x_range
    y_min, y_max = y_range
    dx = (x_max - x_min) / (W - 1)  # world units per pixel in x
    dy = (y_max - y_min) / (H - 1)  # world units per pixel in y
    p = p_world * dx
    q = q_world * dy

    p, q = _robustify_pq(p, q, grad_clip_pct, pq_smooth_sigma)

    z = _poisson_integration(p, q)

    return z
