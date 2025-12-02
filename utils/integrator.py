import cv2
import numpy as np
from utils.initializations import _robustify_pq
from typing import Tuple


def _poisson_integration(p: np.ndarray, q: np.ndarray):
    """
    Creates a laplacian, throws it into the frequenxy domain
    """
    H, W = p.shape
    lap = np.zeros((H, W), np.float32)
    lap[1:-1, 1:-1] = (
        (p[1:-1, 2:] - p[1:-1, :-2]) + (q[2:, 1:-1] - q[:-2, 1:-1])
    ) * 0.5
    F = cv2.dct(lap.astype(np.float32))
    i, j = np.meshgrid(np.arange(W), np.arange(H))
    denom = 2 * (np.cos(np.pi * i / W) + np.cos(np.pi * j / H) - 2)
    denom[0, 0] = 1.0
    Z = F / denom
    Z[0, 0] = 0.0
    return cv2.idct(Z)


def integrate_normals(
    normals: np.ndarray,
    x_range: Tuple[float, float] = (-1.0, 1.0),
    y_range: Tuple[float, float] = (-1.0, 1.0),
):
    """
    Given normals (H,W,3), returns the depth by integrating over them.
    """
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

    p, q = _robustify_pq(p, q)  # , grad_clip_pct, pq_smooth_sigma)

    z = _poisson_integration(p, q)

    return z
