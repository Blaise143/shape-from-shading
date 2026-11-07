import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
from pathlib import Path


def calculate_normals_numpy(depth_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute surface normals from a depth map, restricted to a mask.

    Lightly Gaussian-smooths the depth map, takes its spatial gradients, and
    converts them to unit normals (``n = normalize([-p, -q, 1])``). Pixels
    outside `mask` are zeroed out.

    Args:
        depth_map: Depth map of shape ``(H, W)``.
        mask: Boolean region-of-interest mask of shape ``(H, W)``.

    Returns:
        np.ndarray: Normal map of shape ``(H, W, 3)``, zeroed outside `mask`.
    """
    depth_map = gaussian_filter(depth_map, sigma=0.2)
    dZ_dy, dZ_dx = np.gradient(depth_map.astype(np.float32))

    Nx = -dZ_dx
    Ny = -dZ_dy
    Nz = np.ones_like(depth_map)

    normals = np.stack((-Nx, -Ny, Nz), axis=-1)  # H x W x 3

    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-8)

    normals[~mask] = 0  # np.nan

    return normals


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    gt_data_dir = repo_root / "GT_data4"

    objects: list[str] = ["sphere", "pyramid", "cone", "cylinder", "cube"]
    out_path = gt_data_dir / "Normals"
    os.makedirs(out_path, exist_ok=True)

    for i in objects:
        depth = np.array(Image.open(gt_data_dir / "Depths" / f"{i}.png").convert("L"))
        mask = np.array(Image.open(gt_data_dir / "Masks" / f"{i}.png")) > 0
        normals = calculate_normals_numpy(depth, mask)
        vis_normals = (normals + 1) / 2

        img_data = np.clip((vis_normals * 255.0), 0, 255).astype(np.uint8)
        img = Image.fromarray(img_data, mode="RGB")
        save_path = out_path / f"{i}.png"
        img.save(save_path)
        plt.imshow(vis_normals)  # visualize as RGB
        plt.show()
