import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os


def calculate_normals_numpy(depth_map: np.ndarray, mask: np.ndarray) -> np.ndarray:
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
    objects: list[str] = ["sphere", "pyramid", "cone", "cylinder", "cube"]
    out_path = "../GT_data4/Normals/"
    os.makedirs(out_path, exist_ok=True)

    for i in objects:
        depth = np.array(Image.open(f"../GT_data4/Depths/{i}.png").convert("L"))
        mask = (
            np.array(Image.open(f"../GT_data4/Masks/{i}.png")) > 0
        )  # .convert("L")) > 128
        normals = calculate_normals_numpy(depth, mask)
        vis_normals = (normals + 1) / 2

        img_data = np.clip((vis_normals * 255.0), 0, 255).astype(np.uint8)
        img = Image.fromarray(img_data, mode="RGB")
        save_path = os.path.join(out_path, f"{i}.png")
        img.save(save_path)
        plt.imshow((normals + 1) / 2)  # visualize as RGB
        plt.show()
        breakpoint()
