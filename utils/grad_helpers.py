import torch
import cv2
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import textwrap
import re
import json


def save_params(params: dict, path: str):
    """
    Saves the parameters to json file
    """
    with open(path, "w") as f:
        json.dump(params, f, indent=4, ensure_ascii=False)


def log_details(details: str, md_file: str):
    """
    Takes a triple-quoted experiment details string and writes it to a Markdown file.
    """
    cleaned = textwrap.dedent(details).strip()
    formatted = re.sub(r"\s*-\s*", r"\n- ", cleaned)
    Path(md_file).write_text(formatted.strip() + "\n", encoding="utf-8")


def save_init(img: np.ndarray, save_path: str):
    plt.imshow(img)
    plt.axis("off")
    plt.title("input")
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_normals(n, save_path=None):
    normals = np.asarray(n, dtype=np.float32)
    assert normals.shape[-1] == 3, "Expected normals with 3 channels (x, y, z)"

    rgb = (normals * 0.5 + 0.5).clip(0, 1)

    fig, ax = plt.subplots()
    ax.imshow(rgb)
    ax.axis("off")
    ax.set_title("Normal Map")
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
    else:
        plt.show()

    # Individual components
    labels = ["X", "Y", "Z"]
    for i, label in enumerate(labels):
        comp = (normals[..., i] * 0.5 + 0.5).clip(0, 1)
        fig, ax = plt.subplots()
        ax.imshow(comp, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"{label}-component")

        if save_path:
            path = save_path.replace(".png", f"_{label}.png")
            fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
            plt.close(fig)
        else:
            plt.show()


# def plot_normals(n, save_path=None):
#     normals = np.asarray(n, dtype=np.float32)
#     rgb = (normals * 0.5 + 0.5).clip(0, 1)
#
#     plt.imshow(rgb)
#     plt.axis("off")
#     plt.title("Normal Map")
#     if save_path:
#         plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
#         plt.close()
#     else:
#         plt.show()
#     labels = ["X", "Y", "Z"]
#     for i, label in enumerate(labels):
#         comp = (normals[..., i] * 0.5 + 0.5).clip(0, 1)
#         plt.imshow(comp, cmap="gray", vmin=0, vmax=1)
#         plt.axis("off")
#         plt.title(f"{label}-component")
#
#         if save_path:
#             path = save_path.replace(".png", f"_{label}.png")
#             plt.savefig(path, bbox_inches="tight", pad_inches=0)
#             plt.close()
#         else:
#             plt.show()
#


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


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
        grad_x, grad_y = compute_gradients(image_np)
        grad_x, grad_y = _robustify_pq(grad_x, grad_y)
        normals = compute_normals_from_gradients(grad_x, grad_y)  # (p, q)
        normals = normals.astype(np.float32)
        normals_torch = torch.from_numpy(normals).to(device)
        normals_torch.requires_grad = True
        return normals_torch


if __name__ == "__main__":
    path = "../images/sphere.png"
    from PIL import Image

    img = Image.open(path).convert("L")
    img_np = np.array(img)
    # plt.imshow(img_np)
    # plt.show()
    shape = img_np.shape
    normals = get_init(shape=shape, image_np=img_np, kind="init_grads")
    breakpoint()
