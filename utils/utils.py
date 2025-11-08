import numpy as np
from PIL import Image
from typing import Optional
import torch


def get_img_and_mask(img_path: str, kind: str, mask_path: Optional[str] = None):
    assert kind in ["sphere", "cube", "custom_mask", "ones"], "kind is invalid"

    img = Image.open(img_path).convert("L")
    img_np = np.array(img)

    if kind == "cube":
        if mask_path is None:
            raise RuntimeError("Provide a valid mask path for a cube")
        else:
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask) > 0
    elif kind == "custom_mask":
        if mask_path is None:
            raise RuntimeError("mask invalid")
        else:
            mask = Image.open(mask_path).convert("L")
            mask = np.array(mask) > 0
    else:
        mask = img_np > 0

    if kind == "ones":
        mask = np.ones_like(img_np)

    return img_np, mask


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"
