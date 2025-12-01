import numpy as np
from PIL import Image
from typing import Optional, Tuple
import torch
import os
from datetime import datetime


def get_img_and_mask(
    img_path: str, kind: str, mask_path: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given an image and mask path, returns the image in grayscale and the mask as numpy arrays
    """
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


def make_output_folder(base_dir="out", cue="Cue1", obj="sphere") -> str:
    """Create a structured output folder like out/Cue1/sphere/<timestamp>/"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = os.path.join(base_dir, cue, obj, timestamp)
    os.makedirs(folder, exist_ok=True)
    return folder
