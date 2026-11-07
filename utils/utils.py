import numpy as np
from PIL import Image
from typing import Tuple
import torch
import os
from datetime import datetime


def get_img_and_mask(
    img_path: str, mask_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given an image and mask path, returns the image in grayscale and the mask as numpy arrays.

    Args:
        img_path: Path to the input image.
        mask_path: Path to the binary mask image.

    Returns:
        Tuple[np.ndarray, np.ndarray]: ``(img_np, mask)`` where `img_np` is
        the grayscale image and `mask` is a boolean array (``True`` where
        the mask image's pixel value is greater than 0).
    """
    img = Image.open(img_path).convert("L")
    img_np = np.array(img)

    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask) > 0

    return img_np, mask


def get_device():
    """Pick the best available torch device.

    Returns:
        str: ``"cuda"`` if a CUDA GPU is available, else ``"mps"`` if Apple
        Metal is available, else ``"cpu"``.
    """
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def make_output_folder(base_dir="out", cue="Cue1", obj="sphere") -> str:
    """Create a structured output folder like out/Cue1/sphere/<timestamp>/

    Args:
        base_dir: Base output directory.
        cue: Name of the cue/method subfolder.
        obj: Name of the object subfolder.

    Returns:
        str: The created folder path, ``{base_dir}/{cue}/{obj}/{timestamp}``.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = os.path.join(base_dir, cue, obj, timestamp)
    os.makedirs(folder, exist_ok=True)
    return folder
