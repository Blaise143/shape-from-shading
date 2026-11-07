import os
import json
import numpy as np
from imageio import imread


def compute_metrics(prediction: np.ndarray, ground_truth: np.ndarray, mask: np.ndarray):
    """Compute MSE and MAE between a prediction and ground truth, within a mask.

    Args:
        prediction: Predicted array of shape ``(H, W, C)``.
        ground_truth: Ground-truth array of shape ``(H, W, C)``.
        mask: Boolean or 0/1 array of shape ``(H, W)``; ``True``/nonzero marks
            valid pixels to include in the metrics.

    Returns:
        dict: ``{"mse": float, "mae": float}`` computed over the valid
        (masked) pixels only.
    """

    mask = mask.astype(bool)

    pred_valid = prediction[mask]
    gt_valid = ground_truth[mask]

    mse = float(np.mean((pred_valid - gt_valid) ** 2))
    mae = float(np.mean(np.abs(pred_valid - gt_valid)))

    metrics_dict = {"mse": mse, "mae": mae}

    return metrics_dict


def log(metrics: dict, path: str):
    """Write a metrics dict to ``{path}/metrics.json``, creating `path` if needed.

    Args:
        metrics: JSON-serializable metrics to save.
        path: Directory to save into; created if it doesn't exist.
    """
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)


def evaluate_objects(
    objects: list[str],
    out_path: str,
    gt_dir: str = "GT_data4/Normals",
    mask_dir: str = "GT_data4/Masks",
    direct_pred_dir: str = "out_ananta/Cue1",
    stereo_pred_dir: str = "stereo_outputs/Cue1",
):
    """
    Loop over objects, compute metrics for each prediction type,
    and save a single combined metrics.json using `log()`.

    For each object, loads its ground-truth normals/mask and two sets of
    predicted normals ("direct" and "stereo"), then scores each against
    the ground truth with :func:`compute_metrics`.

    Args:
        objects: Object basenames to evaluate (e.g. ``["sphere", "cube"]``).
        out_path: Directory to write the combined ``metrics.json`` to.
        gt_dir: Directory containing ground-truth normal maps
            (``{gt_dir}/{obj}.png``).
        mask_dir: Directory containing masks (``{mask_dir}/{obj}.png``).
        direct_pred_dir: Directory containing direct-normal-estimation
            predictions (``{direct_pred_dir}/{obj}/normals.npy``).
        stereo_pred_dir: Directory containing stereographic-parametrization
            predictions (``{stereo_pred_dir}/{obj}/normals.npy``).

    Returns:
        dict: Maps each object name to ``{"direct": {...}, "stereo": {...}}``
        metrics dicts (see :func:`compute_metrics`).
    """

    all_results = {}

    for obj in objects:
        print(f"Processing {obj}...")

        gt_path = os.path.join(gt_dir, f"{obj}.png")
        mask_path = os.path.join(mask_dir, f"{obj}.png")
        direct_pred_path = os.path.join(direct_pred_dir, obj, "normals.npy")
        stereo_pred_path = os.path.join(stereo_pred_dir, obj, "normals.npy")

        gt = (imread(gt_path).astype(np.float32) / 255.0) * 2 - 1
        mask = imread(mask_path).astype(bool)
        pred_direct = np.load(direct_pred_path)
        pred_stereo = np.load(stereo_pred_path)
        all_results[obj] = {
            "direct": compute_metrics(pred_direct, gt, mask),
            "stereo": compute_metrics(pred_stereo, gt, mask),
        }

    log(all_results, out_path)

    return all_results


if __name__ == "__main__":
    evaluate_objects(
        objects=["sphere", "pyramid", "cube", "cylinder"],
        out_path="metrics_latest",
    )
