import os
import json
import numpy as np
from imageio import imread


def compute_metrics(prediction: np.ndarray, ground_truth: np.ndarray, mask: np.ndarray):
    """
    prediction: (H,W,C)
    ground_truth: (H,W,C)
    mask: (H,W) boolean or 0/1 — True means valid pixel
    """

    mask = mask.astype(bool)

    pred_valid = prediction[mask]
    gt_valid = ground_truth[mask]

    mse = float(np.mean((pred_valid - gt_valid) ** 2))
    mae = float(np.mean(np.abs(pred_valid - gt_valid)))

    metrics_dict = {"mse": mse, "mae": mae}

    return metrics_dict


def log(metrics: dict, path: str):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)


def evaluate_objects(objects: list[str], out_path: str):
    """
    Loop over objects, compute metrics for each prediction type,
    and save a single combined metrics.json using `log()`.
    """

    all_results = {}

    for obj in objects:
        print(f"Processing {obj}...")

        gt_path = f"GT_data4/Normals/{obj}.png"
        mask_path = f"GT_data4/Masks/{obj}.png"
        direct_pred_path = f"out_ananta/Cue1/{obj}/normals.npy"
        stereo_pred_path = f"stereo_outputs/Cue1/{obj}/normals.npy"

        gt = (imread(gt_path).astype(np.float32) / 255.0) * 2 - 1
        # breakpoint()
        mask = imread(mask_path).astype(bool)
        pred_direct = np.load(direct_pred_path)
        pred_stereo = np.load(stereo_pred_path)
        # print("check out gt, mask")
        # breakpoint()
        all_results[obj] = {
            "direct": compute_metrics(pred_direct, gt, mask),
            "stereo": compute_metrics(pred_stereo, gt, mask),
        }

    log(all_results, out_path)

    return all_results


if __name__ == "__main__":
    objects = ["sphere", "pyramid", "cube", "cylinder"]
    out_path = "metrics_latest"

    evaluate_objects(objects=objects, out_path=out_path)
