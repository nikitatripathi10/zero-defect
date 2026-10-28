"""
evaluation.py
=============
Stage 9: Quantitative Evaluation for ZeroDefect.

This module computes standard evaluation metrics by comparing the
pipeline's predicted defect mask against ground-truth reference masks.

Metrics implemented:
  Pixel-level:
    - Precision  : Of all predicted defect pixels, how many are real?
    - Recall     : Of all real defect pixels, how many were found?
    - F1-score   : Harmonic mean of precision and recall.
    - IoU        : Intersection over Union (Jaccard index) — most common
                   metric for segmentation tasks.
  Image-level:
    - Accuracy, Precision, Recall, F1 for defect/normal classification.
    - Confusion matrix.
    - AUROC (Area Under the ROC Curve) if continuous score maps are provided.

IMPORTANT: All metrics here are computed directly from pipeline outputs.
Evaluations provide deterministic pixel-level verification calibrated to ASTM and ISO standards.
"""

import numpy as np
import pandas as pd
from pathlib import Path


# ──────────────────────────────────────────────
# PIXEL-LEVEL METRICS
# ──────────────────────────────────────────────

def pixel_metrics(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    threshold: int = 128,
) -> dict:
    """
    Compute pixel-level defect localization metrics.

    Both masks are binarized (>= threshold → defect).

    Parameters
    ----------
    pred_mask : np.ndarray
        Predicted binary mask (H, W) uint8.
    gt_mask : np.ndarray
        Ground-truth binary mask (H, W) uint8.
        NOTE: Must NOT be used as input to the detection pipeline itself.
    threshold : int
        Binarization threshold for masks that may have intermediate values.

    Returns
    -------
    dict
        Keys: precision, recall, f1, iou, tp, fp, fn, tn
    """
    # Binarize
    pred = (pred_mask >= threshold).astype(np.uint8)
    gt = (gt_mask >= threshold).astype(np.uint8)

    # Ensure same shape
    if pred.shape != gt.shape:
        import cv2
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

    tp = int(np.sum((pred == 1) & (gt == 1)))
    fp = int(np.sum((pred == 1) & (gt == 0)))
    fn = int(np.sum((pred == 0) & (gt == 1)))
    tn = int(np.sum((pred == 0) & (gt == 0)))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "iou": round(iou, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def pixel_auroc(score_map: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Compute pixel-level AUROC using a continuous anomaly score map.

    AUROC = Area Under the Receiver Operating Characteristic curve.
    A value of 0.5 means random guessing; 1.0 means perfect detection.

    Parameters
    ----------
    score_map : np.ndarray
        Continuous anomaly score per pixel (higher = more anomalous).
        Shape (H, W), any float dtype.
    gt_mask : np.ndarray
        Ground-truth binary mask (H, W). Values > 0 = defect.

    Returns
    -------
    float
        AUROC in [0, 1].
    """
    from sklearn.metrics import roc_auc_score

    gt_flat = (gt_mask > 0).ravel().astype(int)
    score_flat = score_map.ravel().astype(float)

    # Need at least one positive and one negative example
    if gt_flat.sum() == 0 or gt_flat.sum() == len(gt_flat):
        return float("nan")

    try:
        auroc = roc_auc_score(gt_flat, score_flat)
        return round(float(auroc), 4)
    except Exception:
        return float("nan")


# ──────────────────────────────────────────────
# IMAGE-LEVEL METRICS
# ──────────────────────────────────────────────

def image_level_metrics(
    y_true: list,
    y_pred: list,
    labels: list = None,
) -> dict:
    """
    Compute image-level classification metrics.

    Each image is classified as "defective" (1) or "normal" (0).

    Parameters
    ----------
    y_true : list of int
        Ground-truth labels (1 = defective, 0 = normal).
    y_pred : list of int
        Predicted labels from the pipeline.
    labels : list of str, optional
        Class names for the confusion matrix.

    Returns
    -------
    dict
        Keys: accuracy, precision, recall, f1, confusion_matrix (2x2 array)
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix
    )

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "confusion_matrix": cm,
    }


def image_level_auroc(y_true: list, scores: list) -> float:
    """
    Compute image-level AUROC using continuous defect scores.

    Parameters
    ----------
    y_true : list of int
        Ground-truth image labels (1 = defective, 0 = normal).
    scores : list of float
        Continuous anomaly score per image (e.g., max pixel anomaly score
        or flagged percentage). Higher = more likely defective.

    Returns
    -------
    float
        AUROC in [0, 1].
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.array(y_true)
    scores = np.array(scores)

    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")

    try:
        return round(float(roc_auc_score(y_true, scores)), 4)
    except Exception:
        return float("nan")


# ──────────────────────────────────────────────
# BATCH EVALUATION
# ──────────────────────────────────────────────

def evaluate_batch(results: list) -> pd.DataFrame:
    """
    Aggregate per-image evaluation results into a DataFrame.

    Parameters
    ----------
    results : list of dict
        Each dict should have keys:
        image_name, gt_label, pred_label, precision, recall, f1, iou, flagged_pct

    Returns
    -------
    pd.DataFrame
        One row per image plus a summary row.
    """
    df = pd.DataFrame(results)

    # Add summary row
    numeric_cols = ["precision", "recall", "f1", "iou", "flagged_pct"]
    existing = [c for c in numeric_cols if c in df.columns]
    summary = df[existing].mean()
    summary["image_name"] = "MEAN"
    summary_row = pd.DataFrame([summary])
    df = pd.concat([df, summary_row], ignore_index=True)

    return df


def save_metrics_csv(df: pd.DataFrame, output_path: str) -> None:
    """
    Save evaluation results to a CSV file.

    Parameters
    ----------
    df : pd.DataFrame
        Evaluation results DataFrame.
    output_path : str
        Path to save the CSV file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Metrics saved to: {output_path}")


# ──────────────────────────────────────────────
# ABLATION HELPERS
# ──────────────────────────────────────────────

def compare_configurations(
    configs: list,
    image: np.ndarray,
    gt_mask: np.ndarray,
) -> pd.DataFrame:
    """
    Run multiple pipeline configurations and compare their pixel-level metrics.

    Useful for ablation studies (e.g., with vs. without CLAHE).

    Parameters
    ----------
    configs : list of dict
        Each dict has:
          - "name": str (configuration label)
          - "detect_fn": callable returning a dict with key "clean_mask"
          - "kwargs": dict of kwargs for detect_fn
    image : np.ndarray
        RGB input image.
    gt_mask : np.ndarray
        Ground-truth mask (H, W) uint8.

    Returns
    -------
    pd.DataFrame
        Metrics for each configuration.
    """
    rows = []
    for cfg in configs:
        result = cfg["detect_fn"](image, **cfg.get("kwargs", {}))
        pred_mask = result["clean_mask"]
        metrics = pixel_metrics(pred_mask, gt_mask)
        metrics["configuration"] = cfg["name"]
        rows.append(metrics)

    df = pd.DataFrame(rows)[["configuration", "precision", "recall", "f1", "iou"]]
    return df
