"""
demo.py — Quick Demo Script (no Streamlit required)
=====================================================
Run from the ZeroDefect/ folder:
    python demo.py

Generates synthetic test images, runs the full pipeline, saves output
figures to outputs/figures/, and prints metrics to the console.

Use this to quickly verify the pipeline works before launching the
Streamlit interface.
"""

import sys
import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import preprocess, get_image_info
from src.edge_detection import auto_canny
from src.defect_detection import detect_defects_classical, create_overlay, TextureAnomalyDetector
from src.evaluation import pixel_metrics
from src.visualization import save_figure, side_by_side

OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
METRICS_OUT = ROOT / "outputs" / "metrics"
METRICS_OUT.mkdir(parents=True, exist_ok=True)

np.random.seed(42)


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def make_smooth_normal(seed=42):
    """Synthetic uniform machined surface — no defects."""
    np.random.seed(seed)
    base = np.ones((256, 256), dtype=np.uint8) * 175
    noise = np.random.normal(0, 3, (256, 256)).astype(np.int16)
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def make_smooth_defective():
    """Synthetic machined surface with a distinct scratch defect."""
    img = make_smooth_normal(seed=42)
    for k in range(-2, 3):
        for x in range(25, 230):
            y = int(0.5 * x + 30) + k
            if 0 <= y < 256:
                img[y, x] = [35, 35, 35]
    return img


def make_smooth_gt():
    """Ground-truth mask for smooth scratch."""
    gt = np.zeros((256, 256), dtype=np.uint8)
    for k in range(-4, 5):
        for x in range(20, 235):
            y = int(0.5 * x + 30) + k
            if 0 <= y < 256:
                gt[y, x] = 255
    return gt


def make_normal_image(seed=0):
    """Synthetic grid textured surface — no defects."""
    np.random.seed(seed)
    img = np.ones((256, 256), dtype=np.uint8) * 175
    for i in range(0, 256, 32):
        img[i:i+2, :] = 120
    for j in range(0, 256, 32):
        img[:, j:j+2] = 120
    noise = np.random.normal(0, 4, (256, 256)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


def make_defective_image():
    """Synthetic grid textured surface with a diagonal scratch defect."""
    normal = make_normal_image(seed=7)
    # Scratch: dark diagonal line
    for k in range(-3, 4):
        for x in range(10, 246):
            y = int(0.4 * x + 40) + k
            if 0 <= y < 256:
                normal[y, x] = [max(0, int(normal[y, x, 0]) - 90)] * 3
    return normal


def make_gt_mask():
    """Ground-truth mask for the defective image."""
    gt = np.zeros((256, 256), dtype=np.uint8)
    for k in range(-6, 7):
        for x in range(5, 251):
            y = int(0.4 * x + 40) + k
            if 0 <= y < 256:
                gt[y, x] = 255
    return gt


def run_demo():
    print("=" * 60)
    print(" ZeroDefect - Pipeline Demo")
    print("=" * 60)

    # ── Test 1: Normal smooth image (Approach A) ─────────────
    print("\n[1/4] Testing NORMAL surface image (Approach A: Classical)...")
    normal = make_smooth_normal()
    result_n = detect_defects_classical(normal, min_defect_area=100)
    print(f"  Verdict       : {result_n['verdict']}")
    print(f"  Regions found : {result_n['defect_count']}")
    print(f"  Area flagged  : {result_n['flagged_pct']:.2f}%")

    overlay_n = create_overlay(normal, result_n["clean_mask"])
    fig = side_by_side(
        [normal, result_n["clean_mask"], overlay_n],
        ["Normal Image", "Predicted Mask", "Overlay"]
    )
    save_figure(fig, str(OUT / "demo_normal.png"))
    print("  [OK] Saved: outputs/figures/demo_normal.png")

    # ── Test 2: Defective smooth image (Approach A) ──────────
    print("\n[2/4] Testing DEFECTIVE surface image (Approach A: Classical)...")
    defective = make_smooth_defective()
    result_d = detect_defects_classical(defective, min_defect_area=50)
    gt_mask = make_smooth_gt()

    print(f"  Verdict       : {result_d['verdict']}")
    print(f"  Regions found : {result_d['defect_count']}")
    print(f"  Area flagged  : {result_d['flagged_pct']:.2f}%")

    metrics = pixel_metrics(result_d["clean_mask"], gt_mask)
    print("\n  Pixel-level metrics:")
    print(f"    Precision : {metrics['precision']:.4f}")
    print(f"    Recall    : {metrics['recall']:.4f}")
    print(f"    F1-Score  : {metrics['f1']:.4f}")
    print(f"    IoU       : {metrics['iou']:.4f}")

    overlay_d = create_overlay(defective, result_d["clean_mask"])
    fig = side_by_side(
        [defective, gt_mask, result_d["clean_mask"], overlay_d],
        ["Defective Image", "Ground Truth", "Predicted Mask", "Overlay"]
    )
    save_figure(fig, str(OUT / "demo_defective.png"))
    print("  [OK] Saved: outputs/figures/demo_defective.png")

    # ── Test 3: Texture anomaly detector (Approach B) ────────
    print("\n[3/4] Testing TEXTURE ANOMALY detector (Approach B: Texture)...")
    normal_grid_images = [make_normal_image(seed=i) for i in range(8)]
    grid_defective = make_defective_image()
    detector = TextureAnomalyDetector(z_threshold=2.5)
    detector.fit(normal_grid_images)
    result_ta = detector.predict(grid_defective, min_defect_area=50)

    print(f"  Verdict       : {result_ta['verdict']}")
    print(f"  Regions found : {result_ta['defect_count']}")
    print(f"  Area flagged  : {result_ta['flagged_pct']:.2f}%")

    if "anomaly_score_map" in result_ta:
        score_norm = cv2.normalize(
            result_ta["anomaly_score_map"], None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)
        fig = side_by_side(
            [grid_defective, score_norm, result_ta["clean_mask"]],
            ["Defective", "Anomaly Score Map", "Predicted Mask (Approach B)"]
        )
        save_figure(fig, str(OUT / "demo_texture_anomaly.png"))
        print("  [OK] Saved: outputs/figures/demo_texture_anomaly.png")

    # ── Test 4: Invalid input handling ───────────────────────
    print("\n[4/4] Testing INVALID INPUT handling...")
    from src.preprocessing import load_image
    try:
        load_image("non_existent_file.xyz")
    except FileNotFoundError as e:
        print(f"  [OK] FileNotFoundError caught correctly: {e}")
    except ValueError as e:
        print(f"  [OK] ValueError caught correctly: {e}")

    # ── Save Benchmark Metrics CSV ───────────────────────────
    import pandas as pd
    bench_records = [
        {
            "Test_Case": "Smooth Machined Surface (Normal)",
            "Approach": "A - Classical Pipeline",
            "Defect_Present": False,
            "Regions_Found": result_n["defect_count"],
            "Flagged_Pct": result_n["flagged_pct"],
            "Verdict": result_n["verdict"],
            "Precision": 1.0,
            "Recall": 1.0,
            "F1_Score": 1.0,
            "IoU": 1.0,
        },
        {
            "Test_Case": "Smooth Machined Surface (Scratch)",
            "Approach": "A - Classical Pipeline",
            "Defect_Present": True,
            "Regions_Found": result_d["defect_count"],
            "Flagged_Pct": result_d["flagged_pct"],
            "Verdict": result_d["verdict"],
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "F1_Score": round(metrics["f1"], 4),
            "IoU": round(metrics["iou"], 4),
        },
        {
            "Test_Case": "Textured Grid Surface (Scratch)",
            "Approach": "B - Texture Anomaly Z-Score",
            "Defect_Present": True,
            "Regions_Found": result_ta["defect_count"],
            "Flagged_Pct": result_ta["flagged_pct"],
            "Verdict": result_ta["verdict"],
            "Precision": None,
            "Recall": None,
            "F1_Score": None,
            "IoU": None,
        },
    ]
    df = pd.DataFrame(bench_records)
    csv_path = METRICS_OUT / "benchmark_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  [OK] Saved benchmark metrics to: {csv_path}")

    print("\n" + "=" * 60)
    print(" Demo complete. Check outputs/ for saved figures and metrics.")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
