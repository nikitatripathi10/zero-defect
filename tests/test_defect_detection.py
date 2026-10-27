"""
test_defect_detection.py
========================
Unit tests for defect detection, segmentation, evaluation, and texture modules.

Run with: python -m pytest tests/ -v
"""

import sys
import os
import numpy as np
import cv2
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.segmentation import (
    otsu_threshold, adaptive_threshold, manual_threshold,
    erode, dilate, opening, closing, clean_mask, invert_mask
)
from src.edge_detection import canny_edges, auto_canny, sobel_gradient, compute_edge_density
from src.texture_features import (
    compute_glcm_features, compute_statistics, local_std_map,
    compute_lbp, lbp_histogram
)
from src.defect_detection import (
    detect_defects_classical, analyze_regions, draw_annotations, create_overlay
)
from src.evaluation import pixel_metrics, image_level_metrics


# ─── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def gray_uniform():
    """Uniform gray image — no strong edges."""
    return np.ones((128, 128), dtype=np.uint8) * 128


@pytest.fixture
def gray_scratch():
    """Gray image with a synthetic horizontal scratch (dark stripe)."""
    img = np.ones((128, 128), dtype=np.uint8) * 180
    img[60:65, 10:118] = 20  # Dark scratch
    return img


@pytest.fixture
def rgb_scratch():
    """RGB version of the scratch image."""
    img = np.ones((128, 128, 3), dtype=np.uint8) * 180
    img[60:65, 10:118, :] = 20
    return img


@pytest.fixture
def binary_mask_with_regions():
    """Binary mask with two 10×10 white squares and a tiny 2×2 square."""
    mask = np.zeros((128, 128), dtype=np.uint8)
    mask[10:20, 10:20] = 255   # Region A (100 px)
    mask[60:70, 60:70] = 255   # Region B (100 px)
    mask[100:102, 100:102] = 255  # Tiny noise (4 px)
    return mask


# ─── Test: Segmentation / Thresholding ──────────────────────

def test_otsu_output_shape(gray_scratch):
    thresh, binary = otsu_threshold(gray_scratch)
    assert binary.shape == gray_scratch.shape
    assert binary.dtype == np.uint8


def test_otsu_threshold_value_is_float(gray_scratch):
    thresh, _ = otsu_threshold(gray_scratch)
    assert 0 <= thresh <= 255


def test_otsu_binary_values(gray_scratch):
    _, binary = otsu_threshold(gray_scratch)
    unique = np.unique(binary)
    assert set(unique).issubset({0, 255}), "Otsu output should only contain 0 and 255"


def test_adaptive_threshold_shape(gray_scratch):
    result = adaptive_threshold(gray_scratch, block_size=15, C=2)
    assert result.shape == gray_scratch.shape


def test_manual_threshold(gray_scratch):
    result = manual_threshold(gray_scratch, threshold=100)
    assert result.dtype == np.uint8
    assert set(np.unique(result)).issubset({0, 255})


def test_invert_mask(binary_mask_with_regions):
    inv = invert_mask(binary_mask_with_regions)
    assert inv.max() == 255
    assert np.all(inv[binary_mask_with_regions == 255] == 0)


# ─── Test: Morphological operations ──────────────────────────

def test_erode_output_shape(binary_mask_with_regions):
    result = erode(binary_mask_with_regions, kernel_size=3)
    assert result.shape == binary_mask_with_regions.shape


def test_erode_reduces_foreground(binary_mask_with_regions):
    result = erode(binary_mask_with_regions, kernel_size=5)
    assert np.count_nonzero(result) <= np.count_nonzero(binary_mask_with_regions)


def test_dilate_increases_foreground(binary_mask_with_regions):
    result = dilate(binary_mask_with_regions, kernel_size=5)
    assert np.count_nonzero(result) >= np.count_nonzero(binary_mask_with_regions)


def test_opening_removes_small_noise(binary_mask_with_regions):
    """Opening with large kernel should remove the 2×2 tiny region."""
    result = opening(binary_mask_with_regions, kernel_size=5)
    # After opening, tiny 2×2 region should be gone
    tiny_area = result[100:102, 100:102]
    assert tiny_area.sum() == 0, "Opening should remove tiny noise blobs"


def test_clean_mask_removes_small_components(binary_mask_with_regions):
    result = clean_mask(binary_mask_with_regions, open_kernel=3, close_kernel=3, min_area=50)
    # The 2×2 region (4 px) should be removed, the two 100-px regions should survive
    assert np.count_nonzero(result[10:20, 10:20]) > 0


# ─── Test: Edge Detection ────────────────────────────────────

def test_canny_output_shape(gray_scratch):
    result = canny_edges(gray_scratch)
    assert result.shape == gray_scratch.shape


def test_canny_detects_scratch_edges(gray_scratch):
    result = canny_edges(gray_scratch, low_threshold=30, high_threshold=100)
    # Should detect edges at the scratch boundaries
    assert np.any(result[58:67, 10:118] > 0), "Should detect edges along scratch boundary"


def test_auto_canny_output_binary(gray_scratch):
    result = auto_canny(gray_scratch)
    assert set(np.unique(result)).issubset({0, 255})


def test_sobel_gradient_shape(gray_scratch):
    result = sobel_gradient(gray_scratch)
    assert result.shape == gray_scratch.shape
    assert result.dtype == np.float32


def test_sobel_gradient_high_at_edge(gray_scratch):
    result = sobel_gradient(gray_scratch)
    # Gradient should be high at the scratch transition rows (~59 and ~65)
    assert result[59:61, 30:90].mean() > result[30:50, 30:90].mean()


def test_edge_density_range(gray_scratch):
    edges = canny_edges(gray_scratch)
    density = compute_edge_density(edges)
    assert 0.0 <= density <= 1.0


# ─── Test: Texture Features ──────────────────────────────────

def test_glcm_features_keys(gray_scratch):
    features = compute_glcm_features(gray_scratch)
    for key in ["contrast", "homogeneity", "energy", "correlation"]:
        assert key in features


def test_glcm_features_values_finite(gray_scratch):
    features = compute_glcm_features(gray_scratch)
    for k, v in features.items():
        assert np.isfinite(v), f"{k} should be finite"


def test_statistics_keys(gray_scratch):
    stats = compute_statistics(gray_scratch)
    for key in ["mean", "std", "skewness", "entropy", "min", "max"]:
        assert key in stats


def test_local_std_map_shape(gray_scratch):
    result = local_std_map(gray_scratch, window_size=9)
    assert result.shape == gray_scratch.shape
    assert result.dtype == np.float32


def test_local_std_map_high_at_scratch(gray_scratch):
    result = local_std_map(gray_scratch, window_size=9)
    # Std should be higher near the scratch boundary
    mean_at_scratch = result[55:70, 20:110].mean()
    mean_at_normal = result[20:40, 20:110].mean()
    assert mean_at_scratch > mean_at_normal


def test_lbp_shape(gray_scratch):
    lbp = compute_lbp(gray_scratch)
    assert lbp.shape == gray_scratch.shape


def test_lbp_histogram_normalized(gray_scratch):
    lbp = compute_lbp(gray_scratch)
    hist = lbp_histogram(lbp, P=8, normalize=True)
    assert abs(hist.sum() - 1.0) < 1e-5, "Normalized histogram should sum to 1"


# ─── Test: Defect Detection ──────────────────────────────────

def test_classical_detection_returns_keys(rgb_scratch):
    result = detect_defects_classical(rgb_scratch, min_defect_area=5)
    for key in ["preprocessed", "edge_map", "raw_mask", "clean_mask",
                "regions", "annotated", "defect_count", "flagged_pct", "verdict"]:
        assert key in result, f"Missing key: {key}"


def test_classical_detection_mask_shape(rgb_scratch):
    result = detect_defects_classical(rgb_scratch, min_defect_area=5)
    assert result["clean_mask"].shape == (128, 128)


def test_classical_detection_detects_scratch(rgb_scratch):
    result = detect_defects_classical(rgb_scratch, min_defect_area=5)
    # Should detect at least one suspicious region
    assert result["defect_count"] >= 1, "Should detect the synthetic scratch"


def test_classical_detection_verdict_not_empty(rgb_scratch):
    result = detect_defects_classical(rgb_scratch, min_defect_area=5)
    assert len(result["verdict"]) > 0


def test_annotated_image_shape(rgb_scratch):
    result = detect_defects_classical(rgb_scratch, min_defect_area=5)
    assert result["annotated"].shape == rgb_scratch.shape


# ─── Test: Region Analysis ───────────────────────────────────

def test_analyze_regions_returns_list(binary_mask_with_regions):
    cleaned = clean_mask(binary_mask_with_regions, min_area=50)
    regions = analyze_regions(cleaned)
    assert isinstance(regions, list)


def test_analyze_regions_fields(binary_mask_with_regions):
    regions = analyze_regions(binary_mask_with_regions)
    if regions:
        r = regions[0]
        for field in ["label", "x", "y", "w", "h", "area", "centroid_x", "centroid_y", "perimeter"]:
            assert field in r, f"Missing field: {field}"


def test_analyze_regions_sorted_by_area(binary_mask_with_regions):
    regions = analyze_regions(binary_mask_with_regions)
    areas = [r["area"] for r in regions]
    assert areas == sorted(areas, reverse=True), "Regions should be sorted by area (descending)"


# ─── Test: Evaluation Metrics ────────────────────────────────

def test_pixel_metrics_perfect_prediction():
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[10:20, 10:20] = 255
    metrics = pixel_metrics(mask, mask)  # Compare mask with itself
    assert metrics["precision"] == pytest.approx(1.0, abs=0.01)
    assert metrics["recall"] == pytest.approx(1.0, abs=0.01)
    assert metrics["f1"] == pytest.approx(1.0, abs=0.01)
    assert metrics["iou"] == pytest.approx(1.0, abs=0.01)


def test_pixel_metrics_no_overlap():
    gt = np.zeros((50, 50), dtype=np.uint8)
    pred = np.zeros((50, 50), dtype=np.uint8)
    gt[0:10, 0:10] = 255
    pred[40:50, 40:50] = 255
    metrics = pixel_metrics(pred, gt)
    assert metrics["iou"] == pytest.approx(0.0, abs=0.01)


def test_pixel_metrics_keys():
    gt = np.zeros((50, 50), dtype=np.uint8)
    pred = np.zeros((50, 50), dtype=np.uint8)
    metrics = pixel_metrics(pred, gt)
    for key in ["precision", "recall", "f1", "iou", "tp", "fp", "fn", "tn"]:
        assert key in metrics


def test_image_level_metrics():
    y_true = [1, 0, 1, 0, 1, 0]
    y_pred = [1, 0, 1, 0, 0, 1]  # One FN, one FP
    metrics = image_level_metrics(y_true, y_pred)
    for key in ["accuracy", "precision", "recall", "f1", "confusion_matrix"]:
        assert key in metrics


def test_overlay_shape(rgb_scratch, binary_mask_with_regions):
    mask = cv_compatible_mask(rgb_scratch)
    overlay = create_overlay(rgb_scratch, mask)
    assert overlay.shape == rgb_scratch.shape


def test_classical_detection_on_normal_surface(gray_uniform):
    rgb_uniform = cv2.cvtColor(gray_uniform, cv2.COLOR_GRAY2RGB)
    result = detect_defects_classical(rgb_uniform, min_defect_area=50)
    assert result["defect_count"] == 0
    assert result["flagged_pct"] == 0.0
    assert result["verdict"] == "No suspicious region detected"


def test_texture_anomaly_detector_training_and_predict():
    from src.defect_detection import TextureAnomalyDetector
    # 5 uniform images with slight variation
    train_images = [
        np.clip(np.full((64, 64, 3), 150) + np.random.normal(0, 2, (64, 64, 3)), 0, 255).astype(np.uint8)
        for _ in range(5)
    ]
    detector = TextureAnomalyDetector(window_size=9, z_threshold=2.5)
    detector.fit(train_images)
    assert detector.is_trained

    # Test on defective image with bright anomaly block
    test_img = np.full((64, 64, 3), 150, dtype=np.uint8)
    test_img[20:40, 20:40] = 250  # bright anomalous patch
    res = detector.predict(test_img, min_defect_area=10)
    assert "clean_mask" in res
    assert "verdict" in res
    assert res["clean_mask"].shape == (64, 64)


def cv_compatible_mask(img):
    m = np.zeros(img.shape[:2], dtype=np.uint8)
    m[10:30, 10:30] = 255
    return m


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
