"""
test_preprocessing.py
=====================
Unit tests for ZeroDefect preprocessing functions.

Run with: python -m pytest tests/ -v
"""

import sys
import os
import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import (
    load_image_from_bytes, to_grayscale, gaussian_blur, median_blur,
    apply_clahe, normalize_brightness, preprocess, get_image_info, resize_image
)


# ─── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def rgb_image():
    """A simple 100×100 RGB image with known pixel values."""
    np.random.seed(0)
    return np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def gray_image():
    """A 100×100 grayscale image."""
    np.random.seed(1)
    return np.random.randint(50, 200, (100, 100), dtype=np.uint8)


# ─── Test: grayscale conversion ──────────────────────────────

def test_to_grayscale_output_shape(rgb_image):
    gray = to_grayscale(rgb_image)
    assert gray.ndim == 2, "Output should be 2-D (H, W)"
    assert gray.shape == (100, 100)


def test_to_grayscale_passthrough_on_gray(gray_image):
    result = to_grayscale(gray_image)
    assert np.array_equal(result, gray_image), "Grayscale input should be returned unchanged"


def test_to_grayscale_dtype(rgb_image):
    gray = to_grayscale(rgb_image)
    assert gray.dtype == np.uint8


# ─── Test: Gaussian blur ──────────────────────────────────────

def test_gaussian_blur_output_shape(gray_image):
    blurred = gaussian_blur(gray_image, kernel_size=5)
    assert blurred.shape == gray_image.shape


def test_gaussian_blur_even_kernel_correction(gray_image):
    # Even kernel should be auto-corrected to odd
    blurred = gaussian_blur(gray_image, kernel_size=4)
    assert blurred.shape == gray_image.shape


def test_gaussian_blur_reduces_variance(gray_image):
    blurred = gaussian_blur(gray_image, kernel_size=9)
    assert blurred.std() <= gray_image.std() + 1  # Blurring should reduce std


# ─── Test: Median blur ───────────────────────────────────────

def test_median_blur_output_shape(gray_image):
    result = median_blur(gray_image, kernel_size=5)
    assert result.shape == gray_image.shape


def test_median_blur_dtype(gray_image):
    result = median_blur(gray_image, kernel_size=5)
    assert result.dtype == np.uint8


# ─── Test: CLAHE ─────────────────────────────────────────────

def test_clahe_output_shape(gray_image):
    result = apply_clahe(gray_image)
    assert result.shape == gray_image.shape


def test_clahe_dtype(gray_image):
    result = apply_clahe(gray_image)
    assert result.dtype == np.uint8


def test_clahe_value_range(gray_image):
    result = apply_clahe(gray_image)
    assert result.min() >= 0
    assert result.max() <= 255


def test_clahe_accepts_rgb(rgb_image):
    """CLAHE should auto-convert RGB to grayscale."""
    result = apply_clahe(rgb_image)
    assert result.ndim == 2  # Output must be grayscale


# ─── Test: Normalize brightness ──────────────────────────────

def test_normalize_range(gray_image):
    result = normalize_brightness(gray_image)
    assert result.min() == 0
    assert result.max() == 255


# ─── Test: get_image_info ────────────────────────────────────

def test_get_image_info_rgb(rgb_image):
    info = get_image_info(rgb_image)
    assert info["height"] == 100
    assert info["width"] == 100
    assert info["channels"] == 3


def test_get_image_info_gray(gray_image):
    info = get_image_info(gray_image)
    assert info["channels"] == 1


# ─── Test: resize_image ──────────────────────────────────────

def test_resize_output_size(rgb_image):
    resized = resize_image(rgb_image, target_size=(64, 64))
    assert resized.shape == (64, 64, 3)


# ─── Test: full preprocess pipeline ──────────────────────────

def test_preprocess_returns_all_keys(rgb_image):
    result = preprocess(rgb_image)
    for key in ["original", "gray", "blurred", "enhanced", "normalized"]:
        assert key in result, f"Missing key: {key}"


def test_preprocess_correct_shapes(rgb_image):
    result = preprocess(rgb_image)
    H, W = rgb_image.shape[:2]
    assert result["gray"].shape == (H, W)
    assert result["blurred"].shape == (H, W)
    assert result["enhanced"].shape == (H, W)


def test_preprocess_median_blur(rgb_image):
    result = preprocess(rgb_image, blur_method="median", blur_kernel=5)
    assert result["blurred"].shape == rgb_image.shape[:2]


def test_preprocess_no_clahe(rgb_image):
    result = preprocess(rgb_image, use_clahe=False)
    # enhanced should equal blurred when CLAHE is disabled
    assert np.array_equal(result["enhanced"], result["blurred"])


# ─── Test: load from bytes ───────────────────────────────────

def test_load_from_bytes_invalid():
    with pytest.raises(ValueError):
        load_image_from_bytes(b"not_an_image")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
