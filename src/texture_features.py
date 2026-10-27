"""
texture_features.py
===================
Stage 6: Texture Feature Extraction for ZeroDefect.

Surface defects often change the texture of a surface — a crack disrupts
regularity, a contamination patch adds foreign texture, etc.

This module extracts numerical descriptions of texture from image patches
or from the whole image, allowing the pipeline to detect abnormal regions.

Features implemented:
  1. GLCM (Gray-Level Co-occurrence Matrix) — captures spatial relationships
     between pixel intensities: contrast, homogeneity, energy, correlation.
  2. LBP (Local Binary Patterns) — encodes local texture patterns as binary codes.
  3. Simple statistics — mean, std, skewness, entropy.

Why texture matters:
  A normal surface has a predictable, consistent texture.
  A defective region looks different — it deviates from this expected pattern.
"""

import cv2
import numpy as np
from scipy.stats import skew, entropy as scipy_entropy
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern


# ──────────────────────────────────────────────
# GLCM FEATURES
# ──────────────────────────────────────────────

def compute_glcm_features(
    image: np.ndarray,
    distances: list = [1, 3],
    angles: list = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    levels: int = 256,
) -> dict:
    """
    Compute GLCM (Gray-Level Co-occurrence Matrix) texture features.

    A GLCM counts how often pairs of pixels at a given spatial relationship
    (distance and angle) occur in an image. From this matrix we can extract:

    - Contrast    : Measures how much neighboring pixels differ.
                    High contrast → big intensity differences → possible cracks/edges.
    - Homogeneity : How close values are to the diagonal (similar neighbors).
                    Low homogeneity → rough/varied surface texture.
    - Energy      : Sum of squared elements. High energy → uniform texture.
    - Correlation : How correlated a pixel is to its neighbor.
                    Normal surfaces with regular patterns have high correlation.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H, W). Large images may be slow; consider resizing.
    distances : list
        List of pixel distances to use. [1, 3] captures both fine and coarse texture.
    angles : list
        List of angles (radians). Four angles capture all major directions.
    levels : int
        Number of gray levels. 256 = full uint8 range.

    Returns
    -------
    dict
        Keys: contrast, homogeneity, energy, correlation (each averaged over
        all distance/angle combinations).
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # GLCM is computed at specified distances and angles
    glcm = graycomatrix(
        image,
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True,
    )

    features = {
        "contrast": float(graycoprops(glcm, "contrast").mean()),
        "homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "energy": float(graycoprops(glcm, "energy").mean()),
        "correlation": float(graycoprops(glcm, "correlation").mean()),
    }
    return features


def compute_glcm_features_patch(
    image: np.ndarray,
    patch_size: int = 32,
    stride: int = 16,
) -> np.ndarray:
    """
    Compute GLCM contrast feature for sliding patches across the image.

    This creates a spatial map of texture contrast — patches with unusually
    high contrast compared to their neighbors may indicate defective regions.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    patch_size : int
        Size of each square patch (pixels).
    stride : int
        Number of pixels to step between patches. Smaller = finer map.

    Returns
    -------
    np.ndarray
        2-D array of contrast values per patch position (float32).
        Shape ≈ (n_rows, n_cols) covering the image at the chosen stride.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    H, W = image.shape
    rows = range(0, H - patch_size, stride)
    cols = range(0, W - patch_size, stride)

    contrast_map = np.zeros((len(rows), len(cols)), dtype=np.float32)

    for ri, r in enumerate(rows):
        for ci, c in enumerate(cols):
            patch = image[r : r + patch_size, c : c + patch_size]
            glcm = graycomatrix(
                patch, distances=[1], angles=[0], levels=256,
                symmetric=True, normed=True
            )
            contrast_map[ri, ci] = float(graycoprops(glcm, "contrast").mean())

    return contrast_map


# ──────────────────────────────────────────────
# LOCAL BINARY PATTERNS (LBP)
# ──────────────────────────────────────────────

def compute_lbp(
    image: np.ndarray,
    P: int = 8,
    R: float = 1.0,
    method: str = "uniform",
) -> np.ndarray:
    """
    Compute Local Binary Patterns (LBP) of an image.

    LBP encodes each pixel as a binary number by comparing its intensity
    to its circular neighbors. If a neighbor is brighter → 1, else → 0.
    The resulting code captures the local texture pattern.

    "Uniform" LBP patterns are those with at most 2 bit transitions
    (e.g., 00001111) and are the most informative. The remaining
    patterns are grouped into a single "non-uniform" bin.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    P : int
        Number of neighbors to sample in the circular pattern.
    R : float
        Radius of the circular neighborhood.
    method : str
        LBP computation method. "uniform" is standard for texture analysis.

    Returns
    -------
    np.ndarray
        LBP image (H, W) float64 — each pixel's LBP code.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    lbp = local_binary_pattern(image, P, R, method=method)
    return lbp


def lbp_histogram(lbp_image: np.ndarray, P: int = 8, normalize: bool = True) -> np.ndarray:
    """
    Compute the histogram of an LBP image.

    The histogram summarizes the texture: how often each binary pattern
    appears. Comparing histograms between a "normal" patch and a "test"
    patch lets us detect texture anomalies.

    Parameters
    ----------
    lbp_image : np.ndarray
        Output of compute_lbp().
    P : int
        Number of neighbors used when computing the LBP.
    normalize : bool
        If True, normalize histogram to sum to 1 (probability distribution).

    Returns
    -------
    np.ndarray
        1-D histogram array.
    """
    n_bins = P + 2  # P uniform patterns + 1 non-uniform bin
    hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, n_bins))
    if normalize:
        hist = hist.astype(float) / (hist.sum() + 1e-8)
    return hist


# ──────────────────────────────────────────────
# INTENSITY STATISTICS
# ──────────────────────────────────────────────

def compute_statistics(image: np.ndarray) -> dict:
    """
    Compute simple statistical features of pixel intensities.

    These are the simplest possible "texture" features:
    - Mean:     Average brightness.
    - Std Dev:  How much pixel values vary. High std → rough / varied surface.
    - Skewness: Asymmetry of the intensity distribution.
    - Entropy:  Information content. High entropy → complex / noisy texture.
    - Min/Max:  Extremes.

    Parameters
    ----------
    image : np.ndarray
        Grayscale or RGB image.

    Returns
    -------
    dict
        Statistical feature values.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    flat = image.ravel().astype(float)

    # Entropy requires a probability distribution (histogram)
    hist, _ = np.histogram(flat, bins=256, range=(0, 256), density=True)
    ent = float(scipy_entropy(hist + 1e-10))

    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "skewness": float(skew(flat)),
        "entropy": ent,
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "range": float(np.max(flat) - np.min(flat)),
    }


# ──────────────────────────────────────────────
# ANOMALY SCORING FROM TEXTURE
# ──────────────────────────────────────────────

def local_std_map(image: np.ndarray, window_size: int = 15) -> np.ndarray:
    """
    Compute local standard deviation at every pixel using a sliding window.

    A uniform, defect-free surface has low local standard deviation.
    A scratch or crack introduces high local variation → high std in that area.

    This creates a continuous anomaly score map where high values suggest
    potentially defective pixels.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    window_size : int
        Size of the local neighborhood (must be odd).

    Returns
    -------
    np.ndarray
        Local std map (H, W) float32.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    img_f = image.astype(np.float32)

    # E[X] and E[X^2] using box filter (fast approximation of sliding window)
    mean = cv2.boxFilter(img_f, -1, (window_size, window_size))
    mean_sq = cv2.boxFilter(img_f ** 2, -1, (window_size, window_size))

    # Var[X] = E[X^2] - E[X]^2 ; clamp negatives caused by numerical precision
    variance = np.maximum(mean_sq - mean ** 2, 0)
    std_map = np.sqrt(variance)

    return std_map.astype(np.float32)
