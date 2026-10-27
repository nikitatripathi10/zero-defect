"""
segmentation.py
===============
Stage 4 & 5: Thresholding, segmentation, and morphological processing.

This module converts a grayscale or edge image into a binary mask
(0 = background, 255 = suspicious region) using multiple methods.

Stage 4 — Thresholding:
  - Otsu's method (global, automatic)
  - Adaptive thresholding (local, handles uneven lighting)
  - Manual threshold (user-controlled)

Stage 5 — Morphological operations:
  - Erosion, Dilation, Opening, Closing
  - Used to clean up masks and connect relevant regions

Each function includes a plain-language explanation of what it does.
"""

import cv2
import numpy as np


# ──────────────────────────────────────────────
# THRESHOLDING  (Stage 4)
# ──────────────────────────────────────────────

def otsu_threshold(image: np.ndarray) -> tuple:
    """
    Apply Otsu's automatic global thresholding.

    Otsu's method finds the single threshold value that *minimizes*
    the variance of pixel intensities within each of the two classes
    (foreground / background). It is "automatic" because it reads the
    image histogram and picks the best split point.

    Limitation: Works well only when the image has a clear bimodal
    histogram (two distinct peaks). Uneven lighting can fool it.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H, W).

    Returns
    -------
    tuple
        (threshold_value, binary_mask)
        - threshold_value: the automatically chosen cut-off (float)
        - binary_mask: uint8 image — 255 where pixel > threshold, 0 elsewhere
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    thresh_val, binary = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh_val, binary


def adaptive_threshold(
    image: np.ndarray,
    block_size: int = 31,
    C: int = 5,
    method: str = "gaussian",
) -> np.ndarray:
    """
    Apply adaptive (local) thresholding.

    Instead of one global threshold, adaptive thresholding computes a
    separate threshold for each small block of pixels based on local
    neighborhood statistics. This handles images with uneven illumination
    much better than Otsu's method.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H, W).
    block_size : int
        Size of the pixel neighborhood used to calculate the threshold.
        Must be an odd number ≥ 3.
    C : int
        Constant subtracted from the mean or weighted mean.
        Increase to keep fewer pixels as foreground.
    method : str
        "gaussian" (weighted average) or "mean" (simple average of neighborhood).

    Returns
    -------
    np.ndarray
        Binary mask (H, W) uint8.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    if block_size % 2 == 0:
        block_size += 1

    adaptive_method = (
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        if method == "gaussian"
        else cv2.ADAPTIVE_THRESH_MEAN_C
    )

    binary = cv2.adaptiveThreshold(
        image,
        255,
        adaptive_method,
        cv2.THRESH_BINARY,
        block_size,
        C,
    )
    return binary


def manual_threshold(image: np.ndarray, threshold: int = 128) -> np.ndarray:
    """
    Apply a user-specified global threshold.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    threshold : int
        Pixels above this value → 255 (foreground). Below → 0.

    Returns
    -------
    np.ndarray
        Binary mask (H, W) uint8.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return binary


def invert_mask(mask: np.ndarray) -> np.ndarray:
    """Invert a binary mask (swap 0 and 255)."""
    return cv2.bitwise_not(mask)


# ──────────────────────────────────────────────
# MORPHOLOGICAL PROCESSING  (Stage 5)
# ──────────────────────────────────────────────

def get_kernel(size: int = 5, shape: int = cv2.MORPH_RECT) -> np.ndarray:
    """
    Create a morphological structuring element (kernel).

    Parameters
    ----------
    size : int
        Kernel width and height (e.g., 5 → 5×5 kernel).
    shape : int
        cv2.MORPH_RECT  (rectangle)
        cv2.MORPH_ELLIPSE (ellipse — more natural for rounded defects)
        cv2.MORPH_CROSS (cross-shaped)

    Returns
    -------
    np.ndarray
        Kernel array.
    """
    return cv2.getStructuringElement(shape, (size, size))


def erode(mask: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    """
    Erode a binary mask: shrink white (foreground) regions.

    Erosion removes pixels at the *boundary* of foreground regions.
    Use it to eliminate small isolated noise blobs before further analysis.

    Parameters
    ----------
    mask : np.ndarray  Binary mask.
    kernel_size : int  How aggressively to erode. Larger = more shrinking.
    iterations : int   How many times to apply erosion.

    Returns
    -------
    np.ndarray  Eroded mask.
    """
    kernel = get_kernel(kernel_size)
    return cv2.erode(mask, kernel, iterations=iterations)


def dilate(mask: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    """
    Dilate a binary mask: expand white (foreground) regions.

    Dilation adds pixels around foreground boundaries.
    Use it to close small gaps between defect fragments or to expand
    detected regions slightly for better bounding-box visualization.

    Parameters
    ----------
    mask : np.ndarray  Binary mask.
    kernel_size : int  How aggressively to dilate.
    iterations : int   Number of dilation passes.

    Returns
    -------
    np.ndarray  Dilated mask.
    """
    kernel = get_kernel(kernel_size)
    return cv2.dilate(mask, kernel, iterations=iterations)


def opening(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply morphological opening (erosion then dilation).

    Opening removes small noise blobs and thin protrusions while
    preserving larger foreground regions. Useful for cleaning up
    scattered false-positive pixels from thresholding.

    Parameters
    ----------
    mask : np.ndarray  Binary mask.
    kernel_size : int  Structuring element size.

    Returns
    -------
    np.ndarray  Opened mask.
    """
    kernel = get_kernel(kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)


def closing(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply morphological closing (dilation then erosion).

    Closing fills small holes inside foreground regions and connects
    nearby fragments. Useful when a defect appears as disconnected
    segments that logically belong together.

    Parameters
    ----------
    mask : np.ndarray  Binary mask.
    kernel_size : int  Structuring element size.

    Returns
    -------
    np.ndarray  Closed mask.
    """
    kernel = get_kernel(kernel_size)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def clean_mask(
    mask: np.ndarray,
    open_kernel: int = 5,
    close_kernel: int = 7,
    min_area: int = 100,
) -> np.ndarray:
    """
    Standard mask cleaning pipeline:
    1. Opening  → remove small noise blobs
    2. Closing  → fill gaps inside real defect regions
    3. Remove connected components smaller than min_area pixels

    Parameters
    ----------
    mask : np.ndarray
        Raw binary mask (H, W) uint8.
    open_kernel : int
        Kernel size for opening step.
    close_kernel : int
        Kernel size for closing step.
    min_area : int
        Any connected component with fewer pixels than this will be removed.

    Returns
    -------
    np.ndarray
        Cleaned binary mask.
    """
    # Step 1 & 2: opening then closing
    cleaned = opening(mask, open_kernel)
    cleaned = closing(cleaned, close_kernel)

    # Step 3: remove tiny components
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )
    output = np.zeros_like(cleaned)
    for i in range(1, n_labels):  # Skip label 0 (background)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            output[labels == i] = 255

    return output
