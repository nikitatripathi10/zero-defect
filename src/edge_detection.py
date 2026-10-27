"""
edge_detection.py
=================
Stage 3: Edge Detection for ZeroDefect.

Implements Canny edge detection with configurable thresholds.

Key insight: Edges alone do NOT confirm a defect. Normal surface textures
also produce edges. Edge maps are one signal among many — they must be
combined with other information (texture, morphology, context) for meaningful
defect localization.
"""

import cv2
import numpy as np


def canny_edges(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    aperture_size: int = 3,
    use_l2_gradient: bool = False,
) -> np.ndarray:
    """
    Detect edges using the Canny algorithm.

    How Canny works (simplified):
    1. Apply Gaussian smoothing to reduce noise.
    2. Compute intensity gradients (how fast pixel values change).
    3. Suppress weak gradients that are not local maxima (non-max suppression).
    4. Apply double thresholding:
       - Pixels above high_threshold → definite edges.
       - Pixels between low and high → edges only if connected to definite edges.
       - Pixels below low_threshold → discarded.
    5. Track edges by connectivity (hysteresis).

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H, W) uint8.
    low_threshold : int
        Lower bound for edge detection (discard weak gradients below this).
    high_threshold : int
        Upper bound — strong edges above this are always kept.
    aperture_size : int
        Size of Sobel kernel used to compute gradients (3, 5, or 7).
    use_l2_gradient : bool
        If True, use L2 norm (more accurate but slower).

    Returns
    -------
    np.ndarray
        Binary edge map (H, W) uint8 — 255 = edge, 0 = background.
    """
    # Ensure input is grayscale
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    edges = cv2.Canny(
        image,
        low_threshold,
        high_threshold,
        apertureSize=aperture_size,
        L2gradient=use_l2_gradient,
    )
    return edges


def auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """
    Automatic Canny edge detection using the median pixel intensity.

    This chooses thresholds automatically based on image statistics,
    which makes it more robust across images with different brightness levels.

    The thresholds are calculated as:
        low  = max(0,   median * (1 - sigma))
        high = min(255, median * (1 + sigma))

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    sigma : float
        Controls how wide the threshold window is around the median.
        Larger sigma → wider window → detects more edges.

    Returns
    -------
    np.ndarray
        Binary edge map (H, W) uint8.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    v = np.median(image)
    low = int(max(0, (1.0 - sigma) * v))
    high = int(min(255, (1.0 + sigma) * v))

    return cv2.Canny(image, low, high)


def sobel_gradient(image: np.ndarray) -> np.ndarray:
    """
    Compute the Sobel gradient magnitude as a continuous (non-binary) edge map.

    Unlike Canny (which produces a binary output), the Sobel gradient gives
    a floating-point map showing *how strongly* pixels vary in intensity.
    High gradient magnitude = likely edge or texture boundary.

    This is useful as an anomaly score: strongly deviating gradients
    in an otherwise uniform region may indicate a defect.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.

    Returns
    -------
    np.ndarray
        Gradient magnitude map (H, W) float32, values in [0, ~360].
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image_f = image.astype(np.float32)
    grad_x = cv2.Sobel(image_f, cv2.CV_32F, 1, 0, ksize=3)  # Horizontal gradient
    grad_y = cv2.Sobel(image_f, cv2.CV_32F, 0, 1, ksize=3)  # Vertical gradient
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return magnitude


def compute_edge_density(edge_map: np.ndarray) -> float:
    """
    Calculate what fraction of the image consists of detected edges.

    Parameters
    ----------
    edge_map : np.ndarray
        Binary edge map (255 = edge).

    Returns
    -------
    float
        Edge density in [0, 1]. A uniformly textured surface will have
        lower density than a heavily defective or edge-rich surface.
    """
    total_pixels = edge_map.size
    edge_pixels = np.count_nonzero(edge_map)
    return edge_pixels / total_pixels
