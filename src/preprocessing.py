"""
preprocessing.py
================
Stage 1 & 2 of the ZeroDefect pipeline.

Handles:
  - Image loading and validation
  - Grayscale conversion
  - Gaussian and Median blur
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
  - Brightness normalization
  - Side-by-side comparison

All functions are written to be simple, readable, and reusable.
"""

import cv2
import numpy as np
from pathlib import Path


# ──────────────────────────────────────────────
# 1. IMAGE LOADING
# ──────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk and return it as an RGB NumPy array.

    OpenCV reads images in BGR order by default, so we convert to RGB
    so that every other function in this project works in RGB.

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the image file.

    Returns
    -------
    np.ndarray
        Image as an (H, W, 3) uint8 RGB array.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not supported or the image cannot be read.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type '{path.suffix}'. "
            f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    # cv2.imread returns BGR; we convert to RGB immediately.
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(
            f"Could not read image at '{image_path}'. "
            "The file may be corrupted or in an unsupported format."
        )

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def load_image_from_bytes(file_bytes: bytes) -> np.ndarray:
    """
    Decode an image from raw bytes (e.g., from a Streamlit file uploader).

    Parameters
    ----------
    file_bytes : bytes
        Raw bytes of the image file.

    Returns
    -------
    np.ndarray
        Image as an (H, W, 3) uint8 RGB array.
    """
    arr = np.frombuffer(file_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode image from uploaded bytes.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def get_image_info(image: np.ndarray) -> dict:
    """
    Return a summary dictionary describing the image.

    Parameters
    ----------
    image : np.ndarray
        Input image (RGB or grayscale).

    Returns
    -------
    dict
        Keys: height, width, channels, dtype, min_pixel, max_pixel, mean_pixel
    """
    if image.ndim == 2:
        h, w = image.shape
        channels = 1
    else:
        h, w, channels = image.shape

    return {
        "height": h,
        "width": w,
        "channels": channels,
        "dtype": str(image.dtype),
        "min_pixel": int(image.min()),
        "max_pixel": int(image.max()),
        "mean_pixel": round(float(image.mean()), 2),
    }


def resize_image(image: np.ndarray, target_size: tuple = (512, 512)) -> np.ndarray:
    """
    Resize an image to target_size while preserving content.

    Parameters
    ----------
    image : np.ndarray
        Input image.
    target_size : tuple
        (width, height) in pixels. Default: (512, 512).

    Returns
    -------
    np.ndarray
        Resized image.
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


# ──────────────────────────────────────────────
# 2. GRAYSCALE CONVERSION
# ──────────────────────────────────────────────

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB image to grayscale.

    If the image is already grayscale (2-D), it is returned as-is.

    Parameters
    ----------
    image : np.ndarray
        RGB image (H, W, 3) or grayscale (H, W).

    Returns
    -------
    np.ndarray
        Grayscale image (H, W) uint8.
    """
    if image.ndim == 2:
        return image  # Already grayscale
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


# ──────────────────────────────────────────────
# 3. NOISE REDUCTION
# ──────────────────────────────────────────────

def gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 0) -> np.ndarray:
    """
    Apply Gaussian blur to reduce high-frequency noise.

    Gaussian blur works by averaging each pixel with its neighbors,
    giving more weight to nearby pixels (following a bell-curve shape).
    This smooths out random noise without losing major structural features.

    Parameters
    ----------
    image : np.ndarray
        Grayscale or RGB input image.
    kernel_size : int
        Size of the blur kernel (must be odd). Larger = more blurring.
    sigma : float
        Standard deviation of the Gaussian. 0 = auto-calculated from kernel size.

    Returns
    -------
    np.ndarray
        Blurred image.
    """
    # Kernel size must be a positive odd integer
    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def median_blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Apply Median blur — excellent for removing salt-and-pepper noise.

    Unlike Gaussian blur, median blur replaces each pixel with the *median*
    value of its neighborhood. This is very good at removing isolated bright
    or dark speckles while preserving edges better than Gaussian blur.

    Parameters
    ----------
    image : np.ndarray
        Grayscale or RGB input image.
    kernel_size : int
        Size of the kernel (must be odd). Larger = stronger smoothing.

    Returns
    -------
    np.ndarray
        Median-blurred image.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)


# ──────────────────────────────────────────────
# 4. CONTRAST ENHANCEMENT
# ──────────────────────────────────────────────

def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Regular histogram equalization boosts contrast globally, which can
    over-amplify noise. CLAHE works on small tiles independently and
    *clips* contrast amplification to avoid noise blow-up. This makes
    subtle surface defects much more visible.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image (H, W). If RGB is passed, it will be converted.
    clip_limit : float
        Maximum contrast amplification factor. Higher = more contrast.
        Typical range: 1.0 – 4.0.
    tile_grid_size : tuple
        Number of tiles in (x, y). Smaller tiles = more local adaptation.

    Returns
    -------
    np.ndarray
        CLAHE-enhanced grayscale image (H, W) uint8.
    """
    gray = to_grayscale(image) if image.ndim == 3 else image
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


def normalize_brightness(image: np.ndarray) -> np.ndarray:
    """
    Stretch pixel intensities to use the full [0, 255] range.

    This is useful when an image is very dark or very bright overall.
    It re-scales so the darkest pixel becomes 0 and the brightest becomes 255.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.

    Returns
    -------
    np.ndarray
        Normalized image (H, W) uint8.
    """
    normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return normalized.astype(np.uint8)


# ──────────────────────────────────────────────
# 5. FULL PREPROCESSING PIPELINE
# ──────────────────────────────────────────────

def preprocess(
    image: np.ndarray,
    blur_method: str = "gaussian",
    blur_kernel: int = 5,
    use_clahe: bool = True,
    clahe_clip: float = 2.0,
    normalize: bool = False,
) -> dict:
    """
    Run the complete preprocessing pipeline and return all intermediate results.

    This makes it easy to compare each step visually in the interface.

    Parameters
    ----------
    image : np.ndarray
        Original RGB image.
    blur_method : str
        "gaussian" or "median".
    blur_kernel : int
        Kernel size for blur.
    use_clahe : bool
        Whether to apply CLAHE after blurring.
    clahe_clip : float
        CLAHE clip limit.
    normalize : bool
        Whether to normalize brightness.

    Returns
    -------
    dict
        Keys:
          - "original"   : Original RGB image
          - "gray"       : Grayscale image
          - "blurred"    : After blur
          - "enhanced"   : After CLAHE (or blurred if CLAHE disabled)
          - "normalized" : After normalization (same as enhanced if disabled)
    """
    results = {"original": image}

    # Step 1: Grayscale
    gray = to_grayscale(image)
    results["gray"] = gray

    # Step 2: Blur
    if blur_method == "median":
        blurred = median_blur(gray, blur_kernel)
    else:
        blurred = gaussian_blur(gray, blur_kernel)
    results["blurred"] = blurred

    # Step 3: CLAHE contrast enhancement
    if use_clahe:
        enhanced = apply_clahe(blurred, clip_limit=clahe_clip)
    else:
        enhanced = blurred.copy()
    results["enhanced"] = enhanced

    # Step 4: Optional brightness normalization
    if normalize:
        results["normalized"] = normalize_brightness(enhanced)
    else:
        results["normalized"] = enhanced

    return results
