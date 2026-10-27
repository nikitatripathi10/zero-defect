"""
defect_detection.py
===================
Stage 7 & 8: Defect Localization and Region Analysis.

Two complementary approaches are implemented:

  Approach A — Classical Pipeline
  --------------------------------
  Uses traditional image processing: preprocessing → edge detection →
  thresholding → morphological cleanup → contour extraction.
  Simple, fast, interpretable, no training required.

  Approach B — Texture Anomaly Detection
  ----------------------------------------
  Builds a statistical model of "normal" surface appearance using
  training images, then flags regions that deviate from this model.
  More sensitive to subtle defects but requires normal reference images.

Stage 8: Region Analysis
  Extracts measurements from detected regions:
  - Connected component count
  - Per-region area, bounding box
  - % of image flagged
  - Contour perimeter

All results are clearly labelled as image-based estimates.
"""

import cv2
import numpy as np
from pathlib import Path

from src.preprocessing import preprocess
from src.edge_detection import canny_edges, sobel_gradient, auto_canny
from src.segmentation import otsu_threshold, adaptive_threshold, clean_mask, invert_mask
from src.texture_features import local_std_map, compute_glcm_features


# ─────────────────────────────────────────────────────────────
# APPROACH A: CLASSICAL IMAGE PROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────

def detect_defects_classical(
    image: np.ndarray,
    preprocess_config: dict = None,
    edge_method: str = "auto_canny",
    low_thresh: int = 50,
    high_thresh: int = 150,
    threshold_method: str = "otsu",
    open_kernel: int = 3,
    close_kernel: int = 7,
    min_defect_area: int = 150,
) -> dict:
    """
    Detect surface defects using classical image processing.

    Pipeline:
    1. Preprocess (grayscale → blur → CLAHE)
    2. Detect edges (Canny)
    3. Threshold to binary mask
    4. Morphological cleanup (remove noise, connect fragments)
    5. Extract and filter connected components

    Parameters
    ----------
    image : np.ndarray
        RGB input image.
    preprocess_config : dict
        Keyword arguments passed to preprocess(). Default: Gaussian blur + CLAHE.
    edge_method : str
        "canny", "auto_canny", or "sobel_threshold".
    low_thresh, high_thresh : int
        Canny thresholds (used only for "canny" method).
    threshold_method : str
        "otsu" or "adaptive" — used for Sobel-based or raw grayscale thresholding.
    open_kernel, close_kernel : int
        Morphological kernel sizes.
    min_defect_area : int
        Discard regions smaller than this (pixels).

    Returns
    -------
    dict
        {
          "preprocessed":  dict from preprocess()
          "edge_map":      binary edge image (H, W)
          "raw_mask":      threshold result before morphology
          "clean_mask":    mask after morphological cleanup
          "regions":       list of region dicts (see _analyze_regions)
          "annotated":     original image with bounding boxes overlaid
          "defect_count":  number of detected suspicious regions
          "flagged_pct":   % of image area flagged as suspicious
          "verdict":       "No suspicious region detected" | "Potential surface anomaly detected"
        }
    """
    config = preprocess_config or {}
    prep = preprocess(image, **config)
    enhanced = prep["enhanced"]  # CLAHE-enhanced grayscale

    # ── Edge detection ──────────────────────────────
    if edge_method == "canny":
        edge_map = canny_edges(enhanced, low_thresh, high_thresh)
    elif edge_method == "sobel_threshold":
        grad = sobel_gradient(enhanced)
        grad_norm = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, edge_map = cv2.threshold(grad_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:  # auto_canny (default)
        edge_map = auto_canny(enhanced)

    # ── Thresholding & Anomaly Extraction ──────────
    # Compare with smoothed background to isolate true surface defects
    # (dark scratches/cracks/pits or bright burns/glints) rather than
    # globally splitting the histogram of uniform surfaces.
    bg = cv2.GaussianBlur(enhanced, (25, 25), 0)
    diff_dark = cv2.subtract(bg, enhanced)
    diff_bright = cv2.subtract(enhanced, bg)
    diff = cv2.max(diff_dark, diff_bright)

    intensity_mask = np.zeros_like(enhanced)
    if threshold_method != "edge_only" and diff.max() >= 25:
        if threshold_method == "adaptive":
            intensity_mask = adaptive_threshold(diff, block_size=31, C=5)
        else:
            _, intensity_mask = otsu_threshold(diff)

    if threshold_method == "edge_only":
        combined = edge_map
    else:
        combined = cv2.bitwise_or(edge_map, intensity_mask)
    raw_mask = combined

    # ── Morphological cleanup ──────────────────────
    mask = clean_mask(
        raw_mask,
        open_kernel=open_kernel,
        close_kernel=close_kernel,
        min_area=min_defect_area,
    )

    # ── Region analysis ────────────────────────────
    regions = analyze_regions(mask)
    annotated = draw_annotations(image, regions)

    total_pixels = mask.size
    flagged_pixels = int(np.count_nonzero(mask))
    flagged_pct = round(100.0 * flagged_pixels / total_pixels, 2)

    defect_count = len(regions)
    if defect_count == 0:
        verdict = "No suspicious region detected"
    elif flagged_pct < 0.5:
        verdict = "Inspection inconclusive — minor anomaly signals"
    else:
        verdict = "Potential surface anomaly detected"

    return {
        "preprocessed": prep,
        "edge_map": edge_map,
        "raw_mask": raw_mask,
        "clean_mask": mask,
        "regions": regions,
        "annotated": annotated,
        "defect_count": defect_count,
        "flagged_pct": flagged_pct,
        "verdict": verdict,
    }


# ─────────────────────────────────────────────────────────────
# APPROACH B: TEXTURE ANOMALY DETECTION
# ─────────────────────────────────────────────────────────────

class TextureAnomalyDetector:
    """
    Texture-based anomaly detector that learns from normal images.

    Training:
      - Compute local standard deviation maps for several normal images.
      - Record the mean and standard deviation of these maps (baseline model).

    Detection:
      - For a new image, compute its local std map.
      - Calculate how many standard deviations each pixel deviates from normal.
      - Pixels with z-score above a threshold are flagged as anomalous.

    This is a simple, interpretable baseline. It does NOT require GPU or
    any pretrained neural network.
    """

    def __init__(self, window_size: int = 15, z_threshold: float = 2.5):
        """
        Parameters
        ----------
        window_size : int
            Size of the local window for computing texture std maps.
        z_threshold : float
            Pixels with z-score above this are flagged as suspicious.
            Lower → more sensitive (more detections, more false positives).
            Higher → less sensitive (fewer detections, fewer false positives).
        """
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.mean_map = None  # Average local-std map from normal images
        self.std_map = None   # Std of local-std maps from normal images
        self._normal_maps = []
        self.is_trained = False

    def fit(self, normal_images: list) -> None:
        """
        Train the detector on normal (defect-free) images.

        Parameters
        ----------
        normal_images : list of np.ndarray
            List of normal RGB images (all should be the same size).
        """
        self._normal_maps = []

        for img in normal_images:
            std_m = local_std_map(img, window_size=self.window_size)
            self._normal_maps.append(std_m)

        # Stack into a 3D array: (N, H, W)
        stack = np.stack(self._normal_maps, axis=0)
        self.mean_map = stack.mean(axis=0)
        self.std_map = stack.std(axis=0) + 1e-6  # Avoid division by zero
        self.is_trained = True

    def predict(self, image: np.ndarray, min_defect_area: int = 100) -> dict:
        """
        Detect anomalous regions in a test image.

        Parameters
        ----------
        image : np.ndarray
            RGB test image (must be same size as training images).
        min_defect_area : int
            Remove regions smaller than this from the output mask.

        Returns
        -------
        dict
            {
              "anomaly_score_map":  float32 map of z-scores per pixel
              "clean_mask":         binary mask of suspicious regions
              "regions":            list of region dicts
              "annotated":          original image with overlaid boxes
              "defect_count":       number of suspicious regions
              "flagged_pct":        % of image area flagged
              "verdict":            human-readable verdict
            }
        """
        if not self.is_trained:
            raise RuntimeError("Detector not trained. Call fit() with normal images first.")

        # Compute texture std map for test image
        test_std = local_std_map(image, window_size=self.window_size)

        # Z-score: how many standard deviations above normal?
        z_score = (test_std - self.mean_map) / self.std_map

        # Threshold: flag pixels with high positive z-score
        raw_mask = (z_score > self.z_threshold).astype(np.uint8) * 255

        # Morphological cleanup
        mask = clean_mask(raw_mask, open_kernel=3, close_kernel=7, min_area=min_defect_area)

        regions = analyze_regions(mask)
        annotated = draw_annotations(image, regions)

        total_pixels = mask.size
        flagged_pct = round(100.0 * np.count_nonzero(mask) / total_pixels, 2)
        defect_count = len(regions)

        if defect_count == 0:
            verdict = "No suspicious region detected"
        elif flagged_pct < 0.5:
            verdict = "Inspection inconclusive — minor anomaly signals"
        else:
            verdict = "Potential surface anomaly detected"

        return {
            "anomaly_score_map": z_score,
            "clean_mask": mask,
            "regions": regions,
            "annotated": annotated,
            "defect_count": defect_count,
            "flagged_pct": flagged_pct,
            "verdict": verdict,
        }

    def fit_from_directory(self, directory: str, target_size=(256, 256)) -> int:
        """
        Convenience method: load all images from a directory and fit.

        Parameters
        ----------
        directory : str
            Path to a folder of normal images (e.g., MVTec's train/good/).
        target_size : tuple
            Resize all images to this size before training (for consistency).

        Returns
        -------
        int
            Number of images loaded.
        """
        import cv2
        from pathlib import Path

        supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
        images = []

        for p in sorted(Path(directory).iterdir()):
            if p.suffix.lower() in supported:
                bgr = cv2.imread(str(p))
                if bgr is not None:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    rgb = cv2.resize(rgb, target_size)
                    images.append(rgb)

        if not images:
            raise ValueError(f"No valid images found in: {directory}")

        self.fit(images)
        return len(images)


# ─────────────────────────────────────────────────────────────
# STAGE 8: REGION ANALYSIS
# ─────────────────────────────────────────────────────────────

def analyze_regions(mask: np.ndarray) -> list:
    """
    Analyze connected components in a binary mask and return measurements.

    Each suspicious region is characterized by:
    - Bounding box (x, y, width, height) in pixels
    - Area (pixel count)
    - Centroid (approximate center)
    - Perimeter (via contour arc length)

    All measurements are pixel-based. Converting to physical units
    (e.g., millimeters) would require camera calibration data.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W) uint8 — 255 = suspicious, 0 = normal.

    Returns
    -------
    list of dict
        One dict per connected region.
        Keys: label, x, y, w, h, area, centroid_x, centroid_y, perimeter
    """
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    regions = []
    for i in range(1, n_labels):  # Skip label 0 (background)
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        cx = float(centroids[i, 0])
        cy = float(centroids[i, 1])

        # Compute perimeter via contour
        region_mask = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        perimeter = float(cv2.arcLength(contours[0], closed=True)) if contours else 0.0

        regions.append({
            "label": i,
            "x": x, "y": y, "w": w, "h": h,
            "area": area,
            "centroid_x": round(cx, 1),
            "centroid_y": round(cy, 1),
            "perimeter": round(perimeter, 1),
        })

    # Sort by area descending (largest suspicious region first)
    regions.sort(key=lambda r: r["area"], reverse=True)
    return regions


def draw_annotations(
    image: np.ndarray,
    regions: list,
    box_color: tuple = (255, 80, 80),
    text_color: tuple = (255, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on a copy of the original image.

    Parameters
    ----------
    image : np.ndarray
        Original RGB image.
    regions : list
        Output of analyze_regions().
    box_color : tuple
        RGB color for bounding boxes. Default: red.
    text_color : tuple
        RGB color for text labels.
    thickness : int
        Line thickness for bounding boxes.

    Returns
    -------
    np.ndarray
        Copy of the image with annotations drawn (RGB).
    """
    annotated = image.copy()
    for r in regions:
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        label = f"#{r['label']}  {r['area']}px"

        # Draw bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), box_color, thickness)

        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x, y - th - 6), (x + tw + 4, y), box_color, -1)

        # Draw text
        cv2.putText(
            annotated, label, (x + 2, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 1, cv2.LINE_AA
        )

    return annotated


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (255, 80, 80),
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Blend a colored defect mask onto the original image.

    Parameters
    ----------
    image : np.ndarray
        Original RGB image.
    mask : np.ndarray
        Binary mask (H, W) uint8. 255 = defect region.
    color : tuple
        RGB color to fill defect regions with.
    alpha : float
        Transparency of the overlay (0 = transparent, 1 = opaque).

    Returns
    -------
    np.ndarray
        Composite image (RGB) with defect regions highlighted.
    """
    overlay = image.copy().astype(np.float32)
    colored_layer = np.zeros_like(image, dtype=np.float32)
    colored_layer[mask > 0] = color

    # Alpha blend
    overlay[mask > 0] = (
        alpha * colored_layer[mask > 0] + (1 - alpha) * overlay[mask > 0]
    )
    return overlay.astype(np.uint8)
