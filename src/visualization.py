"""
visualization.py
================
Visualization utilities for ZeroDefect.

All visualizations return Matplotlib figures or NumPy arrays
so they can be embedded in both the Streamlit interface and notebooks.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — safe for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


# ── Custom colormap for anomaly score maps ──────────────────
# Low score → blue/green, high score → yellow/red (like a heat map)
_ANOMALY_CMAP = LinearSegmentedColormap.from_list(
    "anomaly",
    ["#0d1b2a", "#1a6b8a", "#f7b731", "#ee5a24"],
    N=256,
)


def side_by_side(images: list, titles: list, cmap_list: list = None, figsize=None) -> plt.Figure:
    """
    Display multiple images side by side in a single Matplotlib figure.

    Parameters
    ----------
    images : list of np.ndarray
        Images to display (RGB or grayscale).
    titles : list of str
        Title for each image.
    cmap_list : list of str or None
        Colormap per image. None → auto-select (gray for 2D, None for RGB).
    figsize : tuple or None
        Figure size. Auto-scaled if None.

    Returns
    -------
    plt.Figure
    """
    n = len(images)
    if figsize is None:
        figsize = (4.5 * n, 4.5)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        cmap = None
        if img.ndim == 2:
            cmap = "gray"
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.axis("off")

    plt.tight_layout()
    return fig


def plot_preprocessing_pipeline(prep_dict: dict) -> plt.Figure:
    """
    Visualize all preprocessing stages side by side.

    Parameters
    ----------
    prep_dict : dict
        Output of preprocessing.preprocess().

    Returns
    -------
    plt.Figure
    """
    stage_keys = ["original", "gray", "blurred", "enhanced"]
    stage_labels = ["Original", "Grayscale", "Blurred", "CLAHE Enhanced"]

    images, titles = [], []
    for key, label in zip(stage_keys, stage_labels):
        if key in prep_dict:
            images.append(prep_dict[key])
            titles.append(label)

    return side_by_side(images, titles, figsize=(5 * len(images), 4.5))


def plot_defect_pipeline(original, edge_map, raw_mask, clean_mask, annotated) -> plt.Figure:
    """
    Visualize the full classical detection pipeline stages.
    """
    images = [original, edge_map, raw_mask, clean_mask, annotated]
    titles = ["Original", "Edge Map", "Raw Mask", "Cleaned Mask", "Annotations"]
    return side_by_side(images, titles, figsize=(4 * 5, 4))


def plot_anomaly_result(original, score_map, clean_mask, overlay) -> plt.Figure:
    """
    Visualize texture anomaly detection results.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.patch.set_facecolor("#0f0f1a")

    plot_configs = [
        (original, "Original Image", None),
        (score_map, "Anomaly Score Map", _ANOMALY_CMAP),
        (clean_mask, "Predicted Defect Mask", "gray"),
        (overlay, "Detection Overlay", None),
    ]

    for ax, (img, title, cmap) in zip(axes, plot_configs):
        ax.set_facecolor("#0f0f1a")
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold")
        ax.axis("off")

        if cmap == _ANOMALY_CMAP:
            # Add colorbar for anomaly score
            sm = plt.cm.ScalarMappable(cmap=_ANOMALY_CMAP,
                                       norm=plt.Normalize(vmin=img.min(), vmax=img.max()))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.yaxis.set_tick_params(color="white")
            plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    plt.tight_layout()
    return fig


def plot_region_summary(regions: list, image_shape: tuple) -> plt.Figure:
    """
    Plot a bar chart summarizing detected region areas.

    Parameters
    ----------
    regions : list of dict
        Output of defect_detection.analyze_regions().
    image_shape : tuple
        (H, W) of the image for percentage calculation.

    Returns
    -------
    plt.Figure
    """
    if not regions:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No suspicious regions detected",
                ha="center", va="center", fontsize=13, color="green",
                transform=ax.transAxes)
        ax.axis("off")
        return fig

    labels = [f"Region {r['label']}" for r in regions]
    areas = [r["area"] for r in regions]
    total_px = image_shape[0] * image_shape[1]
    pcts = [100 * a / total_px for a in areas]

    fig, ax = plt.subplots(figsize=(max(6, len(regions) * 1.2), 4))
    bars = ax.bar(labels, pcts, color="#ee5a24", edgecolor="white", linewidth=0.8)
    ax.set_ylabel("Area (% of image)", fontsize=11)
    ax.set_title("Suspicious Region Sizes", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(pcts) * 1.3 + 0.1)
    ax.tick_params(axis="x", rotation=30)

    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{pct:.2f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_names=("Normal", "Defective")) -> plt.Figure:
    """
    Plot a labeled confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray
        2×2 confusion matrix from sklearn.
    class_names : tuple
        Class labels for axes.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_metrics_bar(metrics: dict, title: str = "Evaluation Metrics") -> plt.Figure:
    """
    Plot a horizontal bar chart of evaluation metrics.

    Parameters
    ----------
    metrics : dict
        Keys: metric names, values: float in [0, 1].
    title : str

    Returns
    -------
    plt.Figure
    """
    # Filter to numeric values only
    filtered = {k: v for k, v in metrics.items()
                if isinstance(v, (int, float)) and 0 <= v <= 1}

    if not filtered:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No metrics available", ha="center", va="center")
        ax.axis("off")
        return fig

    names = list(filtered.keys())
    values = list(filtered.values())
    colors = ["#27ae60" if v >= 0.7 else "#f39c12" if v >= 0.4 else "#e74c3c" for v in values]

    fig, ax = plt.subplots(figsize=(7, 1.0 * len(names) + 1.5))
    bars = ax.barh(names, values, color=colors, edgecolor="white", linewidth=0.7)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    for bar, val in zip(bars, values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)

    plt.tight_layout()
    return fig


def fig_to_numpy(fig: plt.Figure) -> np.ndarray:
    """
    Convert a Matplotlib figure to an RGB NumPy array.

    Useful for embedding figures as images in Streamlit (st.image).

    Parameters
    ----------
    fig : plt.Figure

    Returns
    -------
    np.ndarray
        (H, W, 3) RGB array.
    """
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    arr = np.frombuffer(buf.read(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    plt.close(fig)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_figure(fig: plt.Figure, path: str, dpi: int = 150) -> None:
    """
    Save a Matplotlib figure to disk.

    Parameters
    ----------
    fig : plt.Figure
    path : str
        Output file path (PNG, JPG, etc.).
    dpi : int
    """
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
