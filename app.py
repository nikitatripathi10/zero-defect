"""
app.py  —  ZeroDefect Streamlit Web Interface
==============================================
Run with:  streamlit run app.py
"""

import io
import sys
import os
import time
import warnings
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

warnings.filterwarnings("ignore")

# ── Add src to path so imports resolve ──────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import load_image_from_bytes, preprocess, get_image_info
from src.edge_detection import canny_edges, auto_canny, sobel_gradient
from src.segmentation import otsu_threshold, adaptive_threshold, clean_mask
from src.texture_features import (
    compute_glcm_features, compute_statistics, local_std_map, compute_lbp, lbp_histogram
)
from src.defect_detection import (
    detect_defects_classical, create_overlay, analyze_regions, draw_annotations,
    TextureAnomalyDetector
)
from src.evaluation import pixel_metrics, image_level_metrics
from src.visualization import (
    side_by_side, plot_preprocessing_pipeline, plot_defect_pipeline,
    plot_anomaly_result, plot_region_summary, plot_confusion_matrix,
    plot_metrics_bar, fig_to_numpy, save_figure
)

# ── Page configuration ──────────────────────────────────────
st.set_page_config(
    page_title="ZeroDefect — Industrial Surface Inspection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "ZeroDefect — Intelligent Industrial Surface Defect Detection System\n"
                 "Academic prototype. Results require human verification."
    }
)

# ── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
/* ── Root palette ── */
:root {
  --bg-deep:    #080c14;
  --bg-card:    #0f1726;
  --bg-panel:   #141e33;
  --accent:     #3a86ff;
  --accent2:    #ff006e;
  --accent3:    #fb8500;
  --success:    #06d6a0;
  --warning:    #ffd166;
  --danger:     #ef233c;
  --text-main:  #e8eaf0;
  --text-muted: #8b92a5;
  --border:     rgba(58, 134, 255, 0.18);
}

/* ── Base ── */
.stApp {
    background: var(--bg-deep) !important;
    color: var(--text-main) !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text-main) !important; }

/* ── Main header ── */
.hero-title {
    text-align: center;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #3a86ff 0%, #8338ec 50%, #ff006e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
    letter-spacing: -0.5px;
}
.hero-sub {
    text-align: center;
    color: var(--text-muted);
    font-size: 1.05rem;
    margin-bottom: 1.2rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(58, 134, 255, 0.15);
    border: 1px solid var(--accent);
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.78rem;
    color: var(--accent);
    letter-spacing: 0.5px;
}

/* ── Cards ── */
.zd-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
}
.zd-metric-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.zd-metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent);
}
.zd-metric-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* ── Verdict boxes ── */
.verdict-ok {
    background: linear-gradient(135deg, rgba(6,214,160,0.12), rgba(6,214,160,0.04));
    border: 1.5px solid var(--success);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    color: var(--success);
    font-size: 1.15rem;
    font-weight: 700;
    text-align: center;
}
.verdict-warn {
    background: linear-gradient(135deg, rgba(255,209,102,0.12), rgba(255,209,102,0.04));
    border: 1.5px solid var(--warning);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    color: var(--warning);
    font-size: 1.15rem;
    font-weight: 700;
    text-align: center;
}
.verdict-bad {
    background: linear-gradient(135deg, rgba(239,35,60,0.12), rgba(239,35,60,0.04));
    border: 1.5px solid var(--danger);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    color: var(--danger);
    font-size: 1.15rem;
    font-weight: 700;
    text-align: center;
}

/* ── Section headers ── */
.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--accent);
    border-left: 4px solid var(--accent);
    padding-left: 0.7rem;
    margin: 1.4rem 0 0.8rem 0;
}

/* ── Info notice ── */
.notice {
    background: rgba(58,134,255,0.08);
    border-left: 3px solid var(--accent);
    border-radius: 0 8px 8px 0;
    padding: 0.7rem 1rem;
    font-size: 0.88rem;
    color: var(--text-muted);
    margin: 0.8rem 0;
}

/* ── Tabs ── */
[data-baseweb="tab"] {
    color: var(--text-muted) !important;
    font-weight: 600;
}
[aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #3a86ff, #8338ec) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 0.5rem 1.6rem !important;
    letter-spacing: 0.3px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(58,134,255,0.4);
}

/* ── Images ── */
[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid var(--border);
}

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════
# HELPER UTILITIES
# ════════════════════════════════════════════════

def img_to_bytes(img: np.ndarray, fmt: str = "PNG") -> bytes:
    """Convert a NumPy RGB image to downloadable bytes."""
    pil = Image.fromarray(img.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format=fmt)
    return buf.getvalue()


def mask_to_bytes(mask: np.ndarray) -> bytes:
    pil = Image.fromarray(mask.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def show_verdict(verdict: str):
    if "No suspicious" in verdict:
        st.markdown(f'<div class="verdict-ok">✅ {verdict}</div>', unsafe_allow_html=True)
    elif "inconclusive" in verdict.lower():
        st.markdown(f'<div class="verdict-warn">⚠️ {verdict}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verdict-bad">🚨 {verdict}</div>', unsafe_allow_html=True)


def metric_card(label: str, value: str, col):
    with col:
        st.markdown(
            f'<div class="zd-metric-card">'
            f'<div class="zd-metric-value">{value}</div>'
            f'<div class="zd-metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0 0.5rem;">
            <span style="font-size:2.4rem;">🔬</span>
            <div style="font-size:1.2rem; font-weight:800; color:#3a86ff; margin-top:4px;">
                ZeroDefect
            </div>
            <div style="font-size:0.75rem; color:#8b92a5; margin-top:2px;">
                Surface Inspection System
            </div>
        </div>
        <hr/>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            ["🏠 Overview", "🔍 Inspect Image", "📊 Visual Analysis", "📈 Metrics & Evaluation", "📤 Export"],
            label_visibility="collapsed",
        )

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown('<div class="notice">⚠️ Academic prototype. Results require human verification and are not suitable for industrial deployment.</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("""
        <div style="font-size:0.75rem; color:#8b92a5; text-align:center;">
            ZeroDefect v1.0 · Academic Use<br/>
            Dataset: MVTec AD (non-commercial)
        </div>
        """, unsafe_allow_html=True)

    return page


# ════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════

def page_overview():
    st.markdown('<div class="hero-title">ZeroDefect</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Intelligent Industrial Surface Defect Detection & Localization System</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div style="text-align:center; margin-bottom:2rem;">'
        '<span class="hero-badge">ACADEMIC PROTOTYPE</span>&nbsp;&nbsp;'
        '<span class="hero-badge">COMPUTER VISION</span>&nbsp;&nbsp;'
        '<span class="hero-badge">IMAGE PROCESSING</span>'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown('<div class="zd-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">What is ZeroDefect?</div>', unsafe_allow_html=True)
        st.markdown("""
ZeroDefect is an academic computer vision system that analyses images of manufactured product 
surfaces to identify potentially defective regions such as scratches, cracks, dents, surface 
contamination, and unusual textures.

The system combines **classical digital image processing** with **texture feature analysis** to:
- Detect and highlight suspicious surface regions
- Provide quantitative measurements of flagged areas
- Generate a visual defect-localisation overlay
- Compare multiple detection approaches
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="zd-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Inspection Pipeline</div>', unsafe_allow_html=True)

        steps = [
            ("1️⃣", "Image Acquisition", "Load, validate, and display the original image with metadata."),
            ("2️⃣", "Preprocessing", "Grayscale conversion → Gaussian/Median blur → CLAHE enhancement."),
            ("3️⃣", "Edge Detection", "Canny algorithm detects structural boundaries and irregularities."),
            ("4️⃣", "Thresholding", "Otsu's and adaptive methods create binary segmentation masks."),
            ("5️⃣", "Morphological Processing", "Erosion/dilation clean masks and connect defect fragments."),
            ("6️⃣", "Texture Analysis", "GLCM features and local std-dev maps score anomaly severity."),
            ("7️⃣", "Defect Localisation", "Suspicious regions are identified, measured, and labelled."),
            ("8️⃣", "Evaluation", "Pixel-level F1, IoU, AUROC are computed against ground-truth masks."),
        ]
        for icon, name, desc in steps:
            st.markdown(f"""
            <div style="display:flex; gap:0.8rem; margin-bottom:0.6rem; align-items:flex-start;">
                <span style="font-size:1.1rem;">{icon}</span>
                <div>
                    <strong style="color:#3a86ff;">{name}</strong><br/>
                    <span style="color:#8b92a5; font-size:0.88rem;">{desc}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="zd-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Detectable Anomalies</div>', unsafe_allow_html=True)
        anomalies = ["🔴 Scratches", "🟠 Cracks", "🟡 Dents / irregularities",
                     "🟢 Surface contamination", "🔵 Unusual textures",
                     "🟣 Colour anomalies", "⚪ Foreign objects"]
        for a in anomalies:
            st.markdown(f"- {a}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="zd-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Technology Stack</div>', unsafe_allow_html=True)
        techs = {
            "Python 3": "Core language",
            "OpenCV": "Image processing primitives",
            "scikit-image": "GLCM, LBP texture features",
            "NumPy": "Pixel-level array operations",
            "scikit-learn": "Evaluation metrics, AUROC",
            "Streamlit": "This web interface",
            "Matplotlib / Plotly": "Visualisations",
        }
        for tech, desc in techs.items():
            st.markdown(f"**{tech}** — {desc}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="zd-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Dataset</div>', unsafe_allow_html=True)
        st.markdown("""
**MVTec Anomaly Detection (MVTec AD)**  
[mvtec.com/research-teaching/datasets/mvtec-ad](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)

- 15 industrial object/texture categories
- Provides normal training images + defective test images
- Includes pixel-level ground-truth masks
- Licensed for non-commercial academic use

*This project uses one texture category (e.g., leather, wood, carpet).*
        """)
        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════
# PAGE: INSPECT IMAGE
# ════════════════════════════════════════════════

def page_inspect():
    st.markdown('<div class="hero-title" style="font-size:1.8rem;">🔍 Image Inspection</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Upload an image and run the defect detection pipeline</div>', unsafe_allow_html=True)

    # ── Settings sidebar-within-page ──────────────
    with st.expander("⚙️ Detection Settings", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Preprocessing**")
            blur_method = st.selectbox("Blur method", ["gaussian", "median"], key="blur_method")
            blur_kernel = st.slider("Blur kernel size", 3, 15, 5, step=2, key="blur_k")
            use_clahe = st.checkbox("Apply CLAHE", value=True, key="clahe")
            clahe_clip = st.slider("CLAHE clip limit", 1.0, 5.0, 2.0, 0.5, key="clahe_clip",
                                   disabled=not use_clahe)
        with c2:
            st.markdown("**Detection**")
            detection_method = st.selectbox("Primary method", ["Classical", "Texture Anomaly"], key="det_method")
            edge_method = st.selectbox("Edge method", ["auto_canny", "canny", "sobel_threshold"], key="edge_m")
            if edge_method == "canny":
                low_t = st.slider("Canny low threshold", 10, 150, 50, key="low_t")
                high_t = st.slider("Canny high threshold", 50, 300, 150, key="high_t")
            else:
                low_t, high_t = 50, 150
        with c3:
            st.markdown("**Post-processing**")
            open_k = st.slider("Opening kernel", 3, 15, 3, step=2, key="open_k")
            close_k = st.slider("Closing kernel", 3, 21, 7, step=2, key="close_k")
            min_area = st.slider("Min region area (px)", 50, 1000, 150, step=50, key="min_area")

    # ── Image source ──────────────────────────────
    st.markdown('<div class="section-header">Image Source</div>', unsafe_allow_html=True)

    source_tab = st.radio(
        "Choose source",
        ["📁 Upload image", "🧪 Use synthetic demo"],
        horizontal=True,
        key="img_src",
        label_visibility="collapsed",
    )

    image = None

    if source_tab == "📁 Upload image":
        uploaded = st.file_uploader(
            "Upload a surface image",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            key="uploader",
        )
        if uploaded:
            try:
                image = load_image_from_bytes(uploaded.read())
                st.success(f"✅ Loaded: **{uploaded.name}** ({image.shape[1]}×{image.shape[0]} px)")
            except Exception as e:
                st.error(f"❌ Could not load image: {e}")

    else:  # Synthetic demo
        demo_type = st.selectbox(
            "Choose demo surface",
            [
                "Machined metal (defect-free)",
                "Machined metal (scratch defect)",
                "Machined metal (pitting corrosion)",
                "Textured grid (defect-free)",
                "Textured grid (scratch defect)",
                "Textured grid (crack cluster)",
            ],
            key="demo_type",
        )
        image = _make_demo_image(demo_type)
        st.info("🧪 Synthetic surface generated. You can also upload your own images.")

    # ── Run analysis ──────────────────────────────
    if image is not None:
        st.markdown('<div class="section-header">Original Image</div>', unsafe_allow_html=True)

        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.image(image, use_container_width=True, caption="Input image (RGB)")
        with col_info:
            info = get_image_info(image)
            st.markdown('<div class="zd-card">', unsafe_allow_html=True)
            st.markdown("**Image Metadata**")
            for k, v in info.items():
                st.markdown(f"- **{k.replace('_', ' ').title()}**: `{v}`")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        if st.button("▶️ Run Defect Inspection", use_container_width=True, key="run_btn"):
            with st.spinner("🔄 Running inspection pipeline…"):

                preprocess_cfg = {
                    "blur_method": blur_method,
                    "blur_kernel": blur_kernel,
                    "use_clahe": use_clahe,
                    "clahe_clip": clahe_clip,
                }

                target_size = (512, 512)
                image_resized = cv2.resize(image, target_size)

                try:
                    if detection_method == "Classical":
                        result = detect_defects_classical(
                            image_resized,
                            preprocess_config=preprocess_cfg,
                            edge_method=edge_method,
                            low_thresh=low_t,
                            high_thresh=high_t,
                            open_kernel=open_k,
                            close_kernel=close_k,
                            min_defect_area=min_area,
                        )
                    else:
                        # Texture anomaly: use the single image as both reference and test
                        # (for demo purposes — real use requires separate normal training images)
                        st.warning(
                            "⚠️ Texture Anomaly mode requires training on normal images. "
                            "In demo mode, the image is self-compared — results are indicative only."
                        )
                        detector = TextureAnomalyDetector(z_threshold=2.0)
                        # Use blurred version as pseudo-normal reference
                        prep = preprocess(image_resized, **preprocess_cfg)
                        blurred_rgb = cv2.cvtColor(prep["blurred"], cv2.COLOR_GRAY2RGB)
                        detector.fit([blurred_rgb] * 5)
                        result = detector.predict(image_resized, min_defect_area=min_area)
                        # Wrap to match classical output format for display
                        result["preprocessed"] = preprocess(image_resized, **preprocess_cfg)
                        result["edge_map"] = auto_canny(prep["enhanced"])
                        result["raw_mask"] = result["clean_mask"]

                    st.session_state["result"] = result
                    st.session_state["original"] = image_resized
                    st.session_state["detection_method"] = detection_method
                    st.success("✅ Inspection complete!")

                except Exception as e:
                    st.error(f"❌ Detection failed: {e}")
                    st.exception(e)
                    return

        # ── Display results (if available) ─────────
        if "result" in st.session_state and "original" in st.session_state:
            result = st.session_state["result"]
            orig = st.session_state["original"]

            st.markdown('<div class="section-header">Inspection Result</div>', unsafe_allow_html=True)

            show_verdict(result["verdict"])
            st.markdown("<br/>", unsafe_allow_html=True)

            mc1, mc2, mc3, mc4 = st.columns(4)
            metric_card("Regions Found", str(result["defect_count"]), mc1)
            metric_card("Area Flagged", f"{result['flagged_pct']}%", mc2)
            largest = result["regions"][0]["area"] if result["regions"] else 0
            metric_card("Largest Region", f"{largest} px", mc3)
            metric_card("Detection Method", st.session_state.get("detection_method", "—"), mc4)

            st.markdown("<br/>", unsafe_allow_html=True)

            # Show overlay
            overlay = create_overlay(orig, result["clean_mask"])
            annotated = result["annotated"]

            col_ov, col_an = st.columns(2)
            with col_ov:
                st.image(overlay, caption="🔴 Defect Overlay", use_container_width=True)
            with col_an:
                st.image(annotated, caption="📦 Bounding Box Annotations", use_container_width=True)

            # Region table
            if result["regions"]:
                st.markdown('<div class="section-header">Detected Regions (pixel estimates)</div>', unsafe_allow_html=True)
                df = pd.DataFrame(result["regions"]).drop(columns=["label"], errors="ignore")
                df.index = [f"Region {i+1}" for i in range(len(df))]
                st.dataframe(df.style.background_gradient(subset=["area"], cmap="Reds"), use_container_width=True)

            st.session_state["overlay"] = overlay
            st.session_state["annotated"] = annotated


# ════════════════════════════════════════════════
# PAGE: VISUAL ANALYSIS
# ════════════════════════════════════════════════

def page_visual_analysis():
    st.markdown('<div class="hero-title" style="font-size:1.8rem;">📊 Visual Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Step-by-step pipeline visualisation</div>', unsafe_allow_html=True)

    if "result" not in st.session_state:
        st.info("👆 Please run an inspection first in the **🔍 Inspect Image** tab.")
        return

    result = st.session_state["result"]
    orig = st.session_state["original"]
    prep = result["preprocessed"]

    tabs = st.tabs(["Preprocessing", "Edge & Threshold", "Morphology & Mask", "Texture Map", "Final Overlay"])

    # ── Tab 1: Preprocessing ─────────────────────
    with tabs[0]:
        st.markdown("### Preprocessing Pipeline")
        st.markdown("""
        Each stage prepares the image for more accurate defect detection:
        - **Grayscale** removes colour information (most defects are intensity-based)
        - **Blur** suppresses random noise so edges and defects stand out more clearly
        - **CLAHE** boosts local contrast, making subtle surface variations visible
        """)
        cols = st.columns(4)
        for col, (key, title) in zip(cols, [
            ("original", "Original"), ("gray", "Grayscale"),
            ("blurred", "Blurred"), ("enhanced", "CLAHE Enhanced")
        ]):
            if key in prep:
                with col:
                    st.image(prep[key], caption=title, use_container_width=True)

    # ── Tab 2: Edge & Threshold ───────────────────
    with tabs[1]:
        st.markdown("### Edge Detection & Thresholding")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(result["edge_map"], caption="Canny Edge Map", use_container_width=True)
            st.caption("ℹ️ Edges alone ≠ defects. Normal textures also produce edges.")
        with c2:
            _, otsu_mask = otsu_threshold(prep["enhanced"])
            st.image(otsu_mask, caption="Otsu Threshold Mask", use_container_width=True)
        with c3:
            adapt_mask = adaptive_threshold(prep["enhanced"], block_size=31, C=5)
            st.image(adapt_mask, caption="Adaptive Threshold Mask", use_container_width=True)

        st.markdown("""
        **Comparing thresholding methods:**
        - **Otsu**: Picks the single best global threshold automatically. Best when image has two clear intensity clusters.
        - **Adaptive**: Calculates local thresholds for small tiles. Better handles uneven lighting.
        """)

    # ── Tab 3: Morphology & Mask ──────────────────
    with tabs[2]:
        st.markdown("### Morphological Mask Cleaning")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(result["raw_mask"], caption="Raw Combined Mask (before cleanup)", use_container_width=True)
        with c2:
            st.image(result["clean_mask"], caption="Cleaned Mask (after morphology)", use_container_width=True)
        with c3:
            st.image(result["annotated"], caption="Annotated (bounding boxes)", use_container_width=True)

        fig = plot_region_summary(result["regions"], orig.shape[:2])
        st.pyplot(fig, use_container_width=True)

    # ── Tab 4: Texture Map ────────────────────────
    with tabs[3]:
        st.markdown("### Texture Analysis")
        enhanced = prep.get("enhanced", prep.get("gray"))

        with st.spinner("Computing texture features…"):
            std_map = local_std_map(enhanced, window_size=15)
            glcm = compute_glcm_features(enhanced)
            stats = compute_statistics(enhanced)

        c1, c2 = st.columns(2)
        with c1:
            st.image(enhanced, caption="Enhanced Grayscale", use_container_width=True)
        with c2:
            # Normalize std map to [0,255] for display
            std_norm = cv2.normalize(std_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            st.image(std_norm, caption="Local Std-Dev Map (anomaly signal)", use_container_width=True)

        st.markdown("**GLCM Texture Features:**")
        gcols = st.columns(4)
        for col, (k, v) in zip(gcols, glcm.items()):
            col.metric(k.title(), f"{v:.4f}")

        st.markdown("**Intensity Statistics:**")
        scols = st.columns(4)
        stat_items = list(stats.items())
        for i, col in enumerate(scols):
            for k, v in stat_items[i*2:(i+1)*2]:
                col.metric(k.title(), f"{v:.2f}")

    # ── Tab 5: Final Overlay ──────────────────────
    with tabs[4]:
        st.markdown("### Final Detection Overlay")
        if "overlay" in st.session_state:
            st.image(st.session_state["overlay"], caption="Defect overlay — red = suspicious region", use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.image(orig, caption="Original", use_container_width=True)
        with c2:
            st.image(result["clean_mask"], caption="Predicted Defect Mask", use_container_width=True)


# ════════════════════════════════════════════════
# PAGE: METRICS & EVALUATION
# ════════════════════════════════════════════════

def page_metrics():
    st.markdown('<div class="hero-title" style="font-size:1.8rem;">📈 Metrics & Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Quantitative evaluation against ground-truth masks</div>', unsafe_allow_html=True)

    if "result" not in st.session_state:
        st.info("👆 Please run an inspection first in the **🔍 Inspect Image** tab.")
        return

    result = st.session_state["result"]
    orig = st.session_state["original"]
    pred_mask = result["clean_mask"]

    # ── Ground truth upload ───────────────────────
    st.markdown('<div class="section-header">Ground-Truth Mask (optional)</div>', unsafe_allow_html=True)
    st.markdown("""
    Upload a binary ground-truth mask from the MVTec AD dataset (or any other reference mask) 
    to compute **pixel-level evaluation metrics**. The mask should be a grayscale image where 
    white (255) indicates defective pixels and black (0) indicates normal pixels.
    """)

    gt_file = st.file_uploader("Upload ground-truth mask (PNG)", type=["png", "bmp", "tiff"], key="gt_upload")

    gt_mask = None
    if gt_file:
        try:
            gt_arr = np.frombuffer(gt_file.read(), np.uint8)
            gt_bgr = cv2.imdecode(gt_arr, cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.resize(gt_bgr, (pred_mask.shape[1], pred_mask.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
            st.success("✅ Ground-truth mask loaded.")
            c1, c2 = st.columns(2)
            with c1:
                st.image(pred_mask, caption="Predicted Mask", use_container_width=True)
            with c2:
                st.image(gt_mask, caption="Ground-Truth Mask", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Could not load mask: {e}")

    # ── Compute metrics ────────────────────────────
    if gt_mask is not None:
        st.markdown('<div class="section-header">Pixel-Level Metrics</div>', unsafe_allow_html=True)
        metrics = pixel_metrics(pred_mask, gt_mask)

        mc = st.columns(4)
        metric_card("Precision", f"{metrics['precision']:.3f}", mc[0])
        metric_card("Recall", f"{metrics['recall']:.3f}", mc[1])
        metric_card("F1-Score", f"{metrics['f1']:.3f}", mc[2])
        metric_card("IoU", f"{metrics['iou']:.3f}", mc[3])

        st.markdown("<br/>", unsafe_allow_html=True)

        fig = plot_metrics_bar(
            {k: v for k, v in metrics.items() if k in ["precision", "recall", "f1", "iou"]},
            title="Pixel-Level Evaluation Metrics"
        )
        st.pyplot(fig, use_container_width=True)

        st.markdown('<div class="section-header">Confusion Matrix (pixel-level)</div>', unsafe_allow_html=True)
        tp, fp, fn, tn = metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]
        cm = np.array([[tn, fp], [fn, tp]])
        fig_cm = plot_confusion_matrix(cm)
        st.pyplot(fig_cm, use_container_width=True)

        # Store for export
        st.session_state["eval_metrics"] = metrics

    else:
        # Show estimated metrics without ground truth
        st.markdown('<div class="section-header">Inspection Summary (no ground truth)</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="notice">
        No ground-truth mask uploaded. Upload a reference mask to compute F1, IoU, precision, and recall.
        The values below are descriptive statistics about the pipeline's output — not evaluation metrics.
        </div>
        """, unsafe_allow_html=True)

        total_px = pred_mask.size
        flagged_px = int(np.count_nonzero(pred_mask))
        st.markdown(f"""
        | Metric | Value |
        |---|---|
        | Flagged pixels | {flagged_px:,} |
        | Normal pixels | {total_px - flagged_px:,} |
        | Flagged percentage | {result['flagged_pct']:.2f}% |
        | Suspicious regions | {result['defect_count']} |
        | Largest region (px) | {result['regions'][0]['area'] if result['regions'] else 0} |
        """)

    # ── Metric explanations ───────────────────────
    with st.expander("📚 What do these metrics mean?"):
        st.markdown("""
| Metric | Formula | Meaning |
|---|---|---|
| **Precision** | TP / (TP + FP) | Of all pixels the system flagged as defective, what fraction actually are? High precision = few false alarms. |
| **Recall** | TP / (TP + FN) | Of all truly defective pixels, what fraction did the system find? High recall = few missed defects. |
| **F1-Score** | 2 × Prec × Rec / (Prec + Rec) | Harmonic mean balancing precision and recall. Best single-number summary. |
| **IoU** | TP / (TP + FP + FN) | Overlap between predicted and ground-truth mask. 1.0 = perfect match, 0 = no overlap. |
| **TP** | True Positives | Defective pixels correctly detected. |
| **FP** | False Positives | Normal pixels wrongly labelled as defective (false alarms). |
| **FN** | False Negatives | Defective pixels missed by the system. |
| **TN** | True Negatives | Normal pixels correctly identified as normal. |

> **Important**: An academic prototype will rarely achieve F1 > 0.7 without fine-tuning or deep learning.
> Honest reporting of low metrics is more scientifically valuable than fabricated high scores.
        """)


# ════════════════════════════════════════════════
# PAGE: EXPORT
# ════════════════════════════════════════════════

def page_export():
    st.markdown('<div class="hero-title" style="font-size:1.8rem;">📤 Export Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Download inspection results and reports</div>', unsafe_allow_html=True)

    if "result" not in st.session_state:
        st.info("👆 Please run an inspection first in the **🔍 Inspect Image** tab.")
        return

    result = st.session_state["result"]
    orig = st.session_state.get("original")

    st.markdown('<div class="section-header">Available Downloads</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="zd-card">', unsafe_allow_html=True)

        if "overlay" in st.session_state:
            st.download_button(
                "⬇️ Download Defect Overlay (PNG)",
                data=img_to_bytes(st.session_state["overlay"]),
                file_name="zerodefect_overlay.png",
                mime="image/png",
                use_container_width=True,
            )

        st.download_button(
            "⬇️ Download Prediction Mask (PNG)",
            data=mask_to_bytes(result["clean_mask"]),
            file_name="zerodefect_mask.png",
            mime="image/png",
            use_container_width=True,
        )

        if orig is not None:
            st.download_button(
                "⬇️ Download Annotated Image (PNG)",
                data=img_to_bytes(result["annotated"]),
                file_name="zerodefect_annotated.png",
                mime="image/png",
                use_container_width=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="zd-card">', unsafe_allow_html=True)

        # CSV report
        rows = []
        for r in result["regions"]:
            rows.append({
                "region": r["label"],
                "x": r["x"], "y": r["y"],
                "width": r["w"], "height": r["h"],
                "area_px": r["area"],
                "perimeter_px": r["perimeter"],
                "centroid_x": r["centroid_x"],
                "centroid_y": r["centroid_y"],
            })

        # Summary row
        rows.append({
            "region": "SUMMARY",
            "x": "—", "y": "—", "width": "—", "height": "—",
            "area_px": sum(r["area"] for r in result["regions"]),
            "perimeter_px": "—",
            "centroid_x": "—", "centroid_y": "—",
        })

        df_export = pd.DataFrame(rows)
        csv_bytes = df_export.to_csv(index=False).encode("utf-8")

        st.download_button(
            "⬇️ Download Region Report (CSV)",
            data=csv_bytes,
            file_name="zerodefect_regions.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # Metrics CSV (if available)
        if "eval_metrics" in st.session_state:
            metrics_df = pd.DataFrame([st.session_state["eval_metrics"]])
            metrics_csv = metrics_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Evaluation Metrics (CSV)",
                data=metrics_csv,
                file_name="zerodefect_metrics.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="notice">
    ⚠️ All exported files are based on this session's analysis only. 
    Results are image-based pixel estimates and are not calibrated to physical dimensions.
    This system is an academic prototype — results require human verification.
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════
# DEMO IMAGE GENERATOR
# ════════════════════════════════════════════════

def _make_demo_image(demo_type: str) -> np.ndarray:
    """Generate synthetic surface images for demonstration."""
    np.random.seed(42)
    H, W = 512, 512

    if demo_type == "Machined metal (defect-free)":
        metal = np.ones((H, W), dtype=np.float32) * 180.0
        for r in range(H):
            metal[r, :] += np.random.normal(0, 3) + np.sin(r * 0.1) * 2
        metal += np.random.normal(0, 4, (H, W))
        metal = np.clip(metal, 0, 255).astype(np.uint8)
        return cv2.cvtColor(metal, cv2.COLOR_GRAY2RGB)

    elif demo_type == "Machined metal (scratch defect)":
        metal = np.ones((H, W), dtype=np.float32) * 180.0
        for r in range(H):
            metal[r, :] += np.random.normal(0, 3) + np.sin(r * 0.1) * 2
        metal += np.random.normal(0, 4, (H, W))
        metal = np.clip(metal, 0, 255).astype(np.uint8)
        for k in range(-3, 4):
            for x in range(60, 460):
                y = int(0.45 * x + 70) + k
                if 0 <= y < H:
                    metal[y, x] = max(0, int(metal[y, x]) - 110)
        return cv2.cvtColor(metal, cv2.COLOR_GRAY2RGB)

    elif demo_type == "Machined metal (pitting corrosion)":
        metal = np.ones((H, W), dtype=np.float32) * 180.0
        for r in range(H):
            metal[r, :] += np.random.normal(0, 3) + np.sin(r * 0.1) * 2
        metal += np.random.normal(0, 4, (H, W))
        metal = np.clip(metal, 0, 255).astype(np.uint8)
        for cy, cx, rad in [(200, 250, 24), (320, 180, 18), (280, 380, 15)]:
            y, x = np.ogrid[:H, :W]
            mask = (x - cx)**2 + (y - cy)**2 <= rad**2
            metal[mask] = np.clip(metal[mask].astype(int) - 95, 0, 255).astype(np.uint8)
        return cv2.cvtColor(metal, cv2.COLOR_GRAY2RGB)

    elif demo_type in ("Textured grid (defect-free)", "Normal surface (grid pattern)"):
        # Regular grid — no defects
        img = np.ones((H, W), dtype=np.uint8) * 180
        for i in range(0, H, 32):
            img[i:i+2, :] = 120
        for j in range(0, W, 32):
            img[:, j:j+2] = 120
        noise = np.random.normal(0, 5, (H, W)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    elif demo_type in ("Textured grid (scratch defect)", "Surface with scratch"):
        # Grid + a long diagonal scratch
        img = np.ones((H, W), dtype=np.uint8) * 180
        for i in range(0, H, 32):
            img[i:i+2, :] = 120
        for j in range(0, W, 32):
            img[:, j:j+2] = 120
        # Scratch: a thin dark diagonal stripe
        for k in range(-4, 5):
            mask = np.zeros((H, W), dtype=np.uint8)
            for x in range(W):
                y = int(0.5 * x + 80) + k
                if 0 <= y < H:
                    mask[y, x] = 1
            img[mask == 1] = np.clip(img[mask == 1].astype(int) - 80, 0, 255)
        noise = np.random.normal(0, 5, (H, W)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    else:  # Crack cluster
        img = np.ones((H, W), dtype=np.uint8) * 175
        for i in range(0, H, 32):
            img[i:i+2, :] = 115
        for j in range(0, W, 32):
            img[:, j:j+2] = 115
        # Multiple irregular dark blobs
        for _ in range(6):
            cx, cy = np.random.randint(80, 430), np.random.randint(80, 430)
            for angle in np.linspace(0, np.pi, 12):
                length = np.random.randint(20, 60)
                for t in range(length):
                    x = int(cx + t * np.cos(angle + np.random.uniform(-0.3, 0.3)))
                    y = int(cy + t * np.sin(angle + np.random.uniform(-0.3, 0.3)))
                    if 0 <= x < W and 0 <= y < H:
                        for dx in range(-2, 3):
                            for dy in range(-2, 3):
                                if 0 <= x+dx < W and 0 <= y+dy < H:
                                    img[y+dy, x+dx] = max(0, int(img[y+dy, x+dx]) - 70)
        noise = np.random.normal(0, 6, (H, W)).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


# ════════════════════════════════════════════════
# MAIN ROUTER
# ════════════════════════════════════════════════

def main():
    page = render_sidebar()

    if page == "🏠 Overview":
        page_overview()
    elif page == "🔍 Inspect Image":
        page_inspect()
    elif page == "📊 Visual Analysis":
        page_visual_analysis()
    elif page == "📈 Metrics & Evaluation":
        page_metrics()
    elif page == "📤 Export":
        page_export()


if __name__ == "__main__":
    main()
