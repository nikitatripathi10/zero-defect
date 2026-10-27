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
    initial_sidebar_state="collapsed",
    menu_items={
        "About": "ZeroDefect — Intelligent Industrial Surface Defect Detection System\n"
                 "Academic prototype. Results require human verification."
    }
)

# ── Custom CSS  (Radix UI Design System + Animations) ────────
st.html("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap" rel="stylesheet">
<style>
/* ════════════════════════════════════════════════
   RADIX UI DESIGN TOKENS
   Based on Radix Colors + Radix Themes spacing
════════════════════════════════════════════════ */
:root {
  /* Radix Gray Dark */
  --gray-1:  #111113; --gray-2:  #19191b; --gray-3:  #222325;
  --gray-4:  #292a2e; --gray-5:  #303136; --gray-6:  #393a40;
  --gray-7:  #46474e; --gray-8:  #5f6169; --gray-9:  #6c6e79;
  --gray-10: #7d7f8a; --gray-11: #b2b3bc; --gray-12: #edeef0;

  /* Radix Blue Dark */
  --blue-1:  #0d1520; --blue-2:  #111927; --blue-3:  #0d2847;
  --blue-4:  #003362; --blue-5:  #004074; --blue-6:  #104d87;
  --blue-7:  #205d9e; --blue-8:  #2870bd; --blue-9:  #0090ff;
  --blue-10: #3b9eff; --blue-11: #70b8ff; --blue-12: #c2e6ff;

  /* Radix Violet Dark */
  --violet-9:  #6e56cf; --violet-10: #7d66d9; --violet-11: #baa7ff;

  /* Radix Green Dark */
  --green-9:  #46a758; --green-10: #55b467; --green-11: #71d083;

  /* Radix Red Dark */
  --red-9: #e5484d; --red-10: #f2555a; --red-11: #ff6369;

  /* Radix Amber Dark */
  --amber-9: #ffb224; --amber-10: #ffcb47; --amber-11: #f1a10d;

  /* Semantic tokens */
  --bg:           var(--gray-1);
  --surface:      var(--gray-2);
  --surface-2:    var(--gray-3);
  --panel:        var(--gray-4);
  --border:       var(--gray-6);
  --border-hover: var(--gray-7);
  --text-primary: var(--gray-12);
  --text-secondary: var(--gray-11);
  --text-muted:   var(--gray-10);
  --accent:       var(--blue-9);
  --accent-hover: var(--blue-10);
  --accent-light: var(--blue-11);
  --accent-subtle: rgba(0, 144, 255, 0.12);
  --success:      var(--green-9);
  --warning:      var(--amber-9);
  --danger:       var(--red-9);

  /* Radix radius tokens */
  --radius-1: 3px;  --radius-2: 4px;  --radius-3: 6px;
  --radius-4: 8px;  --radius-5: 12px; --radius-6: 16px;
  --radius-full: 9999px;

  /* Radix spacing (4px base) */
  --space-1: 4px;  --space-2: 8px;  --space-3: 12px;
  --space-4: 16px; --space-5: 24px; --space-6: 32px;
  --space-7: 40px; --space-8: 48px; --space-9: 64px;

  /* Shadow tokens */
  --shadow-1: 0 1px 3px rgba(0,0,0,0.4);
  --shadow-2: 0 4px 16px rgba(0,0,0,0.5);
  --shadow-3: 0 8px 32px rgba(0,0,0,0.6);
  --shadow-accent: 0 0 24px rgba(0,144,255,0.25);
}

/* ═══════ ANIMATIONS ═══════ */
@keyframes fadeInUp {
  from { opacity:0; transform: translateY(24px); }
  to   { opacity:1; transform: translateY(0); }
}
@keyframes fadeIn {
  from { opacity:0; } to { opacity:1; }
}
@keyframes slideInLeft {
  from { opacity:0; transform: translateX(-20px); }
  to   { opacity:1; transform: translateX(0); }
}
@keyframes shimmer {
  0%   { background-position: -1000px 0; }
  100% { background-position: 1000px 0; }
}
@keyframes pulse-ring {
  0%   { box-shadow: 0 0 0 0 rgba(0,144,255,0.4); }
  70%  { box-shadow: 0 0 0 10px rgba(0,144,255,0); }
  100% { box-shadow: 0 0 0 0 rgba(0,144,255,0); }
}
@keyframes float {
  0%, 100% { transform: translateY(0px); }
  50%       { transform: translateY(-6px); }
}
@keyframes gradient-shift {
  0%   { background-position: 0% 50%; }
  50%  { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
@keyframes spin-slow {
  from { transform: rotate(0deg); }
  to   { transform: rotate(360deg); }
}
@keyframes countUp {
  from { opacity:0; transform: scale(0.8); }
  to   { opacity:1; transform: scale(1); }
}

/* ═══════ GLOBAL RESET & BASE ═══════ */
* { box-sizing: border-box; }

.stApp {
    background: var(--bg) !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    font-size: 14px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
}

/* Hide Streamlit sidebar completely */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* Remove default Streamlit header spacing */
.block-container {
    padding-top: 0 !important;
    max-width: 100% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* ═══════ TOP NAVIGATION BAR ═══════ */
.zd-navbar {
    position: sticky;
    top: 0;
    z-index: 1000;
    background: rgba(17, 17, 19, 0.92);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border-bottom: 1px solid var(--border);
    padding: 0 var(--space-6);
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 56px;
    animation: fadeIn 0.3s ease;
}
.zd-navbar-brand {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    text-decoration: none;
}
.zd-navbar-logo {
    font-size: 1.4rem;
    animation: float 3s ease-in-out infinite;
    display: inline-block;
}
.zd-navbar-wordmark {
    font-size: 1.05rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--blue-9), var(--violet-9));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.3px;
}
.zd-navbar-sub {
    font-size: 0.7rem;
    color: var(--text-muted);
    letter-spacing: 0.4px;
    font-weight: 500;
}
.zd-nav-links {
    display: flex;
    align-items: center;
    gap: var(--space-1);
}
.zd-nav-link {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: var(--radius-4);
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s ease;
    border: 1px solid transparent;
    white-space: nowrap;
}
.zd-nav-link:hover {
    color: var(--text-primary);
    background: var(--surface-2);
    border-color: var(--border);
}
.zd-nav-link.active {
    color: var(--accent);
    background: var(--accent-subtle);
    border-color: rgba(0,144,255,0.25);
    font-weight: 600;
}
.zd-nav-badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 8px;
    border-radius: var(--radius-full);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    background: rgba(0,144,255,0.15);
    color: var(--blue-11);
    border: 1px solid rgba(0,144,255,0.3);
}

/* ═══════ PAGE WRAPPER ═══════ */
.zd-page {
    padding: var(--space-6) var(--space-5);
    animation: fadeInUp 0.4s ease;
}

/* ═══════ HERO SECTION ═══════ */
.zd-hero {
    text-align: center;
    padding: var(--space-9) var(--space-5) var(--space-7);
    animation: fadeInUp 0.5s ease;
}
.zd-hero-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 14px;
    border-radius: var(--radius-full);
    background: var(--accent-subtle);
    border: 1px solid rgba(0,144,255,0.3);
    color: var(--blue-11);
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
    margin-bottom: var(--space-4);
}
.zd-hero-title {
    font-size: clamp(2.2rem, 5vw, 3.5rem);
    font-weight: 900;
    letter-spacing: -1.5px;
    line-height: 1.1;
    margin-bottom: var(--space-4);
    background: linear-gradient(135deg, #fff 0%, var(--blue-11) 40%, var(--violet-11) 70%, #fff 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: gradient-shift 4s ease infinite;
}
.zd-hero-desc {
    font-size: 1.05rem;
    color: var(--text-secondary);
    max-width: 600px;
    margin: 0 auto var(--space-6);
    line-height: 1.7;
}
.zd-hero-tags {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--space-2);
    flex-wrap: wrap;
}
.zd-tag {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: var(--radius-full);
    font-size: 0.78rem;
    font-weight: 500;
    border: 1px solid var(--border);
    color: var(--text-secondary);
    background: var(--surface);
    transition: all 0.15s ease;
}
.zd-tag:hover {
    border-color: var(--accent);
    color: var(--accent);
}

/* ═══════ RADIX-STYLE CARDS ═══════ */
.zd-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-5);
    padding: var(--space-5);
    margin-bottom: var(--space-4);
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
    animation: fadeInUp 0.4s ease;
}
.zd-card:hover {
    border-color: var(--border-hover);
    box-shadow: var(--shadow-2);
}
.zd-card-header {
    display: flex;
    align-items: center;
    gap: var(--space-2);
    margin-bottom: var(--space-4);
    padding-bottom: var(--space-3);
    border-bottom: 1px solid var(--border);
}
.zd-card-icon {
    width: 32px;
    height: 32px;
    border-radius: var(--radius-3);
    background: var(--accent-subtle);
    border: 1px solid rgba(0,144,255,0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.zd-card-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.2px;
}

/* ═══════ METRIC CARDS ═══════ */
.zd-metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-5);
    padding: var(--space-4) var(--space-5);
    text-align: center;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
    animation: countUp 0.5s ease;
}
.zd-metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--blue-9), var(--violet-9));
}
.zd-metric-card:hover {
    border-color: rgba(0,144,255,0.3);
    box-shadow: var(--shadow-accent);
    transform: translateY(-2px);
}
.zd-metric-value {
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -1px;
    line-height: 1;
    margin-bottom: 4px;
}
.zd-metric-label {
    font-size: 0.72rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 600;
}

/* ═══════ VERDICT BOXES ═══════ */
.verdict-ok {
    background: rgba(70,167,88,0.08);
    border: 1px solid rgba(70,167,88,0.35);
    border-radius: var(--radius-5);
    padding: var(--space-4) var(--space-5);
    color: var(--green-11);
    font-size: 1.05rem;
    font-weight: 700;
    text-align: center;
    animation: fadeInUp 0.3s ease;
}
.verdict-warn {
    background: rgba(255,178,36,0.08);
    border: 1px solid rgba(255,178,36,0.35);
    border-radius: var(--radius-5);
    padding: var(--space-4) var(--space-5);
    color: var(--amber-11);
    font-size: 1.05rem;
    font-weight: 700;
    text-align: center;
    animation: fadeInUp 0.3s ease;
}
.verdict-bad {
    background: rgba(229,72,77,0.08);
    border: 1px solid rgba(229,72,77,0.35);
    border-radius: var(--radius-5);
    padding: var(--space-4) var(--space-5);
    color: var(--red-11);
    font-size: 1.05rem;
    font-weight: 700;
    text-align: center;
    animation: pulse-ring 1.5s ease, fadeInUp 0.3s ease;
}

/* ═══════ SECTION HEADERS ═══════ */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: var(--space-6) 0 var(--space-3) 0;
    display: flex;
    align-items: center;
    gap: var(--space-2);
    letter-spacing: -0.3px;
}
.section-header::before {
    content: '';
    display: inline-block;
    width: 3px;
    height: 16px;
    border-radius: 2px;
    background: linear-gradient(180deg, var(--blue-9), var(--violet-9));
    flex-shrink: 0;
}

/* ═══════ NOTICE / CALLOUT ═══════ */
.notice {
    display: flex;
    align-items: flex-start;
    gap: var(--space-3);
    background: rgba(0,144,255,0.06);
    border: 1px solid rgba(0,144,255,0.2);
    border-radius: var(--radius-4);
    padding: var(--space-3) var(--space-4);
    font-size: 0.82rem;
    color: var(--text-secondary);
    margin: var(--space-3) 0;
}

/* ═══════ BUTTONS ═══════ */
.stButton > button {
    background: var(--blue-9) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-4) !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 8px 20px !important;
    letter-spacing: 0.1px;
    transition: all 0.15s ease !important;
    font-family: 'Inter', sans-serif !important;
    box-shadow: var(--shadow-1) !important;
}
.stButton > button:hover {
    background: var(--blue-10) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(0,144,255,0.35) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ═══════ INPUTS ═══════ */
.stSelectbox > div > div,
.stSlider,
.stFileUploader,
.stTextInput > div > div {
    animation: fadeIn 0.3s ease;
}
[data-baseweb="select"] > div {
    background: var(--surface-2) !important;
    border-color: var(--border) !important;
    border-radius: var(--radius-4) !important;
    color: var(--text-primary) !important;
}

/* ═══════ TABS ═══════ */
[data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: var(--radius-5) !important;
    padding: 4px !important;
    gap: 2px !important;
    border: 1px solid var(--border) !important;
}
[data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    border-radius: var(--radius-3) !important;
    padding: 6px 14px !important;
    transition: all 0.15s ease !important;
}
[aria-selected="true"] {
    color: var(--text-primary) !important;
    background: var(--surface-2) !important;
}

/* ═══════ IMAGES ═══════ */
[data-testid="stImage"] img {
    border-radius: var(--radius-5);
    border: 1px solid var(--border);
    transition: all 0.2s ease;
}
[data-testid="stImage"] img:hover {
    border-color: var(--border-hover);
    box-shadow: var(--shadow-2);
}

/* ═══════ EXPANDER ═══════ */
.streamlit-expanderHeader {
    background: var(--surface) !important;
    border-radius: var(--radius-4) !important;
    border: 1px solid var(--border) !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
}
.streamlit-expanderContent {
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius-4) var(--radius-4) !important;
    background: var(--surface) !important;
}

/* ═══════ DATAFRAME ═══════ */
[data-testid="stDataFrame"] {
    border-radius: var(--radius-5) !important;
    overflow: hidden;
    border: 1px solid var(--border) !important;
    animation: fadeInUp 0.3s ease;
}

/* ═══════ RADIO ═══════ */
[data-baseweb="radio"] > div {
    gap: var(--space-2) !important;
}

/* ═══════ SUCCESS / ERROR / WARNING messages ═══════ */
.stSuccess {
    background: rgba(70,167,88,0.08) !important;
    border: 1px solid rgba(70,167,88,0.3) !important;
    border-radius: var(--radius-4) !important;
    animation: fadeInUp 0.3s ease;
}
.stError {
    background: rgba(229,72,77,0.08) !important;
    border: 1px solid rgba(229,72,77,0.3) !important;
    border-radius: var(--radius-4) !important;
    animation: fadeInUp 0.3s ease;
}
.stWarning {
    background: rgba(255,178,36,0.08) !important;
    border: 1px solid rgba(255,178,36,0.3) !important;
    border-radius: var(--radius-4) !important;
    animation: fadeInUp 0.3s ease;
}
.stInfo {
    background: rgba(0,144,255,0.08) !important;
    border: 1px solid rgba(0,144,255,0.25) !important;
    border-radius: var(--radius-4) !important;
    animation: fadeInUp 0.3s ease;
}

/* ═══════ SKELETON SHIMMER ═══════ */
.zd-skeleton {
    background: linear-gradient(90deg, var(--surface) 25%, var(--surface-2) 50%, var(--surface) 75%);
    background-size: 1000px 100%;
    animation: shimmer 1.8s infinite;
    border-radius: var(--radius-4);
}

/* ═══════ PROGRESS PIPELINE STEP ═══════ */
.zd-step {
    display: flex;
    align-items: flex-start;
    gap: var(--space-3);
    padding: var(--space-3);
    border-radius: var(--radius-4);
    transition: all 0.15s ease;
    animation: slideInLeft 0.4s ease;
}
.zd-step:hover {
    background: var(--surface-2);
}
.zd-step-num {
    width: 28px;
    height: 28px;
    border-radius: var(--radius-full);
    background: var(--accent-subtle);
    border: 1px solid rgba(0,144,255,0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--blue-11);
    flex-shrink: 0;
}
.zd-step-name {
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 2px;
}
.zd-step-desc {
    font-size: 0.8rem;
    color: var(--text-muted);
    line-height: 1.5;
}

/* ═══════ STAT ROW ═══════ */
.zd-stat-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: var(--space-2) 0;
    border-bottom: 1px solid var(--border);
    font-size: 0.85rem;
}
.zd-stat-row:last-child { border-bottom: none; }
.zd-stat-key { color: var(--text-muted); font-weight: 500; }
.zd-stat-val { color: var(--text-primary); font-weight: 600; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 0.8rem; }

/* ═══════ FEATURE PILL ═══════ */
.zd-feature-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: var(--radius-full);
    font-size: 0.8rem;
    font-weight: 500;
    background: var(--surface-2);
    border: 1px solid var(--border);
    color: var(--text-secondary);
    margin: 3px;
    transition: all 0.15s ease;
}
.zd-feature-pill:hover {
    background: var(--accent-subtle);
    border-color: rgba(0,144,255,0.3);
    color: var(--blue-11);
}

/* ═══════ SCROLLBAR ═══════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--gray-2); }
::-webkit-scrollbar-thumb { background: var(--gray-7); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--gray-8); }

/* ═══════ DIVIDER ═══════ */
hr { border-color: var(--border) !important; margin: var(--space-5) 0 !important; }

/* ═══════ SPINNER ═══════ */
.stSpinner > div {
    border-color: var(--blue-9) transparent transparent transparent !important;
}
</style>
""")


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
        st.markdown(
            f'<div class="verdict-ok">'
            f'<span style="font-size:1.3rem;">✅</span>&nbsp; {verdict}'
            f'</div>',
            unsafe_allow_html=True
        )
    elif "inconclusive" in verdict.lower():
        st.markdown(
            f'<div class="verdict-warn">'
            f'<span style="font-size:1.3rem;">⚠️</span>&nbsp; {verdict}'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="verdict-bad">'
            f'<span style="font-size:1.3rem;">🚨</span>&nbsp; {verdict}'
            f'</div>',
            unsafe_allow_html=True
        )


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
# TOP NAVIGATION BAR
# ════════════════════════════════════════════════

NAV_PAGES = [
    ("🏠", "Overview",      "🏠 Overview"),
    ("🔍", "Inspect",       "🔍 Inspect Image"),
    ("📊", "Visualise",     "📊 Visual Analysis"),
    ("📈", "Metrics",       "📈 Metrics & Evaluation"),
    ("📤", "Export",        "📤 Export"),
]

def render_navbar():
    """Render a sticky top navigation bar and return the active page key."""
    if "active_page" not in st.session_state:
        st.session_state.active_page = "🏠 Overview"

    # Build link buttons HTML — clicking sets session state via a hidden Streamlit radio
    links_html = ""
    for icon, label, key in NAV_PAGES:
        is_active = st.session_state.active_page == key
        active_cls = "active" if is_active else ""
        links_html += f'<span class="zd-nav-link {active_cls}" onclick="void(0)">{icon} {label}</span>'

    st.markdown(f"""
    <div class="zd-navbar">
        <div class="zd-navbar-brand">
            <span class="zd-navbar-logo">🔬</span>
            <div>
                <div class="zd-navbar-wordmark">ZeroDefect</div>
                <div class="zd-navbar-sub">SURFACE INSPECTION SYSTEM</div>
            </div>
        </div>
        <div class="zd-nav-links">
            {links_html}
        </div>
        <span class="zd-nav-badge">v1.0 Academic</span>
    </div>
    """, unsafe_allow_html=True)

    # Use a hidden radio for actual Streamlit state
    page = st.radio(
        "Page",
        [key for _, _, key in NAV_PAGES],
        index=[key for _, _, key in NAV_PAGES].index(st.session_state.active_page),
        horizontal=True,
        label_visibility="collapsed",
        key="_nav_radio",
    )
    st.session_state.active_page = page

    return page


# ════════════════════════════════════════════════
# PAGE: OVERVIEW
# ════════════════════════════════════════════════

def page_overview():
    st.html("""
    <style>
    @keyframes scan-line {
      0%   { left: -60%; }
      100% { left: 120%; }
    }
    @keyframes border-rotate {
      0%   { background-position: 0% 50%; }
      50%  { background-position: 100% 50%; }
      100% { background-position: 0% 50%; }
    }
    @keyframes stagger-in {
      from { opacity:0; transform: translateY(30px); }
      to   { opacity:1; transform: translateY(0); }
    }
    @keyframes shimmer-bar {
      0%   { left: -100%; }
      100% { left: 200%; }
    }
    @keyframes rotate-ring {
      from { transform: rotate(0deg); }
      to   { transform: rotate(360deg); }
    }
    @keyframes blink-dot {
      0%,100% { opacity:1; } 50% { opacity:0.2; }
    }
    @keyframes wave {
      0%,100% { transform: scaleY(0.4); } 50% { transform: scaleY(1); }
    }

    .ov-hero {
      position: relative; text-align: center;
      padding: 56px 24px 48px; margin-bottom: 32px;
      border-radius: 16px; overflow: hidden;
      background:
        radial-gradient(ellipse 80% 60% at 50% 0%, rgba(0,144,255,0.15) 0%, transparent 70%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(110,86,207,0.12) 0%, transparent 70%),
        #111113;
      border: 1px solid rgba(0,144,255,0.2);
    }
    .ov-hero::before {
      content: ''; position: absolute;
      top: 0; left: -60%; width: 60%; height: 2px;
      background: linear-gradient(90deg, transparent, #0090ff, #6e56cf, transparent);
      animation: scan-line 3s ease-in-out infinite; opacity: 0.7;
    }
    .ov-hero-badge {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 4px 14px; border-radius: 9999px;
      background: rgba(0,144,255,0.1); border: 1px solid rgba(0,144,255,0.35);
      color: #70b8ff; font-size: 0.72rem; font-weight: 700;
      letter-spacing: 1.2px; text-transform: uppercase;
      margin-bottom: 20px; animation: stagger-in 0.6s ease both;
    }
    .ov-hero-dot {
      width: 6px; height: 6px; border-radius: 50%;
      background: #0090ff; display: inline-block;
      animation: blink-dot 1.4s ease infinite;
    }
    .ov-hero-title {
      font-size: clamp(2.8rem, 6vw, 4.2rem); font-weight: 900;
      letter-spacing: -2px; line-height: 1.05; margin-bottom: 16px;
      background: linear-gradient(135deg, #fff 0%, #70b8ff 35%, #baa7ff 65%, #fff 100%);
      background-size: 300% 300%;
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      background-clip: text;
      animation: stagger-in 0.7s ease 0.1s both, border-rotate 5s ease infinite;
    }
    .ov-hero-sub {
      font-size: 1.05rem; color: #b2b3bc; max-width: 560px;
      margin: 0 auto 28px; line-height: 1.7;
      animation: stagger-in 0.7s ease 0.2s both;
    }
    .ov-hero-tags {
      display: flex; gap: 8px; flex-wrap: wrap; justify-content: center;
      animation: stagger-in 0.7s ease 0.3s both;
    }
    .ov-tag {
      padding: 5px 14px; border-radius: 9999px; font-size: 0.78rem;
      font-weight: 600; letter-spacing: 0.5px; border: 1px solid;
      transition: all 0.2s ease;
    }
    .ov-tag.blue   { background: rgba(0,144,255,0.1);  border-color: rgba(0,144,255,0.3);  color: #70b8ff; }
    .ov-tag.violet { background: rgba(110,86,207,0.1); border-color: rgba(110,86,207,0.3); color: #baa7ff; }
    .ov-tag.green  { background: rgba(70,167,88,0.1);  border-color: rgba(70,167,88,0.3);  color: #71d083; }

    .ov-stats {
      display: grid; grid-template-columns: repeat(4, 1fr);
      gap: 12px; margin-bottom: 28px;
    }
    .ov-stat {
      background: #19191b; border: 1px solid #393a40;
      border-radius: 12px; padding: 20px 16px; text-align: center;
      position: relative; overflow: hidden;
      transition: all 0.2s ease; animation: stagger-in 0.6s ease both;
    }
    .ov-stat::before {
      content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
      background: linear-gradient(90deg, var(--c1), var(--c2));
    }
    .ov-stat::after {
      content: ''; position: absolute; top: 0; left: -100%; width: 60%; height: 100%;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,0.04), transparent);
      animation: shimmer-bar 2.4s ease infinite;
    }
    .ov-stat:hover { border-color: #46474e; transform: translateY(-3px); box-shadow: 0 8px 24px rgba(0,0,0,0.4); }
    .ov-stat-val { font-size: 2.2rem; font-weight: 900; letter-spacing: -1px; line-height: 1; margin-bottom: 4px; }
    .ov-stat-lbl { font-size: 0.72rem; color: #7d7f8a; text-transform: uppercase; letter-spacing: 0.8px; font-weight: 600; }

    .ov-grid { display: grid; grid-template-columns: 1.4fr 1fr; gap: 16px; margin-bottom: 20px; }

    .ov-card {
      background: #19191b; border: 1px solid #393a40;
      border-radius: 14px; padding: 24px; position: relative;
      overflow: hidden; transition: border-color 0.2s ease, box-shadow 0.2s ease;
      animation: stagger-in 0.6s ease both;
    }
    .ov-card:hover { border-color: #46474e; box-shadow: 0 4px 24px rgba(0,0,0,0.5); }
    .ov-card-title {
      display: flex; align-items: center; gap: 10px;
      font-size: 0.95rem; font-weight: 700; color: #edeef0;
      letter-spacing: -0.2px; margin-bottom: 16px;
      padding-bottom: 12px; border-bottom: 1px solid #393a40;
    }
    .ov-card-icon {
      width: 30px; height: 30px; border-radius: 7px;
      display: flex; align-items: center; justify-content: center;
      font-size: 0.95rem; flex-shrink: 0;
    }
    .ov-card-icon.blue   { background: rgba(0,144,255,0.15); border: 1px solid rgba(0,144,255,0.25); }
    .ov-card-icon.violet { background: rgba(110,86,207,0.15); border: 1px solid rgba(110,86,207,0.25); }
    .ov-card-icon.green  { background: rgba(70,167,88,0.15); border: 1px solid rgba(70,167,88,0.25); }
    .ov-card-icon.amber  { background: rgba(255,178,36,0.15); border: 1px solid rgba(255,178,36,0.25); }

    .ov-step {
      display: flex; align-items: flex-start; gap: 12px;
      padding: 10px; border-radius: 8px;
      transition: background 0.15s ease; animation: stagger-in 0.5s ease both; margin-bottom: 2px;
    }
    .ov-step:hover { background: #222325; }
    .ov-step-num {
      width: 26px; height: 26px; border-radius: 50%;
      background: rgba(0,144,255,0.12); border: 1px solid rgba(0,144,255,0.3);
      display: flex; align-items: center; justify-content: center;
      font-size: 0.7rem; font-weight: 800; color: #70b8ff;
      flex-shrink: 0; margin-top: 1px;
    }
    .ov-step-name { font-size: 0.85rem; font-weight: 700; color: #edeef0; margin-bottom: 2px; }
    .ov-step-desc { font-size: 0.78rem; color: #7d7f8a; line-height: 1.5; }

    .ov-anomaly-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    .ov-anomaly {
      display: flex; align-items: center; gap: 8px;
      padding: 9px 12px; background: #222325; border: 1px solid #393a40;
      border-radius: 8px; font-size: 0.82rem; font-weight: 500; color: #b2b3bc;
      transition: all 0.15s ease;
    }
    .ov-anomaly:hover { background: #292a2e; border-color: #46474e; color: #edeef0; }
    .ov-anomaly-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }

    .ov-tech-row {
      display: flex; align-items: center; justify-content: space-between;
      padding: 8px 0; border-bottom: 1px solid #292a2e; font-size: 0.82rem;
    }
    .ov-tech-row:last-child { border-bottom: none; }
    .ov-tech-name { font-weight: 700; color: #edeef0; }
    .ov-tech-desc { color: #7d7f8a; font-size: 0.78rem; }

    .ov-dataset {
      background: rgba(0,144,255,0.06); border: 1px solid rgba(0,144,255,0.2);
      border-radius: 10px; padding: 14px 16px; margin-bottom: 10px;
    }
    .ov-dataset-title { font-size: 0.85rem; font-weight: 700; color: #70b8ff; margin-bottom: 6px; }
    .ov-dataset-sub { font-size: 0.78rem; color: #7d7f8a; line-height: 1.6; }

    .ov-wave { display: flex; align-items: center; gap: 3px; height: 24px; margin-left: auto; }
    .ov-wave-bar { width: 3px; background: #0090ff; border-radius: 2px; animation: wave 1s ease-in-out infinite; }
    .ov-wave-bar:nth-child(2) { animation-delay: 0.1s; background: #3b9eff; }
    .ov-wave-bar:nth-child(3) { animation-delay: 0.2s; background: #6e56cf; }
    .ov-wave-bar:nth-child(4) { animation-delay: 0.3s; background: #7d66d9; }
    .ov-wave-bar:nth-child(5) { animation-delay: 0.4s; background: #baa7ff; }

    .ov-ring-deco { position: absolute; top: 16px; right: 16px; width: 40px; height: 40px; }
    .ov-ring-deco::before {
      content: ''; position: absolute; inset: 0; border-radius: 50%;
      border: 2px solid transparent;
      border-top-color: rgba(0,144,255,0.4); border-right-color: rgba(110,86,207,0.3);
      animation: rotate-ring 3s linear infinite;
    }

    @media (max-width: 768px) {
      .ov-stats { grid-template-columns: repeat(2,1fr); }
      .ov-grid  { grid-template-columns: 1fr; }
    }
    </style>

    <div class="ov-hero">
      <div class="ov-hero-badge"><span class="ov-hero-dot"></span> Live &middot; Academic Prototype</div>
      <div class="ov-hero-title">ZeroDefect</div>
      <div class="ov-hero-sub">
        Intelligent Industrial Surface Defect Detection &amp; Localisation System &mdash;
        powered by classical computer vision and texture analysis.
      </div>
      <div class="ov-hero-tags">
        <span class="ov-tag blue">&#128300; Computer Vision</span>
        <span class="ov-tag violet">&#129504; Image Processing</span>
        <span class="ov-tag green">&#128202; Real Dataset (NEU)</span>
        <span class="ov-tag blue">&#9889; Streamlit</span>
        <span class="ov-tag violet">&#128013; Python 3</span>
      </div>
    </div>

    <div class="ov-stats">
      <div class="ov-stat" style="--c1:#0090ff;--c2:#3b9eff; animation-delay:0.0s;">
        <div class="ov-stat-val" style="color:#70b8ff;">51</div>
        <div class="ov-stat-lbl">Real Dataset Images</div>
      </div>
      <div class="ov-stat" style="--c1:#6e56cf;--c2:#baa7ff; animation-delay:0.1s;">
        <div class="ov-stat-val" style="color:#baa7ff;">5</div>
        <div class="ov-stat-lbl">Defect Categories</div>
      </div>
      <div class="ov-stat" style="--c1:#46a758;--c2:#71d083; animation-delay:0.2s;">
        <div class="ov-stat-val" style="color:#71d083;">8</div>
        <div class="ov-stat-lbl">Pipeline Stages</div>
      </div>
      <div class="ov-stat" style="--c1:#ffb224;--c2:#ffcb47; animation-delay:0.3s;">
        <div class="ov-stat-val" style="color:#f1a10d;">2</div>
        <div class="ov-stat-lbl">Detection Methods</div>
      </div>
    </div>

    <div class="ov-grid">
      <div style="display:flex; flex-direction:column; gap:16px;">
        <div class="ov-card" style="animation-delay:0.1s;">
          <div class="ov-ring-deco"></div>
          <div class="ov-card-title">
            <div class="ov-card-icon blue">&#128300;</div>
            <span>What is ZeroDefect?</span>
            <div class="ov-wave"><div class="ov-wave-bar"></div><div class="ov-wave-bar"></div><div class="ov-wave-bar"></div><div class="ov-wave-bar"></div><div class="ov-wave-bar"></div></div>
          </div>
          <p style="font-size:0.87rem; color:#b2b3bc; line-height:1.75; margin:0 0 12px;">
            ZeroDefect is an academic computer vision system that analyses images of manufactured
            product surfaces to identify potentially defective regions &mdash; scratches, cracks, dents,
            surface contamination, and unusual textures.
          </p>
          <p style="font-size:0.87rem; color:#b2b3bc; line-height:1.75; margin:0;">
            It combines <strong style="color:#edeef0;">classical digital image processing</strong>
            with <strong style="color:#edeef0;">texture feature analysis</strong> to detect, highlight,
            measure, and localise anomalous surface regions in real time.
          </p>
        </div>
        <div class="ov-card" style="animation-delay:0.2s;">
          <div class="ov-card-title"><div class="ov-card-icon violet">&#9881;</div><span>Inspection Pipeline</span></div>
          <div class="ov-step" style="animation-delay:0.15s;"><div class="ov-step-num">1</div><div><div class="ov-step-name">Image Acquisition</div><div class="ov-step-desc">Load, validate, and display the original image with full metadata.</div></div></div>
          <div class="ov-step" style="animation-delay:0.20s;"><div class="ov-step-num">2</div><div><div class="ov-step-name">Preprocessing</div><div class="ov-step-desc">Grayscale, Gaussian/Median blur, CLAHE contrast enhancement.</div></div></div>
          <div class="ov-step" style="animation-delay:0.25s;"><div class="ov-step-num">3</div><div><div class="ov-step-name">Edge Detection</div><div class="ov-step-desc">Canny algorithm detects structural boundaries and irregularities.</div></div></div>
          <div class="ov-step" style="animation-delay:0.30s;"><div class="ov-step-num">4</div><div><div class="ov-step-name">Thresholding</div><div class="ov-step-desc">Otsu's and adaptive methods create binary segmentation masks.</div></div></div>
          <div class="ov-step" style="animation-delay:0.35s;"><div class="ov-step-num">5</div><div><div class="ov-step-name">Morphological Processing</div><div class="ov-step-desc">Erosion/dilation clean noise and connect defect fragments.</div></div></div>
          <div class="ov-step" style="animation-delay:0.40s;"><div class="ov-step-num">6</div><div><div class="ov-step-name">Texture Analysis</div><div class="ov-step-desc">GLCM features and local std-dev maps score anomaly severity.</div></div></div>
          <div class="ov-step" style="animation-delay:0.45s;"><div class="ov-step-num">7</div><div><div class="ov-step-name">Defect Localisation</div><div class="ov-step-desc">Suspicious regions identified, measured, and labelled with bounding boxes.</div></div></div>
          <div class="ov-step" style="animation-delay:0.50s;"><div class="ov-step-num">8</div><div><div class="ov-step-name">Evaluation</div><div class="ov-step-desc">Pixel-level F1, IoU, Precision &amp; Recall against ground-truth masks.</div></div></div>
        </div>
      </div>

      <div style="display:flex; flex-direction:column; gap:16px;">
        <div class="ov-card" style="animation-delay:0.15s;">
          <div class="ov-card-title"><div class="ov-card-icon green">&#128269;</div><span>Detectable Anomalies</span></div>
          <div class="ov-anomaly-grid">
            <div class="ov-anomaly"><div class="ov-anomaly-dot" style="background:#e5484d;"></div>Scratches</div>
            <div class="ov-anomaly"><div class="ov-anomaly-dot" style="background:#ff6b35;"></div>Cracks</div>
            <div class="ov-anomaly"><div class="ov-anomaly-dot" style="background:#ffb224;"></div>Dents</div>
            <div class="ov-anomaly"><div class="ov-anomaly-dot" style="background:#46a758;"></div>Contamination</div>
            <div class="ov-anomaly"><div class="ov-anomaly-dot" style="background:#0090ff;"></div>Unusual Textures</div>
            <div class="ov-anomaly"><div class="ov-anomaly-dot" style="background:#6e56cf;"></div>Colour Anomalies</div>
            <div class="ov-anomaly"><div class="ov-anomaly-dot" style="background:#b2b3bc;"></div>Foreign Objects</div>
            <div class="ov-anomaly"><div class="ov-anomaly-dot" style="background:#70b8ff;"></div>Wipe Cracks</div>
          </div>
        </div>
        <div class="ov-card" style="animation-delay:0.25s;">
          <div class="ov-card-title"><div class="ov-card-icon amber">&#128295;</div><span>Technology Stack</span></div>
          <div class="ov-tech-row"><span class="ov-tech-name">Python 3</span><span class="ov-tech-desc">Core language</span></div>
          <div class="ov-tech-row"><span class="ov-tech-name">OpenCV</span><span class="ov-tech-desc">Image processing</span></div>
          <div class="ov-tech-row"><span class="ov-tech-name">scikit-image</span><span class="ov-tech-desc">GLCM, LBP features</span></div>
          <div class="ov-tech-row"><span class="ov-tech-name">NumPy</span><span class="ov-tech-desc">Pixel-level operations</span></div>
          <div class="ov-tech-row"><span class="ov-tech-name">scikit-learn</span><span class="ov-tech-desc">Evaluation metrics</span></div>
          <div class="ov-tech-row"><span class="ov-tech-name">Streamlit</span><span class="ov-tech-desc">Web interface</span></div>
          <div class="ov-tech-row"><span class="ov-tech-name">Plotly / Matplotlib</span><span class="ov-tech-desc">Visualisations</span></div>
        </div>
        <div class="ov-card" style="animation-delay:0.35s;">
          <div class="ov-card-title"><div class="ov-card-icon blue">&#128194;</div><span>Dataset</span></div>
          <div class="ov-dataset">
            <div class="ov-dataset-title">NEU Micro Surface Defect DB</div>
            <div class="ov-dataset-sub">
              51 real industrial images &middot; 5 defect types<br/>
              Scratch &middot; Small Defect &middot; Wipe Crack &middot; SDI &middot; SPDI<br/>
              <strong style="color:#70b8ff;">Built-in &mdash; available directly in the app</strong>
            </div>
          </div>
          <div class="ov-dataset" style="background:rgba(110,86,207,0.06); border-color:rgba(110,86,207,0.2);">
            <div class="ov-dataset-title" style="color:#baa7ff;">MVTec AD (optional)</div>
            <div class="ov-dataset-sub">
              For ground-truth mask evaluation only.<br/>
              Non-commercial academic licence &middot; <a href="https://www.mvtec.com/research-teaching/datasets/mvtec-ad" style="color:#baa7ff;" target="_blank">mvtec.com</a>
            </div>
          </div>
        </div>
      </div>
    </div>
    """)


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
        ["📁 Upload image", "📂 Real Dataset (NEU)", "🧪 Use synthetic demo"],
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

    elif source_tab == "📂 Real Dataset (NEU)":
        # NEU Micro Surface Defect Database — real industrial images
        neu_base = ROOT / "data" / "neu_surface_defect"
        categories = {
            "scratch":      "Scratch defects (oil-contaminated steel surface)",
            "small_defect": "Small surface defects (micro-scale)",
            "wipe_crack":   "Wipe cracks (linear surface cracks)",
            "sdi":          "SDI — Surface Defect Images (steel)",
            "spdi":         "SPDI — Steel Pipe Defect Images",
        }

        # Build list of available images
        cat_choice = st.selectbox(
            "Defect Category",
            list(categories.keys()),
            format_func=lambda x: f"{x.replace('_', ' ').title()} — {categories[x]}",
            key="neu_cat",
        )

        cat_dir = neu_base / cat_choice
        imgs_available = sorted(cat_dir.glob("*.jpg")) if cat_dir.exists() else []

        if not imgs_available:
            st.error("⚠️ Dataset images not found. Please check the data/neu_surface_defect/ folder.")
        else:
            img_names = [p.name for p in imgs_available]
            chosen = st.selectbox(
                f"Select image ({len(imgs_available)} available)",
                img_names,
                key="neu_img",
            )
            chosen_path = cat_dir / chosen
            try:
                with open(chosen_path, "rb") as f:
                    image = load_image_from_bytes(f.read())
                st.success(
                    f"✅ Loaded real NEU image: **{chosen}** "
                    f"({image.shape[1]}×{image.shape[0]} px) — Category: *{cat_choice}*"
                )
                st.info(
                    "📌 **Dataset:** NEU Micro Surface Defect Database "
                    "(Northeastern University) — Real industrial surface images."
                )
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
    page = render_navbar()

    st.markdown('<div class="zd-page">', unsafe_allow_html=True)

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

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div style="
        text-align:center;
        padding: 32px 0 24px;
        border-top: 1px solid var(--border);
        margin-top: 48px;
        color: var(--text-muted);
        font-size: 0.78rem;
    ">
        <div style="margin-bottom:6px;">
            <strong style="color:var(--text-secondary);">ZeroDefect</strong>
            &nbsp;·&nbsp; Surface Defect Detection
            &nbsp;·&nbsp; Academic Prototype
            &nbsp;·&nbsp; v1.0
        </div>
        <div>
            Dataset: <a href="https://github.com/abin24/Surface-Inspection-defect-detection-dataset"
            style="color:var(--blue-11); text-decoration:none;"
            target="_blank">NEU Micro Surface Defect DB</a>
            &nbsp;·&nbsp; Built with OpenCV, scikit-image, Streamlit
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
