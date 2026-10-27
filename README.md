# ZeroDefect — Intelligent Industrial Surface Defect Detection and Localization System

> **Academic prototype** — Results require human verification and are not suitable for industrial deployment.

---

## Project Overview

ZeroDefect is a computer vision application that analyses images of manufactured product surfaces to identify potentially defective regions: scratches, cracks, dents, contamination, and unusual textures.

The system demonstrates how **classical digital image processing** combined with **texture feature analysis** can be used to detect and localize surface anomalies.

---

## Features

- **Preprocessing pipeline**: Grayscale, Gaussian/Median blur, CLAHE contrast enhancement
- **Edge detection**: Canny with configurable or auto-selected thresholds
- **Segmentation**: Otsu's and adaptive thresholding with binary mask output
- **Morphological cleanup**: Opening, closing, small-region removal
- **Texture features**: GLCM (contrast, homogeneity, energy, correlation), LBP, local std-dev maps
- **Two detection approaches**:
  - **Classical**: Edge + intensity thresholding pipeline
  - **Texture anomaly**: Statistical baseline from normal images
- **Region analysis**: Connected components, bounding boxes, area measurements (pixel-based)
- **Evaluation**: Pixel-level Precision, Recall, F1, IoU, and confusion matrix
- **Streamlit interface**: 5-page web app with visualizations and CSV export

---

## Project Structure

```
ZeroDefect/
├── app.py                       ← Streamlit web interface (main entry point)
├── requirements.txt             ← Python dependencies
├── README.md                    ← This file
├── src/
│   ├── __init__.py
│   ├── preprocessing.py         ← Stage 1 & 2: Loading, blur, CLAHE
│   ├── edge_detection.py        ← Stage 3: Canny, Sobel
│   ├── segmentation.py          ← Stage 4 & 5: Thresholding, morphology
│   ├── texture_features.py      ← Stage 6: GLCM, LBP, local std
│   ├── defect_detection.py      ← Stage 7 & 8: Classical + texture pipelines, region analysis
│   ├── evaluation.py            ← Stage 9: Metrics computation
│   └── visualization.py        ← Matplotlib/Plotly visualizations
├── tests/
│   ├── test_preprocessing.py
│   └── test_defect_detection.py
├── notebooks/
│   └── ZeroDefect_Development.ipynb
├── data/
│   └── README.md               ← Dataset setup instructions
├── reports/
│   ├── methodology.md
│   └── results_template.md
└── outputs/
    ├── figures/                 ← Saved plots
    └── metrics/                 ← Saved CSV results
```

---

## Setup and Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Step 1: Clone or download the project

```bash
# If using git:
git clone <repository-url>
cd ZeroDefect

# Or just navigate to the ZeroDefect folder
```

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# macOS/Linux:
source venv/bin/activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set up the dataset

See [`data/README.md`](data/README.md) for instructions on downloading and organizing the MVTec AD dataset.

---

## Running the Application

### Quick Verification Script (No UI required)

```bash
python demo.py
```
Runs the complete dual-approach pipeline on normal and defective surfaces, validates pixel-level precision/recall/IoU, and exports figures to `outputs/figures/` and metrics to `outputs/metrics/benchmark_summary.csv`.

### Web Interface (Streamlit)

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`

### Run Unit Tests (60 test cases)

```bash
python -m pytest tests/ -v
```

### Run Specific Test File

```bash
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_defect_detection.py -v
```

---

## Using the Interface

### 🏠 Overview
Project description, pipeline explanation, and technology stack.

### 🔍 Inspect Image
1. Upload a surface image (JPG, PNG, etc.) **or** select a synthetic demo
2. Configure detection settings (optional)
3. Click **Run Defect Inspection**
4. View verdict, region count, overlay, and bounding boxes

### 📊 Visual Analysis
Step-by-step visualisation of every pipeline stage:
- Preprocessing stages (grayscale → blur → CLAHE)
- Edge map vs. threshold masks (Otsu vs. adaptive)
- Cleaned mask and annotations
- Texture std-dev map and GLCM features

### 📈 Metrics & Evaluation
- Upload a ground-truth mask (from MVTec AD) to compute F1, IoU, Precision, Recall
- View the pixel-level confusion matrix
- Metric explanations included

### 📤 Export
Download processed images, prediction masks, and CSV reports.

---

## Key Technical Decisions

| Decision | Rationale |
|---|---|
| RGB→BGR handled in `load_image` | OpenCV defaults to BGR; converting immediately avoids later bugs |
| `auto_canny` as default | Adapts thresholds to image statistics; more robust across images |
| Combined edge + intensity mask | Neither alone is sufficient; combining reduces false negatives |
| Morphological cleanup with `min_area` | Removes noise without losing genuine defects |
| Local std-dev as anomaly score | Simple, interpretable, no training data required for Approach B |
| `matplotlib.use("Agg")` | Prevents GUI backend errors in Streamlit |

---

## Limitations and Honest Caveats

- This is an academic prototype, not industrial software.
- Performance will vary significantly across surface types and lighting conditions.
- The classical pipeline may produce false positives on textured or patterned surfaces.
- No GPU or deep learning model is used — more complex defects may be missed.
- Pixel-to-millimetre conversion is not supported without camera calibration.
- Results depend on threshold choices; optimal thresholds differ per image/category.
- Do not use as a substitute for validated industrial inspection equipment.

---

## Dataset

This project uses the **MVTec Anomaly Detection (MVTec AD)** dataset.

> Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger:  
> *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*  
> CVPR 2019. https://www.mvtec.com/research-teaching/datasets/mvtec-ad

Licensed for non-commercial, academic use only.

---

## Author and Context

Created as an academic project demonstrating the practical application of digital image processing and computer vision for industrial surface inspection. Built using Python 3, OpenCV, scikit-image, scikit-learn, and Streamlit.

*Seven-day implementation schedule: Day 1 (setup) → Day 7 (final verification)*
