# ZeroDefect — Complete Viva & Presentation Guide
### Surface Defect Detection System | Image Processing Project

---

## TABLE OF CONTENTS
1. [Project Overview](#1-project-overview)
2. [Technology Stack — Full Breakdown](#2-technology-stack)
3. [Backend: Image Processing Pipeline (Stage by Stage)](#3-backend-image-processing-pipeline)
4. [Core Algorithms Explained](#4-core-algorithms-explained)
5. [Evaluation Metrics](#5-evaluation-metrics)
6. [Frontend: React + Tailwind UI](#6-frontend-react--tailwind-ui)
7. [UI Design Concepts & Libraries](#7-ui-design-concepts--libraries)
8. [Project Architecture Diagram](#8-project-architecture)
9. [Viva Q&A — Expected Questions & Model Answers](#9-viva-qa)
10. [Slide-by-Slide Presentation Guide](#10-slide-guide)

---

## 1. PROJECT OVERVIEW

**ZeroDefect** is an end-to-end surface defect detection system that uses classical image processing (no deep learning required) to automatically identify surface anomalies on manufactured parts, fabrics, or any textured material.

### What problem does it solve?
In manufacturing, visual quality control is traditionally done by humans, which is:
- Slow (bottleneck on production lines)
- Inconsistent (human fatigue, subjectivity)
- Expensive (labour costs)

ZeroDefect automates this using computer vision, flagging defective regions with bounding boxes, heatmaps, and quality metrics — all inside a rich, interactive web dashboard.

### Key Outcomes
| Feature | Detail |
|---|---|
| Defect localisation | Bounding boxes + segmentation masks |
| Texture anomaly detection | Statistical deviation from normal baseline |
| Quality metrics | F1, IoU, Precision, Recall, Accuracy |
| Export | CSV reports, PNG masks, annotated images |
| Interface | Streamlit backend + React/Tailwind frontend |

---

## 2. TECHNOLOGY STACK

### 2.1 Python Backend Libraries

| Library | Version | Purpose |
|---|---|---|
| **OpenCV** (`cv2`) | >= 4.8 | Core image I/O, filtering, edge detection, morphology, drawing |
| **NumPy** | >= 1.24 | N-dimensional array manipulation (all images are NumPy arrays) |
| **scikit-image** | >= 0.21 | GLCM texture features, LBP, SSIM, image quality metrics |
| **scikit-learn** | >= 1.3 | Confusion matrix, precision/recall/F1 computation |
| **Pillow (PIL)** | >= 10.0 | Image format conversion, thumbnail generation |
| **SciPy** | >= 1.11 | Gaussian filters, statistical functions |
| **pandas** | >= 2.0 | Tabular data for defect region reports |
| **matplotlib / seaborn** | >= 3.7 / 0.12 | Plotting histograms, confusion matrices |
| **Streamlit** | >= 1.40 | Python-based web UI framework |
| **Plotly** | >= 5.17 | Interactive charts in the dashboard |

### 2.2 Frontend Libraries

| Library | Version | Purpose |
|---|---|---|
| **React** | 19 | Component-based UI framework |
| **Vite** | 8 | Ultra-fast dev server and build tool |
| **Tailwind CSS v4** | 4.x | Utility-first CSS framework |
| **Framer Motion** | latest | Declarative animations and transitions |
| **Lucide React** | latest | Icon set |
| **Canvas API** (native) | — | HTML5 particle system with mouse interaction |

---

## 3. BACKEND IMAGE PROCESSING PIPELINE

The system processes each image through 8 defined stages:

```
[Input Image]
     |
     v
Stage 1: PREPROCESSING
  - Convert to Grayscale
  - Gaussian Blur (noise removal)
  - CLAHE (contrast enhancement)
     |
     v
Stage 2: EDGE DETECTION
  - Canny Edge Detector
  - Sobel Gradient (optional)
  - Auto-Canny (adaptive thresholds)
     |
     v
Stage 3: THRESHOLDING / SEGMENTATION
  - Otsu's Global Threshold
  - Adaptive Local Threshold
     |
     v
Stage 4: MORPHOLOGICAL OPERATIONS
  - Opening (remove noise)
  - Closing (fill gaps)
  - Erosion / Dilation
     |
     v
Stage 5: TEXTURE FEATURE EXTRACTION
  - GLCM (Gray-Level Co-occurrence Matrix)
  - LBP (Local Binary Pattern)
  - Local Standard Deviation Map
     |
     v
Stage 6: TEXTURE ANOMALY DETECTION
  - Build normal surface model
  - Z-score / Mahalanobis deviation
     |
     v
Stage 7: DEFECT LOCALISATION
  - Contour extraction (cv2.findContours)
  - Bounding boxes
  - Region labelling
     |
     v
Stage 8: REGION ANALYSIS & EVALUATION
  - Area, perimeter, centroid per region
  - Connected component stats
  - F1 / IoU / Precision / Recall
     |
     v
[Output: Annotated Image + Metrics + CSV Report]
```

---

## 4. CORE ALGORITHMS EXPLAINED

### 4.1 Grayscale Conversion
- **Formula:** `Gray = 0.299 * R + 0.587 * G + 0.114 * B`
- Why: Reduces 3-channel image to 1 channel. Human perception weighs green most, then red, then blue.
- OpenCV: `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)`

---

### 4.2 Gaussian Blur (Noise Removal)
- Convolves the image with a 2D Gaussian kernel.
- Why: Smooths out sensor noise so edge detectors don't fire on random pixel variation.
- OpenCV: `cv2.GaussianBlur(img, (k, k), sigma)`

---

### 4.3 CLAHE — Contrast Limited Adaptive Histogram Equalization
- **Standard HE**: Spreads pixel intensity globally to fill the full 0–255 range.
- **CLAHE improvement**: Divides image into small tiles (e.g., 8x8). Equalization is applied to each tile independently. The "contrast limit" clips the histogram to prevent over-amplifying noise in flat regions.
- Why: Enhances local contrast so subtle surface texture defects (scratches, dents) become more visible.
- OpenCV: `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))`

---

### 4.4 Canny Edge Detection — 4-Step Algorithm
1. **Gaussian smoothing** — remove high-frequency noise.
2. **Gradient computation** — Sobel operators compute Gx and Gy. Magnitude M = sqrt(Gx^2 + Gy^2), Angle theta = arctan(Gy/Gx).
3. **Non-maximum suppression** — thin edges to 1-pixel width by suppressing non-peak gradient values along edge normal direction.
4. **Double thresholding + hysteresis** — strong edges (above highThresh) are kept; weak edges (between low and high) are kept only if connected to a strong edge; everything below lowThresh is discarded.

- OpenCV: `cv2.Canny(gray, low, high)`
- **Auto-Canny**: Computes optimal thresholds using the median pixel intensity: `low = max(0, 0.66 * median)`, `high = min(255, 1.33 * median)`.

---

### 4.5 Sobel Gradient
- Applies two 3x3 kernels to estimate horizontal (Gx) and vertical (Gy) gradients.
- Magnitude G = sqrt(Gx^2 + Gy^2) highlights regions of rapid intensity change.
- OpenCV: `cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)`

---

### 4.6 Otsu's Thresholding
- **Goal**: Automatically find the optimal global threshold T that separates foreground (defect) from background.
- **Method**: Maximises inter-class variance across all possible threshold values.
- **Formula**: `sigma^2_between(T) = w0 * w1 * (mu0 - mu1)^2` — maximise over all T.
- Result: A single optimal threshold without any manual tuning.
- OpenCV: `cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)`

---

### 4.7 Adaptive Thresholding
- Otsu fails on images with non-uniform lighting.
- Adaptive thresholding computes a different threshold for each pixel based on a local neighbourhood (e.g., 11x11 block).
- Two methods: Mean or Gaussian weighted.
- OpenCV: `cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)`

---

### 4.8 Morphological Operations

| Operation | Effect | Definition |
|---|---|---|
| **Erosion** | Shrinks white regions, removes small protrusions | Pixel is 1 only if all pixels under kernel are 1 |
| **Dilation** | Expands white regions, fills small holes | Pixel is 1 if any pixel under kernel is 1 |
| **Opening** | Erosion then Dilation — removes small noise blobs | `(A erode B) dilate B` |
| **Closing** | Dilation then Erosion — fills gaps in regions | `(A dilate B) erode B` |

The structuring element B is a small kernel (e.g., circular or rectangular 3x3, 5x5, etc.).

---

### 4.9 GLCM — Gray-Level Co-occurrence Matrix
- A 2D matrix where entry P(i, j, d, theta) = number of times pixel with intensity i co-occurs with pixel intensity j at distance d and angle theta.
- **Features extracted from GLCM**:
  - **Contrast**: `sum((i-j)^2 * P(i,j))` — local intensity variation
  - **Correlation**: How correlated a pixel is with its neighbour
  - **Energy (ASM)**: `sum(P(i,j)^2)` — uniformity/smoothness
  - **Homogeneity**: `sum(P(i,j) / (1 + |i-j|))` — closeness of distribution to diagonal
- Why: Healthy surfaces have consistent GLCM features. Defective regions show statistical deviation.
- scikit-image: `skimage.feature.graycomatrix`, `skimage.feature.graycoprops`

---

### 4.10 LBP — Local Binary Pattern
- For each pixel, compare its intensity with 8 surrounding neighbours. Write 1 if neighbour >= centre, else 0. Read the 8 bits as a binary number — this is the LBP code.
- The LBP histogram of a region captures the micro-texture pattern.
- Rotation-invariant, scale-invariant, computationally cheap.
- Normal surface = predictable LBP histogram. Defects break the pattern.
- scikit-image: `skimage.feature.local_binary_pattern`

---

### 4.11 Local Standard Deviation Map
- For each pixel, compute the standard deviation within a small neighbourhood window.
- Produces a "texture energy" heatmap: smooth surfaces = low std, scratches/defects = high std.
- Used to build the texture anomaly detection heatmap.

---

### 4.12 Contour Detection & Region Analysis
- After thresholding and morphological cleanup, the binary mask contains white blobs = candidate defects.
- `cv2.findContours` traces the boundary of each blob (Suzuki-Abe algorithm).
- Per contour: area, perimeter, bounding rectangle, centroid.
- `cv2.connectedComponentsWithStats` gives connected component labelling and count of distinct defect regions.

---

## 5. EVALUATION METRICS

### 5.1 Confusion Matrix Elements

For binary defect classification (defect vs no-defect):

| | Predicted Defect | Predicted OK |
|---|---|---|
| **Actual Defect** | TP (True Positive) | FN (False Negative) |
| **Actual OK** | FP (False Positive) | TN (True Negative) |

---

### 5.2 Key Metrics

| Metric | Formula | Meaning |
|---|---|---|
| **Precision** | TP / (TP + FP) | Of all flagged defects, how many were real? |
| **Recall (Sensitivity)** | TP / (TP + FN) | Of all actual defects, how many were caught? |
| **F1-Score** | 2 * (P * R) / (P + R) | Harmonic mean — balances Precision and Recall |
| **Accuracy** | (TP + TN) / Total | Overall correct predictions |
| **IoU (Jaccard Index)** | TP / (TP + FP + FN) | Overlap between predicted and actual defect mask |
| **Specificity** | TN / (TN + FP) | How well are non-defects correctly cleared? |

### 5.3 Why F1 over Accuracy?
Defect datasets are imbalanced — most pixels are defect-free. A model saying "no defect" on everything achieves 95%+ accuracy but catches zero real defects. F1-Score handles imbalance by focusing on defect-class performance.

### 5.4 Why IoU?
IoU measures spatial overlap at the pixel level. Essential for segmentation tasks — even if a defect is detected, the mask must closely match the actual defect region.

---

## 6. FRONTEND: REACT + TAILWIND UI

### 6.1 React (v19)
- **Component-based architecture**: Each UI panel (Navbar, Hero, Inspector, Pipeline, Metrics, Export) is an isolated, reusable JSX component.
- **Hooks used**: useState, useEffect, useRef, useCallback — for state management, lifecycle, DOM refs, and memoization.
- **Virtual DOM**: React maintains a virtual copy of the DOM; on state change it diffs and only updates changed nodes (reconciliation) — very efficient for animated dashboards.

### 6.2 Vite (Build Tool)
- Ultra-fast HMR (Hot Module Replacement) — changes reflect in browser in under 50ms.
- ES module-based dev server (no bundling in dev mode).
- Production `npm run build` produces optimised, tree-shaken, minified bundles with code splitting.

### 6.3 Tailwind CSS v4
- Utility-first CSS: compose styles directly in JSX via class names.
- Zero runtime: Tailwind scans all JSX at build time and emits only the used CSS utilities (tree shaking).
- v4 uses a Vite plugin (`@tailwindcss/vite`) instead of PostCSS — tighter integration, faster builds.

### 6.4 Framer Motion
- Declarative animation library for React.
- Key concepts used:
  - `motion.div` — wraps any element to make it animatable.
  - `initial / animate / exit` — define from/to animation states.
  - `whileHover / whileTap` — gesture-based micro-interactions.
  - `variants + staggerChildren` — cascading staggered reveal of lists.
  - `AnimatePresence` — animates components on mount/unmount (page transitions).
  - `useScroll + useTransform` — scroll-linked parallax effects.

---

## 7. UI DESIGN CONCEPTS & LIBRARIES

### 7.1 Glassmorphism
- **Technique**: Translucent panels with backdrop-filter: blur() giving a frosted glass appearance.
- CSS properties: `background: rgba(255,255,255,0.05)`, `backdrop-filter: blur(20px)`, `border: 1px solid rgba(255,255,255,0.1)`, `box-shadow: 0 8px 32px rgba(0,0,0,0.3)`.
- Inspired by: Aceternity UI glassmorphism card patterns.

### 7.2 Aurora / Fluid Gradient Background
- CSS @keyframes animation that shifts a multi-stop gradient over time.
- Layers multiple blurred, offset gradient orbs (violet, electric blue, cyan) to create the fluid aurora effect.
- Gradient background-position animated on a looping 12–18s cycle.

### 7.3 Interactive Particle Canvas
- Pure HTML5 Canvas + JavaScript (no external library).
- Particle system: Creates ~120 particles, each with position (x, y), velocity (vx, vy), size, and opacity.
- Mouse repulsion: On mousemove, calculates distance to each particle. If within threshold, applies a repulsion force vector.
- Constellation lines: If distance between two particles < 100px, draws a semi-transparent connecting line.
- Animation loop: requestAnimationFrame drives the render loop at 60fps.

### 7.4 Border Beam (Magic UI)
- A glowing highlight that travels around the border of a card.
- Implemented with CSS @keyframes rotating a bright arc around the border.
- Creates a "scanning" / "active" visual feel on metric cards.

### 7.5 Spotlight Effect (Aceternity UI)
- A radial gradient that follows the mouse cursor, creating a spotlight glow.
- Tracks mousemove and updates CSS variables --spotlight-x / --spotlight-y to drive gradient position.

### 7.6 Design System
| Token | Value |
|---|---|
| Primary colour | #7c3aed (violet-700) |
| Accent | #06b6d4 (cyan-500) |
| Electric Blue | #3b82f6 (blue-500) |
| Background | #020617 (slate-950) |
| Surface | rgba(255,255,255,0.05) |
| Font | Inter (Google Fonts) |

---

## 8. PROJECT ARCHITECTURE

```
ZeroDefect/
|
|-- app.py                     # Streamlit main application
|
|-- src/                       # Python backend modules
|   |-- preprocessing.py       # Grayscale, Gaussian blur, CLAHE
|   |-- edge_detection.py      # Canny, Sobel, Auto-Canny
|   |-- segmentation.py        # Otsu, Adaptive threshold, morphology
|   |-- texture_features.py    # GLCM, LBP, local std map
|   |-- defect_detection.py    # Classical pipeline + anomaly detection
|   |-- evaluation.py          # F1, IoU, Precision, Recall metrics
|   `-- visualization.py       # Annotated image rendering, heatmaps
|
|-- frontend/                  # React + Vite frontend
|   `-- src/
|       |-- App.jsx            # Root component + routing
|       |-- components/
|       |   |-- Navbar.jsx           # Navigation bar
|       |   |-- Hero.jsx             # Landing hero section
|       |   |-- InspectorView.jsx    # Upload + telemetry
|       |   |-- PipelineView.jsx     # 5-stage pipeline visual
|       |   |-- MetricsView.jsx      # Quality metrics dashboard
|       |   `-- ExportView.jsx       # Export console
|       |-- index.css          # Global design tokens
|       `-- main.jsx           # React entry point
|
|-- reports/                   # Generated reports
|-- data/                      # Sample images
|-- tests/                     # pytest test suite
`-- requirements.txt           # Python dependencies
```

---

## 9. VIVA Q&A

### SECTION A: Image Processing Fundamentals

**Q1. Why do you convert to grayscale first?**
> Colour adds 3x the data volume without helping for texture/edge analysis. Defects show as intensity anomalies, not colour differences. Grayscale simplifies computation and focuses the algorithm on luminance patterns.

**Q2. What is CLAHE and why use it over standard histogram equalization?**
> Standard HE redistributes global histogram — in images with both dark and bright areas, it over-amplifies noise in flat regions. CLAHE divides the image into tiles (e.g. 8x8), equalises each tile independently, and clips the histogram to a limit to prevent noise amplification. This gives better local contrast enhancement for subtle surface defects.

**Q3. What is the difference between Canny and Sobel edge detection?**
> Sobel computes gradient magnitude — it gives a "thick" edge map. Canny builds on top of Sobel but adds non-maximum suppression (thinning edges to 1 pixel) and hysteresis thresholding (connecting weak edges to strong ones). Canny produces cleaner, thinner, more connected edges.

**Q4. Explain Otsu's thresholding.**
> Otsu's method automatically finds the optimal global threshold by maximising the inter-class variance (between foreground and background) across all possible threshold values 0–255. It assumes the histogram is bimodal (two peaks). No manual parameter tuning is required.

**Q5. When would adaptive thresholding outperform Otsu?**
> When the image has non-uniform illumination (shadows, vignetting, uneven lighting). Otsu applies one threshold globally — if lighting varies, some regions get mis-thresholded. Adaptive thresholding computes a different threshold per local neighbourhood, handling the variation gracefully.

**Q6. What are morphological opening and closing operations?**
> Opening = Erosion followed by Dilation. It removes small noise blobs (objects smaller than the structuring element). Closing = Dilation followed by Erosion. It fills small holes/gaps inside detected defect regions. Both preserve the shape and size of larger objects.

**Q7. What does a GLCM represent?**
> A Gray-Level Co-occurrence Matrix (GLCM) counts how often pairs of pixel intensities (i, j) co-occur at a given spatial offset (distance d, angle theta). Features like contrast, correlation, energy, and homogeneity extracted from GLCM characterise the surface texture. Defective regions show different GLCM statistics than healthy ones.

**Q8. What is LBP and what makes it useful?**
> Local Binary Pattern encodes the micro-texture of a neighbourhood by thresholding 8 surrounding pixels against the centre pixel. The resulting 8-bit code and its histogram are rotation-invariant texture descriptors. They are very fast to compute (no floating point) and effective at distinguishing texture patterns.

---

### SECTION B: Metrics & Evaluation

**Q9. Why is accuracy a bad metric for defect detection?**
> Defect images are highly imbalanced — most pixels are defect-free. A naive classifier that always says "no defect" achieves high accuracy (e.g., 97%) but catches zero actual defects. F1-Score and IoU are better because they focus on the minority (defect) class performance.

**Q10. Explain IoU (Intersection over Union).**
> IoU = Area of overlap between predicted mask and ground-truth mask divided by Area of union of both masks. Range: 0 (no overlap) to 1 (perfect match). For segmentation tasks, IoU measures spatial accuracy — not just whether a defect was detected, but whether the exact defect region was correctly localised.

**Q11. What is the F1-Score?**
> F1 = 2 * (Precision * Recall) / (Precision + Recall). It is the harmonic mean of Precision (how many flagged defects are real) and Recall (how many real defects were caught). Harmonic mean punishes extreme imbalance — a model with 100% recall but 1% precision gets F1 of approximately 2%, not 50%.

**Q12. What does high recall but low precision mean in this context?**
> High recall = most actual defects are caught (good). Low precision = many false alarms (non-defective regions flagged as defective). In manufacturing, you might prefer high recall (miss no defects) even at the cost of some false positives that get re-inspected.

---

### SECTION C: System Design

**Q13. Why use Streamlit for the web interface?**
> Streamlit lets you build interactive web apps purely in Python, with zero HTML/CSS/JS required. It auto-rerenders on widget interaction, making it ideal for data science dashboards where the backend is Python. It integrates natively with NumPy, Pandas, OpenCV, and Plotly.

**Q14. Why did you also build a React frontend?**
> Streamlit's UI is functional but not premium-looking. React + Tailwind CSS + Framer Motion allows complete design control — glassmorphism cards, aurora backgrounds, interactive particles, smooth animations — giving the project a production-quality, enterprise-grade look for demonstrations and presentations.

**Q15. What is Framer Motion and how is it used here?**
> Framer Motion is a declarative animation library for React. It is used for: scroll-triggered reveals (elements fade/slide in when they enter the viewport), staggered text animations (letters/words animate in sequence), hover micro-interactions (scale, glow effects), and page transition animations (AnimatePresence mounts/unmounts with animation).

**Q16. Explain the interactive particle canvas effect.**
> An HTML5 Canvas element covers the background. A JavaScript particle system creates ~120 particles with random positions and velocities. The animation loop (requestAnimationFrame) updates positions each frame. On mousemove, distance to each particle is computed — if within 120px, a repulsion force pushes the particle away from the cursor. Particles within 100px of each other are connected with a faint line, creating a constellation network. This creates a dynamic, interactive background that responds to user movement.

**Q17. What is glassmorphism?**
> A design trend that simulates frosted glass: semi-transparent panels (background: rgba(255,255,255,0.05)), with backdrop-filter: blur(20px) which blurs whatever is behind the panel, a subtle light border (border: 1px solid rgba(255,255,255,0.1)), and a soft drop shadow. This creates depth and layering without heavy opaque panels.

---

### SECTION D: Advanced / Tricky Questions

**Q18. How does your texture anomaly detection work?**
> It builds a statistical model of "normal" surface texture using reference images. For each image patch, it extracts GLCM features and/or local standard deviation maps. These are compared against the normal distribution (mean + std). Regions that deviate beyond a threshold (e.g., more than 2 standard deviations) are flagged as anomalous. This is an unsupervised approach — no labelled defect images needed.

**Q19. What are the limitations of classical image processing for defect detection?**
> 1. Sensitive to lighting changes — requires consistent illumination setup.
> 2. Threshold parameters may need tuning per material type.
> 3. Cannot learn from labelled data the way deep learning can.
> 4. May struggle with very subtle texture defects or camouflaged anomalies.
> 5. No generalisation — parameters tuned for steel may not work for fabric.

**Q20. How would you scale this to a real production line?**
> 1. Replace Streamlit with a FastAPI backend + proper database (PostgreSQL).
> 2. Replace classical pipeline with a fine-tuned deep learning model (e.g., UNet for segmentation).
> 3. Deploy on GPU-accelerated cloud (e.g., GCP with NVIDIA T4).
> 4. Real-time camera integration via RTSP stream + OpenCV VideoCapture.
> 5. Alert system (Slack/email) when defect rate exceeds threshold.
> 6. CI/CD pipeline for model retraining as new defect patterns emerge.

---

## 10. SLIDE GUIDE

**Slide 1 — Title Slide**
- Title: "ZeroDefect: Automated Surface Defect Detection"
- Subtitle: "Classical Image Processing + Modern Interactive Dashboard"
- Your name, course, date

**Slide 2 — Problem Statement**
- Quality control in manufacturing is manual: slow, inconsistent, expensive
- Show a photo of a defective surface
- State the goal: Detect, localise, and quantify surface defects automatically

**Slide 3 — System Overview**
- Show the full pipeline diagram (8 stages, from the Architecture section above)
- Highlight: Input -> 8 Processing Stages -> Output (annotated image + metrics)

**Slide 4 — Preprocessing (Stage 1)**
- Show before/after images: original -> grayscale -> CLAHE enhanced
- Explain each step briefly
- Mention why CLAHE > standard equalization

**Slide 5 — Edge Detection (Stage 2)**
- Side-by-side: original image, Sobel result, Canny result
- Explain: Canny gives clean thin edges via NMS + hysteresis
- Show Auto-Canny with adaptive thresholds

**Slide 6 — Segmentation & Morphology (Stages 3–4)**
- Show: Otsu mask vs Adaptive mask
- Show: Before/after Opening and Closing
- Explain: Opening removes noise, Closing fills defect region gaps

**Slide 7 — Texture Analysis (Stage 5)**
- GLCM feature table (contrast, energy, homogeneity)
- LBP visualization — show LBP image and histogram
- Local standard deviation heatmap overlay

**Slide 8 — Defect Localisation (Stage 7)**
- Show annotated output image with coloured bounding boxes
- Show defect mask / segmentation overlay
- Mention: cv2.findContours, connected component labelling

**Slide 9 — Evaluation Metrics (Stage 8)**
- Show confusion matrix
- Table: Precision, Recall, F1, IoU, Accuracy
- Explain why F1 and IoU matter more than accuracy for this task

**Slide 10 — Frontend: React Dashboard**
- Screenshot of the React app with aurora background
- Highlight glassmorphism cards, animated metrics, pipeline visualization
- Mention: React, Tailwind CSS, Framer Motion

**Slide 11 — Design Highlights**
- Aurora background (CSS keyframes)
- Interactive particle canvas (HTML5 Canvas + mouse repulsion)
- Glassmorphism panels (backdrop-blur)
- Framer Motion animations (scroll reveal, stagger, hover)

**Slide 12 — Technology Stack Summary**
| Layer | Technology |
|---|---|
| Image Processing | OpenCV, NumPy, scikit-image |
| Metrics | scikit-learn |
| Backend UI | Streamlit + Plotly |
| Frontend | React 19 + Vite 8 |
| Styling | Tailwind CSS v4 |
| Animation | Framer Motion |
| Effects | HTML5 Canvas API |

**Slide 13 — Limitations & Future Work**
- Limitations: lighting sensitivity, parameter tuning per material
- Future: Deep learning (UNet), GPU inference, real-time camera feed, FastAPI backend

**Slide 14 — Demo**
- Live demo of the Streamlit application
- Upload a sample image, show the full pipeline output
- Show metrics, annotated image, export CSV

**Slide 15 — Q&A**
- "Thank you. Questions?"

---

## QUICK REFERENCE: FORMULAS TO MEMORISE

| Concept | Formula |
|---|---|
| Grayscale | Gray = 0.299R + 0.587G + 0.114B |
| Sobel magnitude | G = sqrt(Gx^2 + Gy^2) |
| Otsu inter-class variance | w0 * w1 * (mu0 - mu1)^2 — maximise |
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1-Score | 2 * P * R / (P + R) |
| IoU | TP / (TP + FP + FN) |
| GLCM Contrast | sum((i-j)^2 * P(i,j)) |
| GLCM Energy | sum(P(i,j)^2) |

---

*Document prepared for ZeroDefect project viva and presentation.*
*All formulas, concepts, and Q&A are based on the actual implemented codebase.*
