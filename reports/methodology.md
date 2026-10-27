# ZeroDefect — Methodology

## 1. Problem Statement

Manufacturing quality control requires inspection of product surfaces for defects such as scratches, cracks, dents, and contamination. Manual visual inspection is slow, inconsistent, and error-prone at scale. This project explores how classical digital image processing and texture analysis can automate the detection and localization of suspicious surface regions.

---

## 2. System Architecture

```
Input Image
     │
     ▼
[Stage 1] Image Acquisition
  - Load & validate
  - Resize if needed
  - Display metadata
     │
     ▼
[Stage 2] Preprocessing
  - Grayscale conversion
  - Gaussian / Median blur
  - CLAHE contrast enhancement
     │
     ▼
[Stage 3] Edge Detection
  - Canny (auto-threshold)
  - Sobel gradient magnitude
     │
     ▼
[Stage 4] Thresholding & Segmentation
  - Otsu's global threshold
  - Adaptive (local) threshold
  - Combined edge + intensity mask
     │
     ▼
[Stage 5] Morphological Processing
  - Opening (remove noise)
  - Closing (fill gaps)
  - Min-area component filter
     │
     ▼
[Stage 6] Texture Feature Extraction
  - GLCM: contrast, homogeneity, energy, correlation
  - LBP: local binary pattern histograms
  - Local std-dev map (anomaly signal)
     │
     ▼
[Stage 7] Defect Localization
  Approach A (Classical):  edge + threshold + morphology → contours
  Approach B (Texture):    fit normal baseline → z-score anomaly map
     │
     ▼
[Stage 8] Region Analysis
  - Connected component labelling
  - Bounding boxes, areas, centroids, perimeters
  - Overlay generation
     │
     ▼
[Stage 9] Evaluation
  - Pixel-level: Precision, Recall, F1, IoU
  - Image-level: Accuracy, AUROC (if score maps available)
  - CSV export
```

---

## 3. Algorithm Details

### 3.1 Canny Edge Detection

**Purpose**: Identify intensity transitions that may correspond to defect boundaries.

**Steps**:
1. Gaussian smoothing (kernel σ auto-determined from image median)
2. Gradient computation (Sobel in x and y directions)
3. Non-maximum suppression along gradient direction
4. Double thresholding: strong edges (>high), weak edges (low–high), discard (<low)
5. Edge tracking by hysteresis: keep weak edges connected to strong edges

**Auto-threshold selection** (implemented in `auto_canny`):
```
v    = median(image)
low  = max(0,   v * (1 - σ))     # σ = 0.33
high = min(255, v * (1 + σ))
```

**Limitation**: Normal textured surfaces (e.g., grid, leather grain) also produce edges. Edges alone are not sufficient to confirm a defect.

---

### 3.2 Thresholding

**Otsu's Method**:  
Minimizes intra-class variance (or equivalently maximizes inter-class variance) to find the single optimal global threshold. Best for bimodal histograms.

**Adaptive Thresholding**:  
Computes a local threshold for each pixel based on the weighted mean of its neighborhood (Gaussian-weighted or simple mean). Handles uneven illumination better than Otsu's.

**Combined mask**:  
`mask = edge_map OR intensity_mask`  
This reduces false negatives by capturing both edge-based and intensity-based anomalies.

---

### 3.3 Morphological Operations

| Operation | Formula | Purpose |
|---|---|---|
| Erosion | A ⊖ B | Shrink foreground; remove thin protrusions and noise |
| Dilation | A ⊕ B | Expand foreground; connect nearby fragments |
| Opening | (A ⊖ B) ⊕ B | Remove small noise blobs |
| Closing | (A ⊕ B) ⊖ B | Fill holes inside defect regions |

Structuring element: rectangular kernel, configurable size.

---

### 3.4 GLCM Features

The **Gray-Level Co-occurrence Matrix** (GLCM) P(i, j | d, θ) counts the number of times pixel pairs at distance d and angle θ have intensities i and j.

| Feature | Formula | Interpretation |
|---|---|---|
| Contrast | Σ (i-j)² P(i,j) | Intensity variation between neighbors. High → edges/defects |
| Homogeneity | Σ P(i,j) / (1 + \|i-j\|) | Closeness to diagonal. Low → rough texture |
| Energy | Σ P(i,j)² | Textural uniformity. High → repeated pattern |
| Correlation | Σ (i-μᵢ)(j-μⱼ)P(i,j) / (σᵢσⱼ) | Statistical dependency. High → regular pattern |

---

### 3.5 Local Standard Deviation Map

For each pixel (x, y), compute the standard deviation of intensities in a W×W neighborhood:

```
local_std(x, y) = sqrt(E[I²] - E[I]²)  over window W×W
```

Implemented efficiently using `cv2.boxFilter` for E[I] and E[I²].

**Anomaly scoring**: A pixel is flagged if its z-score exceeds a threshold:
```
z(x, y) = (local_std(x,y) - mean_normal_std) / std_normal_std
flag if z > z_threshold
```

---

### 3.6 Evaluation Metrics

#### Pixel-level

| Metric | Formula |
|---|---|
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| F1-Score | 2 × P × R / (P + R) |
| IoU (Jaccard) | TP / (TP + FP + FN) |

Where:
- **TP**: pixels predicted defective that are truly defective
- **FP**: pixels predicted defective that are actually normal
- **FN**: truly defective pixels that were missed

#### AUROC
Area Under the ROC Curve — measures discrimination ability across all thresholds. 0.5 = random, 1.0 = perfect.

---

## 4. Implementation Notes

- All functions are self-contained with no global state.
- Fixed random seed (`np.random.seed(42)`) used in demo image generation.
- Paths are configurable; no hardcoded absolute paths in src/.
- OpenCV BGR→RGB conversion happens immediately at image load time.
- Streamlit `matplotlib.use("Agg")` prevents GUI conflicts.
- `clean_mask()` chains opening → closing → small-component removal in one call.

---

## 5. Known Limitations

1. **Patterned surfaces**: Regular textures (grid, weave) produce many edges that may be misclassified as defects without additional context.
2. **Lighting variation**: Uneven illumination can cause false positives even with CLAHE applied.
3. **Threshold sensitivity**: Optimal thresholds differ between surface categories; no single threshold works universally.
4. **Small defects**: Defects smaller than `min_defect_area` pixels are discarded to reduce noise, at the cost of missing fine surface cracks.
5. **No physical calibration**: Region areas are reported in pixels only. Millimetre-scale measurements require camera calibration.
6. **Texture baseline requires clean training data**: TextureAnomalyDetector degrades if training images contain defects.
