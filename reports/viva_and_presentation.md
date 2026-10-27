# ZeroDefect — Viva Questions & Presentation Guide

---

## Presentation Outline (8–10 Slides)

| Slide | Title | Content |
|---|---|---|
| 1 | **Title** | ZeroDefect, your name, date, institution |
| 2 | **Problem & Motivation** | Why surface defects matter; cost of missed defects in industry |
| 3 | **Project Objectives** | What the system does; what it does NOT claim |
| 4 | **Pipeline Overview** | Diagram of 9 stages from acquisition to evaluation |
| 5 | **Dataset** | MVTec AD, category used, normal vs. defective split, GT masks |
| 6 | **Approach A — Classical** | Preprocessing → Canny → Threshold → Morphology visuals |
| 7 | **Approach B — Texture** | GLCM features, local std-dev map, z-score anomaly scoring |
| 8 | **Results & Metrics** | Screenshots, confusion matrix, F1/IoU table |
| 9 | **Limitations & Failure Cases** | Honest discussion; images where it fails |
| 10 | **Conclusion & Future Work** | Summary; potential improvements (deep learning, multi-scale) |

---

## Viva Questions and Model Answers

### 🔵 Image Processing Fundamentals

**Q: What is CLAHE and why is it better than regular histogram equalization?**  
A: CLAHE — Contrast Limited Adaptive Histogram Equalization — works on small tiles (e.g., 8×8) independently rather than the whole image at once. It clips the contrast amplification factor to prevent noise blow-up in already-uniform regions. Regular histogram equalization applies a single global transform that can over-amplify noise in flat areas. CLAHE makes subtle surface variations visible while controlling noise, which is critical for detecting fine defects like shallow scratches.

**Q: Why do we convert images to grayscale before processing?**  
A: Most defects on manufactured surfaces manifest as intensity (brightness) changes rather than colour changes. Grayscale reduces the dimensionality from 3 channels to 1, simplifying processing and reducing computation time. Our edge detection, thresholding, and GLCM features all operate on intensity values. If the dataset contained colour-based defects (e.g., rust staining), we would need to retain colour information.

**Q: What is the Canny edge detection algorithm and how does it work?**  
A: Canny is a multi-stage edge detector. First, Gaussian smoothing removes noise. Then Sobel operators compute the gradient magnitude and direction. Non-maximum suppression thins edges to one pixel by suppressing gradient values that are not local maxima in the gradient direction. Double thresholding classifies gradients as strong (>high threshold), weak (between thresholds), or discarded (<low). Hysteresis keeps weak edges connected to strong edges and discards isolated weak ones.

**Q: What is Otsu's thresholding?**  
A: Otsu's method finds the threshold that minimizes the weighted sum of within-class variances for two classes (foreground/background). Equivalently, it maximizes between-class variance. The algorithm scans all possible threshold values (0–255) and picks the one producing the most separated groups. It is automatic but assumes a bimodal histogram — it degrades on images with uneven lighting.

**Q: What does morphological opening do?**  
A: Opening is erosion followed by dilation using the same structuring element. Erosion shrinks foreground regions, removing thin protrusions and small noise blobs. Dilation restores the size of larger regions. The net effect: small isolated noise blobs (smaller than the kernel) are removed, while larger genuine defect regions are preserved.

---

### 🟡 Texture Analysis

**Q: What is a GLCM and what does the contrast feature measure?**  
A: A Gray-Level Co-occurrence Matrix (GLCM) counts how often pairs of pixel values co-occur at a specified spatial relationship (distance d, angle θ). For a 256-grey-level image, the GLCM is a 256×256 matrix where P(i,j) = number of times a pixel with value i has a neighbor with value j at distance d and angle θ. The contrast feature = Σ(i−j)² × P(i,j), measuring the intensity variation between neighbors. High contrast indicates large differences between adjacent pixels, suggesting edges or defect boundaries.

**Q: What are Local Binary Patterns?**  
A: LBP encodes each pixel as a binary number by comparing it to its circular neighbors. For each neighbor: if brighter than the center → 1, else → 0. The resulting binary sequence (e.g., 00001111 for 8 neighbors) is the LBP code for that pixel. "Uniform" LBPs have at most 2 bit transitions and are the most informative. The LBP image histogram summarizes the texture — different surface textures produce different distributions of LBP codes. Comparing histograms between normal and test patches detects texture anomalies.

**Q: How does the local standard deviation map work as an anomaly detector?**  
A: For each pixel, we compute the standard deviation of its W×W neighborhood. A uniform defect-free surface has low local std throughout. A scratch or contamination patch introduces sharp intensity changes within a neighborhood, producing high local std. We establish a baseline by computing this map for several normal images, recording their per-pixel mean and standard deviation. At test time, we compute a z-score: z = (test_std − normal_mean) / normal_std. Pixels with z > threshold are flagged as anomalous.

---

### 🔴 Evaluation Metrics

**Q: Explain Precision and Recall and the trade-off between them.**  
A: Precision = TP/(TP+FP): Of all pixels we flagged, how many were truly defective? High precision = few false alarms. Recall = TP/(TP+FN): Of all truly defective pixels, how many did we find? High recall = few missed defects. There is a trade-off: lowering the detection threshold increases recall (we catch more defects) but decreases precision (we also produce more false alarms). The F1-score balances both.

**Q: What is IoU and why is it preferred for segmentation tasks?**  
A: IoU (Intersection over Union), also called Jaccard index = TP/(TP+FP+FN). It measures the overlap between predicted and ground-truth masks as a fraction of their union. Unlike Accuracy, IoU penalizes both false positives and false negatives explicitly. For segmentation where defective pixels are sparse, accuracy can be misleadingly high (predicting everything as normal gives high accuracy when defects are rare). IoU focuses specifically on the defective region.

**Q: What is AUROC and why does it not require a threshold?**  
A: AUROC = Area Under the Receiver Operating Characteristic Curve. The ROC curve plots True Positive Rate (Recall) vs. False Positive Rate (FP/(FP+TN)) at all possible thresholds. AUROC is the area under this curve. A value of 0.5 means the model is no better than random; 1.0 means perfect discrimination. AUROC is threshold-independent, making it useful for comparing continuous anomaly score maps without committing to a specific binarization threshold.

---

### 🟢 Dataset and Experimental Design

**Q: Why must ground-truth masks be kept separate from the detection pipeline?**  
A: Ground-truth masks represent the correct answers. If we allowed the pipeline to "see" these masks during processing, we would be training on test data — an invalid experiment. The pipeline must generate its predictions independently; GT masks are only used afterwards to compute evaluation metrics.

**Q: What is the difference between image-level and pixel-level evaluation?**  
A: Image-level evaluation classifies each image as "defective" or "normal" (binary classification). Pixel-level evaluation goes further — it compares the predicted defect mask with the ground-truth mask pixel by pixel. A system that correctly identifies an image as defective but highlights the wrong region would score well at image level but poorly at pixel level. Pixel-level metrics (F1, IoU) are more demanding and more informative for localization tasks.

**Q: What are the main limitations of this system?**  
A: 1) Patterned or textured surfaces produce many edges that can be mistaken for defects. 2) Optimal thresholds vary per surface category — no single setting works universally. 3) Very fine or low-contrast defects may be lost during morphological cleanup (min_area filter). 4) The texture anomaly detector requires clean normal training images. 5) No physical calibration — areas are in pixels, not millimeters. 6) Results depend heavily on preprocessing quality (CLAHE, blur settings).

---

### 🟣 Software Engineering

**Q: Why is your preprocessing organized into separate functions?**  
A: Modularity: each function does one job (e.g., apply_clahe does only CLAHE). This makes testing easier (each function has its own unit test), allows reconfiguration without modifying other code, and makes the code readable for beginners. Functions return intermediate results rather than printing them, so they can be used in both the Streamlit app and the Colab notebook without modification.

**Q: How did you verify your pipeline works correctly?**  
A: Three levels of verification: 1) Unit tests (pytest) check each function's output shape, dtype, and correctness against synthetic images with known properties (e.g., a scratch should produce detectable edges). 2) Visual inspection of each pipeline stage in the interface. 3) Metric computation against ground-truth masks on real dataset images. Any pipeline bug that produces wrong mask shapes or NaN values would be caught at the unit test level.

---

## Possible Difficult Questions from Examiners

1. Why might your system fail on leather but work on grid?
2. How would you extend this to video inspection?
3. What would you do if the dataset has very few defective examples?
4. How does your approach compare to using a pre-trained neural network like PatchCore?
5. What does a confusion matrix show that individual metrics don't?
6. If recall is 0.95 but precision is 0.1, is your system useful? Why or why not?
7. Why do you use `min_area` to filter small components? What is the risk?
8. Why does Gaussian blur help edge detection?
9. What happens to GLCM computation time for a 2000×2000 image?
10. How would you validate that your normal training images are truly defect-free?
