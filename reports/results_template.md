# ZeroDefect — Results Report Template

**Instructions**: Fill in this template with your actual experimental results.  
Do NOT leave placeholder values in a submitted report.  
All numbers must come from real code execution.

---

## 1. Experimental Setup

| Parameter | Value |
|---|---|
| Dataset | MVTec AD — Category: ___________ |
| Training images (normal) | ___ images |
| Test images (normal) | ___ images |
| Test images (defective) | ___ images |
| Defect types tested | _________________________ |
| Image resolution | ___×___ pixels (after resize) |
| Preprocessing | Gaussian blur k=___, CLAHE clip=___ |
| Edge method | _________________________ |
| Thresholding | _________________________ |
| Morphology open kernel | ___ |
| Morphology close kernel | ___ |
| Min defect area | ___ px |
| z-threshold (Approach B) | ___ |

---

## 2. Qualitative Results

### Example: Normal Image
*(Insert screenshot of original + predicted mask — mask should be mostly empty)*

| Stage | Description |
|---|---|
| Original | Uniform surface texture, no visible defects |
| Predicted mask | [Mostly empty / minor noise / describe] |
| Verdict | No suspicious region detected |

### Example: Defective Image (type: ___)
*(Insert screenshot of original + overlay)*

| Stage | Description |
|---|---|
| Original | [Describe what defect looks like visually] |
| Edge map | [What edges appear; which correspond to defect vs. texture] |
| Predicted mask | [Does it cover the defect? Any false positives?] |
| Verdict | Potential surface anomaly detected |

---

## 3. Quantitative Results — Approach A (Classical)

### Per-Image Results

| Image | GT Label | Pred Label | Precision | Recall | F1 | IoU |
|---|---|---|---|---|---|---|
| test/good/000 | Normal | _______ | — | — | — | — |
| test/good/001 | Normal | _______ | — | — | — | — |
| test/defect1/000 | Defective | _______ | ___ | ___ | ___ | ___ |
| test/defect1/001 | Defective | _______ | ___ | ___ | ___ | ___ |
| *(continue)* | | | | | | |
| **MEAN** | | | **___** | **___** | **___** | **___** |

### Summary Metrics

| Metric | Approach A (Classical) |
|---|---|
| Mean Pixel Precision | |
| Mean Pixel Recall | |
| Mean Pixel F1-Score | |
| Mean Pixel IoU | |
| Image-Level Accuracy | |
| Image-Level AUROC | |

---

## 4. Quantitative Results — Approach B (Texture Anomaly)

| Metric | Approach B (Texture) |
|---|---|
| Mean Pixel Precision | |
| Mean Pixel Recall | |
| Mean Pixel F1-Score | |
| Mean Pixel IoU | |
| Image-Level Accuracy | |
| Image-Level AUROC | |

---

## 5. Approach Comparison

| Metric | Approach A | Approach B | Winner |
|---|---|---|---|
| Pixel F1 | | | |
| Pixel IoU | | | |
| Image Accuracy | | | |
| Speed (approx) | | | |
| False Positive Rate | | | |

---

## 6. Ablation Study

*(Compare configurations using the same test images)*

| Configuration | Pixel F1 | Pixel IoU | Notes |
|---|---|---|---|
| Without CLAHE | | | |
| With CLAHE | | | |
| Without morphology | | | |
| With morphology (current) | | | |
| Open kernel=3 vs kernel=9 | | | |

---

## 7. Failure Cases

### Case 1: False Positives on Textured Surface
*(Describe what happened; insert image if possible)*

**Root cause**: [e.g., grid pattern produces many edges that survive thresholding]  
**Potential fix**: [e.g., increase min_area, adjust threshold per-category]

### Case 2: Missed Defect (False Negative)
*(Describe what happened)*

**Root cause**: [e.g., low-contrast scratch not detected after CLAHE]  
**Potential fix**: [e.g., stronger CLAHE, lower Canny thresholds]

### Case 3: Invalid Input
*(Test with a non-image file — describe error message received)*

---

## 8. Discussion

*(Write 200–400 words covering:)*
- Which approach performed better and why
- Where the system failed and what caused the failure
- How results might improve with more training data or deep learning
- Deployment readiness and production verification assessment

---

## 9. Conclusion

*(Write 100–200 words summarizing what was achieved, what was learned, and what comes next)*
