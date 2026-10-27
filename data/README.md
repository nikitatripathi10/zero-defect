# ZeroDefect — Data Directory

## Purpose

This directory holds the dataset images used by the ZeroDefect inspection pipeline.

---

## Recommended Dataset: MVTec AD

**MVTec Anomaly Detection (MVTec AD)**  
https://www.mvtec.com/research-teaching/datasets/mvtec-ad

### License
The MVTec AD dataset is licensed for **non-commercial research and academic use only**.  
Read the full license at: https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads  
Cite the dataset in any publication.

### Citation
```
Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger:
"MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection"
CVPR 2019.
```

---

## Setup Instructions

### Step 1: Download one category

1. Visit https://www.mvtec.com/company/research/datasets/mvtec-ad
2. Register and download one texture category (recommended: **leather** or **wood**)
3. Extract the downloaded ZIP file

### Step 2: Organize the data

Place the extracted category inside this `data/` folder:

```
data/
└── leather/               ← or any other category
    ├── train/
    │   └── good/          ← Normal (defect-free) training images
    │       ├── 000.png
    │       ├── 001.png
    │       └── ...
    └── test/
        ├── good/          ← Normal test images
        │   ├── 000.png
        │   └── ...
        ├── color/         ← Defective (color anomaly) test images
        │   ├── 000.png
        │   └── ...
        ├── cut/           ← Defective (cut) test images
        │   └── ...
        └── ...
```

For the Texture Anomaly Detector (`TextureAnomalyDetector.fit_from_directory`), 
point it to `data/leather/train/good/`.

For evaluation, use images from `data/leather/test/` and the corresponding 
ground-truth masks from `data/leather/ground_truth/`.

---

## Alternative Datasets

If MVTec AD is unavailable, the following open-access datasets may be used:

| Dataset | URL | License |
|---|---|---|
| DAGM 2007 | https://resources.mpi-inf.mpg.de/conferences/dagm/2007/prizes.html | Research use |
| KolektorSDD | https://www.vicos.si/resources/kolektorsdd/ | Research use |
| NEU Surface Defect | http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html | Research use |

Always read and comply with the dataset license before use.

---

## Data Guidelines

- **Never include ground-truth masks as input to the detection pipeline.**  
  Ground-truth masks are only for evaluation after predictions are made.
- **Keep training and test data strictly separate.**
- **Do not tune thresholds using final test labels.**
- Document which category and split you used in your evaluation report.
