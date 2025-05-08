# Multi-Illumination Defect Classification

This repository provides a PyTorch-based implementation of a **multi-illumination surface defect classification model**. It combines an autoencoder-based encoder-decoder structure with cross-reconstruction, multi-view fusion, and smoothed supervision for robust performance under varying lighting conditions.

The dataset is organized into separate directories for training, validation, and testing (train/, valid/, and test/), each containing subfolders named after defect classes (e.g., BrightLine, Dent, Scratch, etc.). This structure supports supervised learning with clearly separated splits for training and evaluating model generalization.

# Image Patch Generator from JSON Annotations

This script extracts fixed-size image patches (e.g., 200×200) from original images using rectangle-shaped annotations provided in JSON files (e.g., LabelMe format). The cropped patches are saved with consistent naming and stored in a mirrored directory structure (train/valid/test by class).

##  Preprocessing Instructions

To generate standardized 200×200 image patches from annotated defect regions, please follow the steps below:

### 1️⃣ Download the Dataset

The original multi-illumination surface defect dataset can be downloaded from the following repository:

🔗 **[Fusion-of-multi-light-source-illuminated-images-for-defect-inspection](https://github.com/Xavierman/Fusion-of-multi-light-source-illuminated-images-for-defect-inspection)**

This dataset contains images categorized by defect type and split into `train/`, `valid/`, and `test/` folders.

---

### 2️⃣ Prepare Annotations

1. Download the `annotation.zip` file containing the rectangle-based JSON annotations.
2. Extract all `.json` files from `annotation.zip`.
3. Move the extracted `.json` files into the root directory of the original dataset (i.e., the same level as the `train`, `valid`, and `test` folders).
