# Multi-Illumination Defect Classification

This repository provides a PyTorch-based implementation of a **multi-illumination surface defect classification model**. It combines an autoencoder-based encoder-decoder structure with cross-reconstruction, multi-view fusion, and smoothed supervision for robust performance under varying lighting conditions.

# Image Patch Generator from JSON Annotations

This script extracts fixed-size image patches (e.g., 200×200) from original images using rectangle-shaped annotations provided in JSON files (e.g., LabelMe format). The cropped patches are saved with consistent naming and stored in a mirrored directory structure (train/valid/test by class).

