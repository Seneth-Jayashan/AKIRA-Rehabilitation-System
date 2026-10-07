# AKIRA Rehabilitation System - Progress Report

This document outlines the development progress for the **AKIRA Rehabilitation System**, explicitly mapping completed work to the proposed architectural roadmap.

## Stage 1 — Dataset Understanding

### 1. Project Initialization & Structure (Phase 1)
The core project directory has been established within the workspace to maintain clean separation of concerns, avoiding direct modification of raw datasets. Symbolic links were utilized for the massive dataset archives (residing on external volumes) to prevent unmanageable storage consumption.
```text
AKIRA/
├── datasets/
│   ├── raw/ (symlinked to external storage)
│   ├── processed/
│   └── metadata/
├── src/
│   ├── preprocessing/
│   ├── feature_extraction/
│   ├── segmentation/
│   ├── models/
│   └── evaluation/
└── documentation/
```

### 2. Dataset Auditing (Phase 2 & Phase 3)
An automated Python inspection script was created (`scratch/inspect_datasets.py`) to peek inside massive `.zip` and `.rar` datasets (such as GAITEX and MM-Motion) without extracting hundreds of gigabytes. The findings were documented in `dataset_inventory.md`.
*   **AnkleImage**: Analyzed structure; found to contain Biomechanical/Force Plate data and EMG signals rather than standard IMU signals in its Task 5.
*   **PHYTMO**: Verified structure; contains high-quality 9-DOF IMU data (Accel, Gyro, Mag) recorded at ~100Hz with explicit activity labels.
*   **MM-Motion**: Identified as a highly dimensional (48-channel per limb) movement stability dataset stored across 16 massive zip files.

---

## Stage 2 — Data Preparation

### 3. Common Data Format & Dynamic Loaders (Phase 4)
To ensure the machine-learning pipeline can ingest any dataset seamlessly, a unified `BaseDatasetLoader` was designed in `src/preprocessing/base_loader.py`. 
*   **Data Type Routing**: Based on an architectural correction, the loader dynamically distinguishes between `DataType.IMU` and `DataType.BIOMECHANICAL` to prevent inappropriate kinematic modeling.
*   **Unified Schema**:
    *   **IMU**: `[ax, ay, az, gx, gy, gz]`
    *   **Biomechanical**: `[fx, fy, fz, mx, my, mz, cop_x, cop_y, cop_z, emg]`

**Implemented Loaders**:
*   `PHYTMOLoader`: Successfully tested by parsing over 18.6 million samples across 23 activities.
*   `MMMotionLoader`: Implemented with direct-from-zip streaming capabilities.
*   `AnkleImageLoader`: Registered explicitly as a BIOMECHANICAL data source to process ground-reaction and muscle activation features.

### 4. Sensor Calibration, Filtering & Normalization (Phase 6)
A digital signal processing module (`src/preprocessing/filters.py`) was introduced to apply data-type specific filters:
*   **IMU Filtering**: A 20Hz zero-phase Butterworth low-pass filter is applied to `[ax, ay, az, gx, gy, gz]` signals to remove high-frequency environmental noise while preserving human kinematics.
*   **Biomechanical Filtering**: A 15Hz low-pass filter is provisioned for Force Plate and CoP data to remove impact vibrations, with scaffolding laid for EMG envelope extraction (rectification and low-pass).

### 5. Time-Series Segmentation (Phase 7)
Machine-learning models require discrete windows of data rather than continuous streams. `TimeSeriesSegmenter` (`src/segmentation/windowing.py`) was built to convert continuous trial data into sliding windows.
*   **Configuration**: 2.0-second windows with a 50% overlap.
*   **Leakage Prevention**: The logic explicitly groups by `subject_id`, `session_id`, and `trial_id` before slicing, ensuring that windows never cross the boundary between two disparate recordings.

### 6. Feature Extraction (Phase 9)
To translate the raw sensor windows into meaningful descriptors for the AI models, `FeatureExtractor` (`src/feature_extraction/extractors.py`) was implemented.
*   **Time Domain**: Computes `mean`, `std`, `var`, `min`, `max`, `range`, `rms`, and Zero-Crossing Rate (`zcr`).
*   **Frequency Domain**: Computes Power Spectral Density (PSD) via Welch’s Method, extracting `dominant_freq`, `spectral_energy`, and `frequency_entropy`.

### 7. End-to-End Orchestration
The `PreprocessingPipeline` (`src/preprocessing/pipeline.py`) was finalized to orchestrate all the aforementioned steps automatically. Feeding any dataset loader into this pipeline executes format detection, sampling-rate verification, routing, filtering, normalization, segmentation, and feature extraction, ultimately outputting a single, flattened Pandas DataFrame ready for Stage 4 (AI Models).
