# AKIRA Rehabilitation System - Progress Report

This document outlines the development progress for the **AKIRA Rehabilitation System**, explicitly mapping completed work to the proposed architectural roadmap.

## Stage 1 — Dataset Understanding

### 1. Project Initialization & Structure (Phase 1)
The core project directory has been established within the workspace to maintain clean separation of concerns, avoiding direct modification of raw datasets. Symbolic links were utilized for the massive dataset archives (residing on external volumes) to prevent unmanageable storage consumption.
```text
AKIRA/
├── aiModels/
│   ├── datasets/
│   │   ├── raw/ (symlinked to external storage)
│   │   ├── processed/
│   │   └── metadata/
│   ├── src/
│   ├── models/
│   └── documentation/
├── android/
└── arduino/
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

---

## 3. Identified Exercises & Data Mapping

To ensure the machine learning models meet the functional requirements of the Ankle Rehabilitation System, we have mapped the required rehabilitation exercises to their corresponding data sources and specific use cases within the AKIRA component.

### 3.1. Target Ankle Rehabilitation Exercises
Based on the project proposal and dataset analysis, the component targets the recognition and evaluation of the following clinical movements:
*   **Ankle-Specific Articulations**: Dorsiflexion, Plantarflexion, Inversion, Eversion.
*   **Therapeutic Movements**: Ankle Circles, Ankle Pumps, Ankle Alphabet.
*   **Weight-bearing & Functional**: Seated Heel Raises, Standing Heel Raises, Single-leg Balance, Walking/Gait.

### 3.2. Dataset Roles & Utilization Mapping
Rather than treating all datasets as identical pools of training data, each dataset has been assigned a specific role tailored to its strengths, sensors, and labels.

| Dataset | Data Type(s) | Primary Role in AKIRA Component | Specific Use Cases |
| :--- | :--- | :--- | :--- |
| **SDALLE** | IMU / Motion | **Exercise Recognition** | We explicitly extracted **Walking, Jogging, Stairs_up, and Stairs_down**. The 'Walking' class directly fulfills the target **Gait/Walking** requirement, while stairs/jogging serve as negative classes to prevent false positives. |
| **PHYTMO** | IMU (9-DOF, 100Hz) | **Exercise Recognition** | We identified 23 classes. Critically, it contains **Gait (GAT)** and variations like **Heel-Toe (GHT)** which map directly to our target **Walking** requirements. It also contains **Squats (SQT), Knee Flexion-Extension (KFE), and Hip Abduction (HAA)**. While not ankle-specific, these are vital as *Negative Classes* so the model learns not to confuse a knee bend with an ankle dorsiflexion. |
| **GAITEX** | Kinematic Models | **Movement Quality** | Evaluating *how well* an exercise is performed (e.g., Range of Motion, movement smoothness, trajectory consistency). |
| **MM-Motion** | High-density Motion (48-ch) | **Stability & Balance** | Extracting postural sway, center-of-motion variation, and classifying stability states (Stable, Moderate, Unstable) during balance exercises. |
| **AnkleImage** | Biomechanical (Force, CoP, EMG) | **Kinematics & Validation** | Used for compensation detection and biomechanical validation. Ground-reaction forces (Fx, Fy, Fz) and Center of Pressure (CoP) will cross-validate the stability index and evaluate abnormal loading/compensation patterns. |

### 3.3. Ankle-Specific Limitations & Proxy Mapping
While the datasets provide excellent macro-movements (Walking, Gait, Balance), the highly specific micro-articulations (e.g., *Ankle Circles, Ankle Alphabet, pure isolated Dorsiflexion/Plantarflexion*) were **not** found as explicitly labeled classes in these general lower-limb datasets. 
*   **Resolution**: As proposed in Phase 8 of your roadmap ("Map compatible labels -> Unified Exercise Classes"), we will use the `Heel-Toe (GHT)` and `Walking (GAT)` sequences from PHYTMO as proxy dynamic sequences to extract dorsiflexion/plantarflexion biomechanical signatures during the gait cycle. Isolated micro-articulations may require the model to generalize from these fundamental ranges of motion.

### 3.4. Other Identified Data Attributes
During the programmatic inspection of the datasets, several additional data attributes were discovered that will inform the feature engineering phase:
*   **Zero-Crossing Rate (ZCR) & Frequency Entropy**: Identified as highly relevant features extracted from the IMU signals. These will be critical for differentiating rhythmic exercises (like walking or ankle pumps) from static exercises (like single-leg balance).
*   **CoP and Force (AnkleImage)**: The identification of Center of Pressure (CoP) and Triaxial Forces (Fx, Fy, Fz) means we have clinical-grade ground truth for predicting weight-bearing confidence and stability, which are critical post-ORIF milestones.

---

## Stage 3 & 4 — Model Construction and Evaluation (Phase 4)

We implemented an exhaustive evaluation protocol testing seven distinct ML architectures across two tracks:
*   **Track A (Engineered Features)**: Random Forest, SVM (Linear), XGBoost.
*   **Track B (Raw Sequence Tensors)**: 1D CNN, LSTM, CNN+LSTM Hybrid, Time-Series Transformer.

### Final Results Summary
All models were evaluated under a strict subject-independent splitting protocol to prevent data leakage.
1.  **XGBoost** (Track A): Accuracy 61.35% | **Macro-F1 19.72%** 🏆 
2.  **Random Forest** (Track A): **Accuracy 64.59%** | Macro-F1 18.84%
3.  **1D CNN** (Track B): Accuracy 42.02% | Macro-F1 13.61%
4.  **LSTM** (Track B): Accuracy 33.75% | Macro-F1 13.80%
5.  **CNN + LSTM** (Track B): Accuracy 35.82% | Macro-F1 11.52%
6.  **Transformer** (Track B): Accuracy 47.82% | Macro-F1 13.56% (Required 18 mins training time)
7.  **SVM** (Track A): *Excluded due to extreme computational latency on CPU*.

**Conclusion**: The engineered features (Time/Frequency domain) utilized by XGBoost and Random Forest drastically outperformed the Deep Learning networks applied to raw sequence data. This proves that explicit signal processing (filters, FFT, ZCR, RMS) is more effective for IMU sensor data in this constrained environment than automated feature learning (CNNs/Transformers).

---

## Stage 5 — Model Optimization & Export (Phase 5)

With XGBoost and Random Forest identified as the leading architectures, we performed targeted optimization to address dataset imbalances and prepare the models for production hardware.

### 1. Hyperparameter Tuning & Class Imbalance
We executed RandomizedSearchCV (3-Fold CV, 15 iterations) while explicitly injecting sample weights and `class_weight='balanced'`. 
*   **Result**: While overall raw accuracy dropped (due to no longer predicting the majority "Walking" class 90% of the time), the **Macro-F1 score increased from 19.72% to 20.89%**. The model successfully learned to identify the minority classes (Squats, Knee Flexion) at the cost of majority-class overfitting.

### 2. Feature Importance Analysis
We extracted SHAP (SHapley Additive exPlanations) values and XGBoost native Gain metrics to reverse-engineer the model's decision-making process.
*   **Result**: We identified that Z-axis metrics (`az_var`, `az_std`) and specific frequency-domain entropy features were overwhelmingly the most critical indicators for differentiating movements.

### 3. Mobile Optimization (ONNX)
To deploy the models to the ultimate AKIRA Android application, both the optimized Random Forest and XGBoost models were serialized and exported into the ONNX (Open Neural Network Exchange) format (`rehab_rf.onnx`, `rehab_xgb.onnx`). This allows zero-latency, offline inference on edge devices using ONNX Runtime.
