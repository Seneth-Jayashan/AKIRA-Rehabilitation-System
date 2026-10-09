# AKIRA AI Model Experiment Protocol

This document defines the scientifically controlled experimental protocol for comparing machine-learning models for the AKIRA Rehabilitation System. The goal is not to force a specific architecture, but to empirically evaluate multiple models under strict, subject-independent cross-validation to select the most robust baseline.

## 1. Unified Exercise Taxonomy Mapping

To train a generalized model, disparate labels from multiple public datasets (e.g., PHYTMO, SDALLE) are mapped into a unified set of target and negative classes.

### 1.1 Data Mapping Table
| Raw Dataset Label | Dataset Source | Standardized Label | Role |
| :--- | :--- | :--- | :--- |
| `Walking` | SDALLE | **Walking** | Target |
| `GAT`, `GAT0`, `GAT1` | PHYTMO | **Walking** | Target |
| `GHT` | PHYTMO | **Heel-Toe Gait** | Target / Proxy |
| `SQT`, `SQT0`, `SQT1` | PHYTMO | **Squat** | Negative |
| `KFE`, `KFEL0`, `KFER1`... | PHYTMO | **Knee Flexion** | Negative |
| `HAA`, `HAAL0`, `HAAR1`... | PHYTMO | **Hip Abduction** | Negative |
| `Jogging` | SDALLE | **Jogging** | Negative |
| `Stairs_up`, `Stairs_down` | SDALLE | **Stairs** | Negative |
| `Calib`, `LCalib`, `RCalib` | PHYTMO | *Dropped* | Calibration |

### 1.2 Ankle-Specific Post-Training Targets
The following micro-articulations lack explicit public data labels. They will be integrated during **Phase 6 (AKIRA Dataset Fine-Tuning)** using real-time patient collection:
*   Dorsiflexion
*   Plantarflexion
*   Ankle Circles
*   Ankle Alphabet

---

## 2. Subject-Independent Dataset Split
To prevent data leakage caused by overlapping sliding windows, the dataset is strictly split by `subject_id`. Windows belonging to the same trial or subject will *never* cross into both training and validation sets.

**Split Configuration**:
*   **70%** Training Subjects
*   **15%** Validation Subjects (used for hyperparameter tuning)
*   **15%** Testing Subjects (held-out for final model comparison)

---

## 3. Experimental Tracks

We evaluate the models across two primary research tracks to answer the question: *Do engineered features or learned temporal representations perform better for exercise recognition?*

### Track A: Feature-Based Classical ML
Input: Extracted time-domain and frequency-domain features (Mean, STD, ZCR, Dominant Frequency, Spectral Energy, etc.)
*   **Model 1**: Random Forest (Baseline)
*   **Model 2**: Support Vector Machine (SVM)
*   **Model 3**: XGBoost

### Track B: Raw Sequence Deep Learning
Input: Segmented, normalized continuous IMU time-series windows (e.g., `2.0s` window of `ax, ay, az, gx, gy, gz`).
*   **Model 4**: 1D Convolutional Neural Network (CNN)
*   **Model 5**: Long Short-Term Memory (LSTM)
*   **Model 6**: CNN-LSTM Hybrid
*   **Model 7**: Time-Series Transformer (Optional/Advanced)

---

## 4. Evaluation Metrics
Models will not be judged solely on accuracy. The final architecture will be selected based on a holistic evaluation of the following metrics on the exact same held-out test set:

1.  **Classification Metrics**:
    *   Accuracy
    *   Macro-F1 Score (Critical for class imbalances)
    *   Precision & Recall
2.  **Error Analysis**:
    *   Confusion Matrices (Identifying which specific exercises are confused)
3.  **Deployment Feasibility**:
    *   Inference time (ms per window)
    *   Computational complexity / Model size (for future Android deployment)
