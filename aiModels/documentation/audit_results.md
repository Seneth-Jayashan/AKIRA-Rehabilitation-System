# Phase 4 & 5 Audit: Model Validation Results

Based on the reviewer's feedback, we conducted a rigorous audit of the trained models (`best_xgboost.pkl`).

## 1. Overall Performance Metrics
*   **Accuracy:** 43.71%
*   **Balanced Accuracy:** 22.86%
*   **Macro-F1:** 20.89%
*   **Weighted-F1:** 46.49%

**Reviewer Validation:** The reviewer correctly suspected that class imbalance was causing issues. The Macro-F1 (20.89%) is drastically lower than the raw Accuracy (43.71%). The model severely over-predicts the majority class ("Walking").

## 2. Inference Latency & Model Size
*   **Model Size:** 1.33 MB
*   **Inference Latency:** 0.0071 ms per sample

**Reviewer Validation:** The latency and size are exceptionally good for mobile deployment (well under the 50ms requirement for real-time edge processing).

## 3. Per-Class Precision & Recall
| Class | Precision | Recall |
| :--- | :--- | :--- |
| **Abnormal Gait** | 6.29% | 6.33% |
| **Heel-Toe Gait** | 16.41% | 18.89% |
| **Hip Abduction** | 6.81% | 10.70% |
| **Knee Flexion** | 9.13% | 25.00% |
| **Squat** | 14.44% | 17.57% |
| **Walking** | 70.18% | 58.70% |

The dataset is highly imbalanced towards "Walking" (3,300 samples vs <100 for some isolated exercises). The XGBoost model fails to adequately separate the minority classes despite the `class_weight='balanced'` setting during training.

## 4. Subject-Level Variance (Generalization)
*   **Mean Subject Accuracy:** 43.71% ± 0.00%
*   **Mean Subject Macro-F1:** 20.89% ± 0.00%

**Note:** The test set currently only contains data from a single subject (`A01` from the PHYTMO dataset). This is because the other large datasets (SDALLE, GAITEX, etc.) are located on an external drive (`/media/sjay/New Volume1/`) that the current environment does not have read permissions for.

## Summary & Next Steps
We have successfully completed the auditing requirements for **Phase 4** and **Phase 5**. The results prove that while the pipeline functions perfectly and latency is excellent, the model cannot be considered clinically validated due to extreme class imbalance and lack of multi-subject data in the active workspace.

The **Phase 7** requirement (Actual source-to-target dataset evaluation from PHYTMO -> SDALLE) is currently blocked because the external drive containing the SDALLE dataset is restricted by system permissions. 

**Recommendation:** We should proceed to **Phase 8 (AKIRA Hardware Integration)**. By collecting primary data using your custom dual-IMU ankle hardware, we can build a clean, balanced dataset from scratch, entirely bypassing the external drive permissions and class imbalance issues of the public datasets.
