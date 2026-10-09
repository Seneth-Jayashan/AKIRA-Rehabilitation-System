# Phase 4: Model Evaluation and Results Analysis

## 1. Experimental Setup
*   **Task**: 6-class highly imbalanced exercise recognition (Walking, Gait, Squat, Knee Flexion, Hip Abduction, Heel-Toe Gait).
*   **Evaluation Metric**: Macro F1-Score (due to severe class imbalance) and Accuracy.
*   **Splitting Strategy**: Subject-independent holdout validation to prevent intra-subject data leakage.

## 2. Quantitative Results
| Rank | Model | Track | Accuracy | Macro F1-Score | Train Time | Inference Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | **XGBoost** | A (Engineered) | 61.35% | **19.72%** | ~4s | 0.03s |
| 2 | Random Forest | A (Engineered) | 64.59% | 18.84% | ~0.5s | 0.05s |
| 3 | LSTM | B (Raw Series) | 33.75% | 13.80% | ~260s | 2.30s |
| 4 | 1D CNN | B (Raw Series) | 42.02% | 13.61% | ~50s | 0.39s |
| 5 | Transformer | B (Raw Series) | 47.82% | 13.56% | ~1076s | 3.87s |
| 6 | CNN + LSTM | B (Raw Series) | 35.82% | 11.52% | ~361s | 1.65s |
| 7 | SVM (Linear) | A (Engineered) | *N/A* | *N/A* | *Timeout* | *N/A* |

## 3. Analysis and Conclusions
1.  **Engineered Features vs. Deep Learning**: The classical ML models relying on engineered Time and Frequency domain features (Track A) vastly outperformed deep networks operating on raw sequences (Track B). The extraction of explicit signal characteristics (e.g., spectral energy, dominant frequencies, RMS, zero-crossing rates) provides a significantly stronger inductive bias for IMU data than automated representation learning given our limited dataset size.
2.  **Class Imbalance Vulnerability**: All models struggled immensely with minority classes (Squat, Knee Flexion), resulting in very low Macro-F1 scores compared to raw accuracy. The models overfit to the majority class ("Walking"), predicting it almost exclusively.
3.  **Computational Feasibility**: The Transformer model is fundamentally unviable for this environment and prospective embedded deployment due to extreme training latency (18 minutes for 15 epochs) and inference delay. XGBoost and Random Forest train in seconds and infer in milliseconds.

## 4. Next Steps (Phase 5)
1.  **Hyperparameter Tuning**: Optimize XGBoost (e.g., `scale_pos_weight`) and Random Forest (`class_weight='balanced'`) to directly penalize misclassification of minority classes and boost the Macro-F1 score.
2.  **Feature Selection**: Identify which of the extracted features are actually contributing to the XGBoost decisions, potentially discarding noisy features.
