# AKIRA Wrist Component — Model 3: 1D CNN Quaternion Classifier

**Date:** 2026-10-10
**Task:** Implementation of 1D Convolutional Neural Network (CNN)

## 1. Objective and Setup
This phase developed **Model 3: a 1D CNN** designed to classify wrist rehabilitation movements using raw temporal sequences of spatial quaternions (`W, X, Y, Z`) extracted exclusively from the `3DCDataset`. 

This differs fundamentally from Model 2 (Random Forest), which collapsed the temporal sequences into 20 static statistical features. Model 3 leverages a 1D CNN to dynamically learn temporal patterns and sequence dependencies directly.

## 2. Dataset and Preprocessing
- **Source:** `01_DATASETS/RAW/3DCDataset-main.zip` (1,936 sequences).
- **Target Classes:** 0 (Neutral), 1 (Radial Deviation), 2 (Wrist Flexion), 3 (Ulnar Deviation).
- **Sequence Handling:** Temporal sequences were preserved in order and padded/truncated to a fixed window length of `MAX_TIME_STEPS = 250` to create uniform `(250, 4)` input tensors.
- **Data Integrity:** No data was fabricated, and missing sensor samples were natively handled by post-zero padding without altering the existing physical sequence structures.

## 3. Data Leakage Prevention
To guarantee a fair evaluation and eliminate data leakage, the exact same `GroupShuffleSplit` strategy (grouped by `participant_id` with `random_state=42`) used in Model 2 was deployed. 
- **Train Split:** 17 isolated participants.
- **Test Split:** 5 held-out participants.
No participant evaluated in the test set existed in the training set.

## 4. Model Architecture & Training
- **Architecture:** 
  - `Conv1D (64 filters, kernel=5) + ReLU`
  - `MaxPooling1D (pool=2)`
  - `Conv1D (128 filters, kernel=3) + ReLU`
  - `GlobalAveragePooling1D`
  - `Dense (64) + Dropout (0.5)`
  - `Dense (4) + Softmax`
- **Training Strategy:** Adam optimizer, sparse categorical cross-entropy loss. `EarlyStopping` was implemented (patience=5) to prevent overfitting, dynamically restoring the best weights based on validation loss.

## 5. Evaluation and Comparison
On the strict, participant-independent held-out test set, Model 3 achieved:
- **Accuracy:** 47.05%
- **Macro F1-Score:** 44.63%

**Model Comparison (CNN vs. Random Forest):**
Evaluated under identical split conditions, the Random Forest (Accuracy: 52.7%) outperformed the 1D CNN (Accuracy: 47.05%). 
This is scientifically expected: because absolute quaternions define fixed orientations rather than dynamic kinematics (e.g., velocity peaks), the static variance/range statistics engineered for the Random Forest provide a stronger absolute spatial signal than the temporal convolutions, which struggle to find consistent translational velocity patterns in absolute quaternion tracks across different participants' resting arm angles.

## 6. Output Artifacts
- **Model Binary:** `09_MODELS/1d_cnn_quaternion_classifier.keras`
- **Config:** `09_MODELS/1d_cnn_quaternion_config.json`
- **Metrics:** `08_RESULTS/1d_cnn_quaternion_metrics.json`
- **Visualizations:** `07_VISUALIZATIONS/1d_cnn_quaternion/` (Confusion matrix & training history).
- **Inference Script:** `12_SCRIPTS/modeling/predict_1d_cnn_quaternion.py`

## 7. Limitations
This CNN confirms that temporal convolutions on uncalibrated, absolute single-IMU quaternions struggle to generalize across participant arm anatomies. To achieve clinical-grade recognition (as required by the final AKIRA system), synchronized dual-IMU kinematics mapping the true relative joint angle is strictly required. No clinical validation is claimed for this research prototype.
