# AKIRA Wrist Component — Model 4: XGBoost Quaternion Classifier

**Date:** 2026-10-10
**Task:** Development and evaluation of XGBoost for quaternion sequence classification.

---

## 1. Pipeline Verification and Preprocessing
The existing pipeline for Model 2 (Random Forest) was successfully audited and adapted for Model 4 (XGBoost).
- **Dataset Used:** `01_DATASETS/RAW/3DCDataset-main.zip`
- **Class Labels:** The four target classes (`NEUTRAL`, `RADIAL_DEVIATION`, `WRIST_FLEXION`, `ULNAR_DEVIATION`) were preserved, explicitly encoded via Scikit-Learn's `LabelEncoder` to support XGBoost's integer constraints.
- **Features Extracted:** The same 20 engineered temporal statistics (`Mean`, `Std`, `Min`, `Max`, `Range` for `W, X, Y, Z`) were extracted.
- **Data Splitting:** To enforce a fair and rigorous comparison, the exact same `GroupShuffleSplit` (grouped by `participant_id` with `random_state=42`) was utilized, completely isolating the same 5 test participants from the training and hyperparameter configuration process.

## 2. XGBoost Training Configuration
The model was trained strictly on the training partition with the following configuration:
- **Objective:** `multi:softprob`
- **Learning Rate:** `0.1`
- **Max Depth:** `6`
- **Estimators:** `100`
- **Class Imbalance Handling:** Implemented via Scikit-Learn's `compute_sample_weight(class_weight='balanced')` applied to the training set instances.

## 3. Evaluation and Comparison
Tested strictly against the held-out participant cohort (no hyperparameters were tuned against this set):

**Metrics:**
- **Accuracy:** 46.36%
- **Macro Precision:** 45.46%
- **Macro Recall:** 46.36%
- **Macro F1-Score:** 45.29%

**Model Comparison (Identical Test Conditions):**
1. **Model 2 (Random Forest):** 52.73% Accuracy
2. **Model 3 (1D CNN):** 47.05% Accuracy
3. **Model 4 (XGBoost):** 46.36% Accuracy

### Analysis
XGBoost performed marginally worse than the 1D CNN and noticeably worse than the Random Forest ensemble on the same exact features. While XGBoost is a powerful gradient boosting algorithm, its performance ceiling in this project is strictly dictated by the fundamental limitation of the dataset: **single-sensor absolute quaternions**. 

Because absolute spatial orientations do not translate directly into relative joint-angle kinematics (due to different participants possessing vastly different resting arm/wrist positions), the statistical variance features provide weak generalized signals. The Random Forest ensemble manages to loosely capture these broad bounds slightly better than XGBoost's sequential tree boosting, but neither algorithm can overcome the lack of physical kinematic grounding.

## 4. Inference Verification
Inference was successfully tested using the script `predict_xgboost_quaternion.py` on a held-out sequence (`3DCDataset-main/Participant9/train/IMU/3dc_IMU_gesture_3_9.txt`). 
- **Result:** The model successfully executed the pipeline, returning a predicted class of `ULNAR_DEVIATION` with a confidence probability of `59.80%`.

## 5. Value and Conclusion
**Does XGBoost add useful value over Random Forest?** 
No. XGBoost demonstrates lower accuracy and macro F1 scores than the existing Random Forest model without providing any additional interpretability or inference-speed advantages that would justify replacing Model 2. 

**Clinical Applicability:** 
This model is a mathematical prototype operating on absolute spatial sequences. It does not map relative range of motion and possesses no clinical rehabilitation validation. It cannot and should not be used for post-ORIF clinical assessment.

## 6. Generated Artifacts
- **Model:** `09_MODELS/xgboost_quaternion_classifier.joblib` (contains model and `LabelEncoder`)
- **Features:** `09_MODELS/xgboost_quaternion_features.json`
- **Metrics:** `08_RESULTS/xgboost_quaternion_metrics.json`
- **Classification Report:** `08_RESULTS/xgboost_quaternion_classification_report.csv`
- **Confusion Matrix:** `07_VISUALIZATIONS/xgboost_quaternion/confusion_matrix.png`
- **Training Script:** `12_SCRIPTS/modeling/train_xgboost_quaternion.py`
- **Prediction Script:** `12_SCRIPTS/modeling/predict_xgboost_quaternion.py`
