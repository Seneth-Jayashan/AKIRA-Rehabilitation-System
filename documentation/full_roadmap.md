# AKIRA Rehabilitation System - Full R&D Roadmap

## Phase 1: Project Initialization & Data Acquisition
- [x] Repository structure setup
- [x] Symbolic linking of external large datasets
- [x] Basic dependency installation

## Phase 2 & 3: Dataset Auditing & Understanding
- [x] Inspection of PHYTMO, MM-Motion, AnkleImage, GAITEX, SDALLE datasets
- [x] Discovery of data structures, frequencies, and missing ankle-specific articulation classes
- [x] Define proxy classes (e.g. Walking/Gait to substitute fundamental range of motion)

## Phase 4: Data Engineering & Initial Model Evaluation
- [x] Universal Dataset Loader (IMU vs Biomechanical)
- [x] DSP Filtering (20Hz low-pass Butterworth)
- [x] Temporal Windowing & Subject-Independent Splitting
- [x] Feature Extraction (Time/Frequency domains)
- [x] **Track A**: Train/Evaluate Classical ML (Random Forest, SVM, XGBoost)
- [x] **Track B**: Train/Evaluate Deep Learning (1D CNN, LSTM, CNN-LSTM, Transformer)
- [x] Generate Confusion Matrices and Result Summaries

## Phase 5: Model Optimization & Fine-Tuning
- [ ] Hyperparameter Tuning for top models (XGBoost, Random Forest) via GridSearchCV / RandomizedSearchCV
- [ ] Address Class Imbalance via `class_weight='balanced'` and `scale_pos_weight`
- [ ] Feature Importance Analysis (SHAP / Feature Permutation) to identify crucial sensors
- [ ] Inference Optimization (Quantization/ONNX conversion for mobile)

## Phase 6: Exercise Quality & Stability Estimation
- [ ] Map AnkleImage Force Plate (Fx, Fy, Fz, CoP) to Stability Indices
- [ ] Develop Regression model to predict balance/weight-bearing confidence from IMU
- [ ] Extract Kinematic smoothness metrics

## Phase 7: Cross-Dataset Generalization (Zero-Shot / Few-Shot)
- [ ] Train on PHYTMO, test on SDALLE or GAITEX to evaluate out-of-distribution robustness
- [ ] Apply Transfer Learning if required

## Phase 8: AKIRA Hardware Integration (Fine-Tuning)
- [ ] Collect primary data using dual-IMU ankle hardware built for AKIRA
- [ ] Fine-tune the pre-trained XGBoost / Random Forest on AKIRA data
- [ ] Finalize model for Android / Embedded deployment
