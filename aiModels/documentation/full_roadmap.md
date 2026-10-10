# AKIRA Rehabilitation System - Full R&D Roadmap

## Phase 1: Project Initialization & Data Acquisition
- [x] Repository structure setup
- [x] Symbolic linking of external large datasets
- [x] Basic dependency installation

## Phase 2 & 3: Dataset Auditing & Understanding
- [x] Inspection of PHYTMO, MM-Motion, AnkleImage, GAITEX, SDALLE datasets
- [x] Discovery of data structures, frequencies, and missing ankle-specific articulation classes
- [x] Define proxy classes (e.g. Walking/Gait to substitute fundamental range of motion)

## Phase 4: Data Engineering and Baseline Models
- [x] Universal Dataset Loader (IMU vs Biomechanical)
- [x] DSP Filtering (20Hz low-pass Butterworth)
- [x] Temporal Windowing & Subject-Independent Splitting
- [x] Feature Extraction (Time/Frequency domains)
- [x] Train/Evaluate Classical ML and Deep Learning architectures
- [x] Reproducibility and class-wise evaluation audit

## Phase 5: Optimization and Export
- [x] Hyperparameter Tuning for top models via GridSearchCV / RandomizedSearchCV
- [x] Address Class Imbalance via `class_weight='balanced'`
- [x] Feature Importance Analysis (SHAP)
- [x] Inference Optimization (ONNX conversion for mobile)
- [x] Verify saved models, preprocessing parity, and latency

## Phase 6: Movement Quality and Stability
- [x] CoP-based stability calculation (Algorithms Implemented)
- [x] Smoothness metrics (RMS Jerk, SPARC) (Algorithms Implemented)
- [x] IMU-to-stability regression (POC Simulated)
- [ ] IMU-to-stability regression validated with valid paired data
- [ ] Smoothness metrics reference-based validation

## Phase 7: Cross-Dataset Generalization
- [x] Synthetic domain-shift and transfer learning experiment
- [ ] Actual source-to-target dataset evaluation (PHYTMO -> SDALLE/GAITEX)
- [ ] Subject-independent target testing
- [ ] Compare against the synthetic domain-shift experiment

## Phase 8: AKIRA Hardware and Ankle-Specific Data (Next Major Stage)
- [ ] Collect paired tibia/foot IMU recordings from AKIRA hardware
- [ ] Define a repeatable, supervised data-collection protocol
- [ ] Label ankle exercises and establish reference measurements
- [ ] Calibrate sensors and validate ankle kinematics
- [ ] Fine-tune or retrain suitable models

## Phase 9: Final System Validation (Clinical Readiness)
- [ ] Integrate offline inference with the Android application
- [ ] Evaluate real-time latency, reliability, and failure handling
- [ ] Validate exercise recognition, ROM, and movement-quality estimates
- [ ] Conduct appropriate expert/clinical validation before clinical claims
