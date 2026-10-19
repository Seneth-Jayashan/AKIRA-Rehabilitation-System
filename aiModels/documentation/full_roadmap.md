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
- [ ] *Deferred to Phase 8/9:* IMU-to-stability regression validated with valid paired data
- [ ] *Deferred to Phase 8/9:* Smoothness metrics reference-based validation

## Phase 7: Cross-Dataset Generalization
- [x] Synthetic domain-shift and transfer learning experiment
- [x] Actual source-to-target dataset evaluation (PHYTMO -> SDALLE/GAITEX)
- [x] Subject-independent target testing (SDALLE represents completely independent subjects)
- [x] Compare against the synthetic domain-shift experiment (Documented in Progress Report)

## Phase 8: AKIRA Hardware and Ankle-Specific Data (Active Stage)
- [x] Configure ESP32 firmware for synchronized paired tibia/foot IMU recordings (PCA9548A CH0 & CH1 @ 100Hz, 12 motion channels)
- [x] Build multi-member Android data collection tool with standardized trial/exercise metadata and participant-specific CSV export
- [x] Implement AKIRA Python preprocessing pipeline loader (`AkiraDatasetLoader`) with quality checks, sync verification, and subject-independent splits
- [ ] Conduct pilot collection with participating members across standardized ankle exercises
- [ ] Calibrate sensors and validate ankle kinematics on collected trials
- [ ] Retrain / fine-tune Random Forest, XGBoost, and deep models on participant-independent splits

## Phase 9: Final System Validation (Clinical Readiness)
- [ ] Integrate offline inference with the Android application
- [ ] Evaluate real-time latency, reliability, and failure handling
- [ ] Validate exercise recognition, ROM, and movement-quality estimates
- [ ] Conduct appropriate expert/clinical validation before clinical claims
