# AKIRA Wrist Component — Random Forest Exercise Recognition Report

**Date:** 2026-10-10
**Task:** Implementation of Supervised Random Forest Exercise Recognition Pipeline

## 1. Dataset Selection and Audit Results
An exhaustive audit of the existing `Wrist_Component` data repository was conducted. The objective was to identify a dataset containing both:
- **Raw IMU data** (`ax, ay, az, gx, gy, gz`).
- **Ground-truth labels** for specific wrist rehabilitation exercises (Wrist Flexion, Extension, Radial/Ulnar Deviation, Pronation, Supination).

### Findings:
- **WatchHand Dataset:** Contains raw IMU data but lacks target wrist exercise labels (contains generic gestures).
- **3DCDataset:** Contains target wrist exercise labels but lacks raw IMU data (contains only quaternions).
- **Wrist Kinematics Dataset:** Contains pointing-task joint angles, not raw IMU accelerometer/gyroscope signals.

**Conclusion:** No existing dataset supports supervised training for the specified wrist exercises using accelerometer and gyroscope data.

## 2. Pipeline Execution Status
In strict adherence to the project's scientific constraints—which prohibit the fabrication of sensor data, the invention of labels, and the misrepresentation of quaternion data as raw IMU data—the pipeline was **halted after Phase 1 (Dataset Audit)**.

The following phases could not be executed due to the data gap:
- Phase 2: Select the training dataset
- Phase 3: Data preprocessing and feature extraction
- Phase 4: Participant-independent splitting
- Phase 5: Train the Random Forest model
- Phase 6: Evaluation
- Phase 7: Prediction script

## 3. Differentiation from the Autoencoder Prototype
It is critical to distinguish this halted Random Forest pipeline from the currently functioning Autoencoder:
- **Autoencoder Anomaly Detection (Functioning):** The Autoencoder is an *unsupervised* model. It was successfully trained using the WatchHand dataset because it only requires raw IMU data to learn the distribution of generic movements and flag deviations (anomalies). It does not require explicit exercise class labels.
- **Random Forest Exercise Recognition (Halted):** The Random Forest is a *supervised* model. It cannot be trained without specific, accurately labeled examples of each target wrist exercise.

## 4. Required Data for Pipeline Completion (Final AKIRA System)
To complete this pipeline and fulfill the requirements of the final AKIRA system, a new clinical or controlled dataset must be collected with the following parameters:
1. **Hardware:** Synchronized dual-IMUs (Forearm and Dorsum of the Hand).
2. **Signals:** `ax, ay, az, gx, gy, gz` collected from both sensors simultaneously.
3. **Labels:** Explicit timestamp annotations for:
   - Wrist Flexion
   - Wrist Extension
   - Radial Deviation
   - Ulnar Deviation
   - Pronation
   - Supination
4. **Participants:** A sufficient cohort of subjects (ideally post-ORIF patients) to allow for robust participant-independent cross-validation (e.g., GroupShuffleSplit).

Once this dual-IMU labeled dataset is acquired and placed in `01_DATASETS/RAW/`, the Random Forest supervised training and prediction scripts can be successfully authored and executed.
