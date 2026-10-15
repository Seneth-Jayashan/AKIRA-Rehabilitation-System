# Quaternion-Based Wrist Movement Classification Using Random Forest
## Final Report

**Date:** 2026-10-10
**Task:** Implementation of Model 2 (Random Forest) Using 3DCDataset Quaternions

### 1. Dataset Used
The experiment utilized the **3DCDataset**, specifically extracting the 1,936 IMU text files located within the participant test and train directories (`01_DATASETS/RAW/3DCDataset-main.zip`).

### 2. Input Representation
The dataset did **not** contain raw accelerometer and gyroscope telemetry. Instead, it provided chronologically ordered 3D spatial orientations explicitly encoded as **Quaternions (W, X, Y, Z)**. These were strictly preserved as orientation inputs and were not misrepresented as raw IMU kinematics.

### 3. Exercise Labels Recognized
Gestures were mapped based on the original 3DCDataset reference mappings:
- **0:** NEUTRAL (Rest/Other)
- **1:** RADIAL_DEVIATION
- **2:** WRIST_FLEXION
- **3:** ULNAR_DEVIATION

### 4. Feature Extraction Method
Because quaternions represent absolute spatial orientations rather than continuous kinematic signals, window-level temporal statistics were extracted across the duration of each recording to capture orientation distribution and magnitude of change.
- **Features Extracted per Quaternion Component (W, X, Y, Z):**
  - Mean
  - Standard Deviation (capturing variance in orientation)
  - Minimum & Maximum
  - Range (Max - Min, capturing the absolute magnitude of orientation change)
- *Total Features:* 20 numeric columns.

### 5. Participant-Independent Split
To strictly prevent data leakage, a **GroupShuffleSplit** strategy was employed, grouped by `participant_id`. 
- **Train Set:** Data from 17 independent participants (approx. 80% of data).
- **Test Set:** Data from 5 held-out participants (approx. 20% of data).
- No temporal overlapping windows or samples from the same participant crossed the train/test boundary.

### 6. Test Results (Participant-Independent Evaluation)
Evaluating the Random Forest model exclusively on the unseen test participants yielded:
- **Accuracy:** 52.7%
- **Macro Precision:** 53.6%
- **Macro Recall:** 52.7%
- **Macro F1-Score:** 52.2%

*Note:* In a 4-class prediction problem, a random baseline is 25%. Achieving ~53% accuracy on strict participant-independent testing demonstrates that the model successfully learned generalized spatial orientation patterns associated with the wrist deviations, despite the lack of true kinematic signals (velocity/acceleration).

### 7. Limitations and Data Leakage Risks
- **Absolute vs. Relative Orientation:** Quaternions represent absolute orientation in space. Without a secondary calibration sensor, the absolute starting position of a participant's arm heavily biases the data. While statistical range/variance features mitigate this, true relative kinematics are missing.
- **Leakage Risk Mitigated:** GroupShuffleSplit completely prevented the model from memorizing participant-specific resting orientations.

### 8. Exact Commands for Training and Prediction
To retrain the model and regenerate all metrics/visualizations from scratch:
```powershell
python 12_SCRIPTS/modeling/train_random_forest_quaternion.py
```
To run the read-only inference script using the saved artifacts on existing feature data:
```powershell
python 12_SCRIPTS/modeling/predict_random_forest_quaternion.py
```

### 9. Difference Between this Model and the Existing Autoencoder
- **Autoencoder (Model 1):** An *unsupervised anomaly detector* trained on generic WatchHand raw IMU signals (Accelerometer/Gyroscope) to identify irregular movement execution based on reconstruction MSE. It does not classify specific exercise types.
- **Random Forest (Model 2):** A *supervised classifier* trained on 3DCDataset spatial Quaternions to predict explicit, discrete exercise types (e.g., Wrist Flexion).

### 10. Why Quaternion Classification is NOT Clinical Raw-IMU Performance
This experiment is a successful proof-of-concept for spatial sequence classification. However, it does **not** establish raw-IMU or post-ORIF clinical performance because:
1. The model relies on pre-processed quaternions rather than raw 6-axis hardware inputs (`ax, ay, az, gx, gy, gz`).
2. The data originates from healthy subjects, not post-operative DRF patients with impaired kinematics.
3. The AKIRA target architecture requires synchronized dual-IMU (Forearm + Hand) relative range of motion (ROM) analysis, which cannot be simulated by a single absolute quaternion sensor.
