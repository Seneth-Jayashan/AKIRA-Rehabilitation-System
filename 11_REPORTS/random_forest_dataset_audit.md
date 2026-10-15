# Random Forest Dataset Audit Report

**Date:** 2026-10-10
**Task:** Supervised Random Forest Exercise Recognition for AKIRA Wrist Rehabilitation

## Phase 1: Dataset Audit

We audited the existing datasets to identify suitable data for training a supervised Random Forest classifier to recognize wrist rehabilitation exercises (Wrist Flexion, Wrist Extension, Radial Deviation, Ulnar Deviation, Pronation, Supination).

### 1. WatchHand Dataset
- **Sensor Data:** Raw Accelerometer and Gyroscope from a single wrist IMU.
- **Exercises:** General gestures (hand washing, typing, etc.).
- **Evaluation:** Lacks ground-truth labels for the target wrist rehabilitation exercises. Cannot be used for supervised exercise classification without fabricating labels.

### 2. 3DCDataset
- **Sensor Data:** Quaternions only (W, X, Y, Z) from a single forearm sensor.
- **Exercises:** Contains labels for the target exercises (Flexion, Extension, Pronation, etc.).
- **Evaluation:** Does not contain raw Accelerometer and Gyroscope channels. Using quaternions as raw IMU inputs violates the physical requirements of the AKIRA system. Cannot be used.

### 3. 29666735 Wrist Kinematics Dataset
- **Sensor Data:** Pre-computed tracking joint angles (flexion_extension_angle, radial_ulnar_angle).
- **Exercises:** Contains pointing tasks, not isolated rehab exercises.
- **Evaluation:** Does not contain raw IMU accelerometer and gyroscope signals. Cannot be used.

### 4. OpenMuscle Dataset
- **Sensor Data:** Hall effect, pressure, and EMG sensors.
- **Exercises:** Finger and wrist flexions.
- **Evaluation:** Does not contain IMU kinematics. Cannot be used.

### 5. Hand and Arm Rehabilitation 2D Dataset
- **Sensor Data:** 2D Image files.
- **Evaluation:** No time-series sensor data. Cannot be used.

## Conclusion and Stop Decision

**CRITICAL DATA GAP IDENTIFIED**

None of the current datasets simultaneously contain both:
1. **Raw IMU data (Accelerometer and Gyroscope).**
2. **Ground-truth labels for the required wrist rehabilitation exercises.**

According to the strict scientific constraints of the AKIRA project:
> "If none of the datasets contains suitable labelled examples, stop before model training. Explain the data gap and recommend how to obtain appropriate recordings. Do not invent labels or fabricate sensor readings."

Consequently, Phase 2 through Phase 7 of the Random Forest pipeline **cannot be executed** at this time.

### Recommendation for Future Data Collection
To train the Random Forest exercise-recognition model successfully, a new data collection protocol must be executed using the following specifications:
- **Hardware:** Two synchronized 6-axis IMUs (Forearm and Dorsum/Back of Hand).
- **Channels:** `ax, ay, az, gx, gy, gz` from both IMUs.
- **Protocol:** Subjects must perform isolated repetitions of:
  - Wrist Flexion
  - Wrist Extension
  - Radial Deviation
  - Ulnar Deviation
  - Pronation
  - Supination
- **Annotation:** Explicit timestamps or segmented windows accurately labeling the start and end of each specific exercise.

Once this dataset is collected and ingested into `01_DATASETS/RAW/`, the Random Forest training pipeline can be safely executed without data fabrication.
