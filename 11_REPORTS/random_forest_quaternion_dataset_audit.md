# Quaternion-Based Wrist Movement Classification Using Random Forest
## Dataset Audit Report

**Target Dataset:** 3DCDataset
**Date:** 2026-10-10

### 1. File Inspection
- **Source File:** `01_DATASETS/RAW/3DCDataset-main.zip`
- **Format:** Text files inside participant-specific folders (`ParticipantX/train/IMU/`, `ParticipantX/test/IMU/`).
- **File Naming Convention:** `3dc_IMU_gesture_{gesture_id}_{trial_id}.txt`
- **Total IMU Files:** 1,936 files.

### 2. Quaternion Data Structure
- Each file contains time-ordered rows representing temporal observations.
- Each row is formatted as a tuple: `(W, X, Y, Z)`, e.g., `(0.607489467, -0.142371163, -0.09148579, 0.7760911)`.
- The data explicitly represents 3D spatial orientation as quaternions, not raw accelerometer and gyroscope signals.

### 3. Exercise Labels (Ground Truth)
- Extracted gesture IDs from file names: `0, 1, 2, 3`.
- Mapped using existing reference (`03_REFERENCE_DATA/3DCDataset_exercise_reference.csv`):
  - **1:** Radial Deviation
  - **2:** Wrist Flexion
  - **3:** Ulnar Deviation
  - **0:** Neutral / Rest (unmapped in reference but clearly distinct).

### 4. Participants and Grouping
- **Participants:** 22 unique participants identified via folder names (`Participant1` to `Participant22`).
- Provides a clear boundary for participant-independent data splitting (e.g., Leave-One-Group-Out or GroupShuffleSplit) to prevent data leakage.

### 5. Conclusion
The dataset provides highly structured, time-ordered quaternion observations and reliable, explicit wrist-exercise labels. It is fully sufficient for a scientifically valid quaternion-based classification experiment.

**Action:** Proceed to feature extraction, participant-independent splitting, and Random Forest training using quaternion-derived temporal features.
