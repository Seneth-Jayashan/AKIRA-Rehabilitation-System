# Stage 1: Standardized Dataset Preparation & Schema Enforcement Report

**Project:** AI-Based Elbow Rehabilitation Assistance System (Post-Olecranon ORIF Recovery)  
**Project Location:** `D:\me\project`  
**Raw Datasets Root:** `D:\me\project\datasets\raw` (13 datasets, 68.23 GB, 55,792 files)  
**Hardware Target:** Dual 6-Axis Wearable IMU (`Upper Arm` + `Forearm` = 12 Canonical Channels)  
**Standardized Preprocessing Standard:** 100.0 Hz Uniform Sampling Rate, 2.0-Second Sliding Windows (200 frames)  
**Stage 1 Scope:** **DATASET PREPARATION ONLY** (No model training or ESP32 firmware deployment until Stage 1 is verified).

---

## 1. Executive Summary & Deliverables

In Stage 1, we established a standardized, leak-free, task-specific dataset foundation for developing and evaluating four deep-learning architectures (TCN, 1D-CNN, LSTM, GRU) and streaming real-time Bluetooth IMU data from an ESP32 microcontroller.

### Key Deliverables Produced:
1. **Standardized Dataset Module (`src/data/standardized_dataset.py`):** Canonical 12-channel assembly, anti-aliased resampling to 100 Hz, anti-leakage sliding windowing, frozen train-only z-score normalization, and live ESP32 packet transformation.
2. **Master Dataset Manifest (`data/processed/stage1/stage1_standardized_manifest.json`):** Comprehensive record documenting all 13 raw datasets, inclusion/exclusion decisions, canonical channels, task specifications, subject splits, trial counts, and anti-leakage proofs.
3. **Frozen Training Normalization Parameters (`configs/stage1/normalization_params.json`):** Channel-wise mean and standard deviation vectors fitted strictly on training partition windows (`split == "train"`), completely isolated from test subjects, serialized for repeatable evaluation and live ESP32 deployment.
4. **Standardized Preprocessing Specification (`configs/stage1/standardized_preprocessing_spec.json`):** Unified pipeline definition specifying channel orders, units, window durations, strides, and task output heads.
5. **Stage 1 Verification Test Suite (`tests/test_stage1_dataset_preparation.py`):** 10 automated unit and integration tests confirming split disjointness, ~80/20 subject ratios, zero cross-trial leakage, train-only scaler fitting, SI unit conversions, and ESP32 preprocessing parity. All **53 tests across the project now pass cleanly in 15.01s**.

---

## 2. Comprehensive Audit of all 13 Raw Datasets

All 13 public datasets in `D:\me\project\datasets\raw\` were inspected under strict read-only mode:

| # | Dataset Folder Name | Size | Files | Formats | Placement / Sensors | Sampling Rate | Total Subjects | Stage 1 Status & Decision |
| :- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `PHYTMO` | 11.68 GB | 7,076 | `.csv`, `.m` | Bilateral Upper Arm (`Rarm, Larm`) + Forearm (`Rforearm, Lforearm`) 9-axis NGIMUs | 100.89 Hz $\rightarrow$ 100.0 Hz | 30 (`A01..E05`) | **INCLUDED (PRIMARY):** Gold standard for exercise classification & execution quality |
| **2** | `Optotrak vs DOT` | 5.43 GB | 7,711 | `.csv`, `.c3d`, `.xls`, `.mat` | Right Upper Arm (`RUA`) + Right Forearm (`RLA`) Xsens DOT + Optotrak 3D optical clusters | 120.0 Hz | 16 (`15` used, `1` excluded) | **INCLUDED (PRIMARY):** Gold standard for continuous joint angles, ROM & 10 tasks |
| **3** | `ULTra-MoCap` | 12.29 GB | 1,102 | `.csv`, `.trc`, `.mot`, `.osim` | Upper Arm (`IMU 4`) + Forearm (`IMU 1`) Delsys Trigno + 60-marker Vicon OpenSim IK | 2,000 Hz $\rightarrow$ 100.0 Hz | 13 (`P01..P13`) | **INCLUDED (PRIMARY):** Multi-speed continuous angles, ROM & speed generalization |
| **4** | `JU-IMU` | 383.21 MB | 574 | `.csv` | Bilateral Upper Arm (`sensor4/5`) + Wrist/Forearm (`sensor1/2`) 9-axis IMUs | 80.0 Hz | 41 (`27 ND, 14 Stroke`) | **INCLUDED (PRIMARY):** 14-class rehabilitation exercise classification & stroke analysis |
| **5** | `ULF_in_ADL` | 24.66 GB | 4,493 | `.mvnx`, `.xlsx` | Bilateral Xsens MVN UpperArm + ForeArm + Xsens 3D joint angles | 60.0 Hz | 25 (`20 Stroke, 5 Healthy`) | **INCLUDED (AUXILIARY):** Stroke angle regression (filtered: >1,500 frame files excluded) |
| **6** | `StrokeRehab` | 5.80 GB | 1,426 | `.csv` | Bilateral Noraxon MyoMotion Upperarm + Forearm | 100.0 Hz | 20 (`C4..C32`, Healthy) | **INCLUDED (AUXILIARY):** Secondary ADL angle reference (stroke folder absent on disk) |
| **7** | `VIDIMU` | 1.38 GB | 2,005 | `.csv`, `.sto`, `.mot`, `.raw` | Right humerus + radius unit quaternions only (8 channels total) | 50.0 Hz | 16 IMU (`S40..S57`) | **INCLUDED (AUXILIARY):** Reference track for quaternion-only models |
| **8** | `REHAB` (REHAB-120) | 1.07 GB | 36,110 | `.npy` | Dual upper-limb 9-axis IMUs | 50.0 Hz | 120 (222 sessions) | **INCLUDED (AUXILIARY):** Exploratory session-split clinical FMA-UE score reference |
| **9** | `EMG elbow dataset` | 304.30 MB | 130 | `.txt` | 2-ch sEMG + 1 scalar derived angle (Zero IMU channels) | ~1,024 Hz | 10 (`subject 1..10`) | **EXCLUDED:** Lacks accelerometer and gyroscope IMU channels entirely |
| **10** | `mm-fit` | 3.53 GB | 302 | `.npy`, `.csv` | Smartwatches on wrists, earbud, pocket phones (No Upper Arm IMU) | ~100 Hz watch, 30 Hz pose | 21 workouts | **EXCLUDED:** Cannot construct upper arm + forearm dual-IMU kinematics |
| **11** | `MoCap and IMU Dataset...` | 1.45 GB | 378 | `.csv` | Single IMU per arm (wrist only + chest; no upper arm IMU) | 50.0 Hz | 6 takes (undocumented) | **EXCLUDED:** Lacks upper arm IMU and per-subject identifiers |
| **12** | `Smartwatch IMU Dataset...` | 576.41 MB | 11 | `.csv`, `.md`, `.json` | Apple Watch Series 6 on wrist only (No Upper Arm IMU) | 99.35 Hz (96.4%) & 20 Hz | 3 (`u01, u04, u05`) | **EXCLUDED:** Wrist-only sensor, underpowered N=3, corrupted mid-file headers |
| **13** | `UCI Biceps Curl` | 1.59 MB | 1 | `.csv` | Belt, arm, forearm, dumbbell (4,024-row sample excerpt) | 49.55 Hz | 5 gym lifters | **EXCLUDED:** Severe subject-class confounding in excerpt; gym dumbbell lifts |

---

## 3. Dataset Compatibility & Why Task-Specific Standardized Datasets are Required

### Physical & Biomechanical Reasons Against Blind Multi-Dataset Concatenation:
1. **Gravity Convention Discrepancy (Free vs. Proper Acceleration):**
   - `Optotrak vs DOT` records **free acceleration** (internal sensor fusion on Xsens DOT subtracts the $9.81\text{ m/s}^2$ gravity vector onboard, yielding $\|\mathbf{a}\| \approx 0\text{ m/s}^2$ at static rest).
   - `PHYTMO`, `ULTra-MoCap`, and `JU-IMU` record **proper acceleration** (includes gravity, yielding $\|\mathbf{a}\| \approx 9.81\text{ m/s}^2$ at static rest).
   - *Consequence:* Merging free-acceleration and proper-acceleration arrays into a single raw training tensor corrupts the accelerometer baseline and destroys orientation tracking.
2. **Disparate Coordinate Reference Frames and Zero-Angle Definitions:**
   - In `Optotrak vs DOT`, elbow angles are computed relative to **ISB anatomical bony landmarks** digitized via optical probes (anatomical neutral: $0^\circ$ elbow extension, $0^\circ$ forearm neutral rotation).
   - In `ULTra-MoCap`, continuous angles are computed using a **60-marker Vicon skeleton solved in OpenSim Inverse Kinematics**, where forearm pronation/supination has a biomechanical neutral offset centered at $\sim 150^\circ\text{ to }160^\circ$ (`pro_sup_r` range: $149^\circ\text{ to }175^\circ$).
   - *Consequence:* Merging raw angle targets across these two optical capture systems creates an artificial, bimodal ground truth that destabilizes neural network convergence.
3. **Sensor Coordinate Alignment:**
   - Commercial wearable IMUs place internal chip axes along differing physical directions (e.g. Xsens DOT vs. Delsys Trigno vs. NGIMU). 
4. **Where Legitimate Combination IS Possible (Classification via Invariant Features):**
   - For **exercise classification**, `Optotrak vs DOT`, `JU-IMU`, and `PHYTMO` share common upper-limb movements (`elbow_flex_ext`, `forearm_pro_sup`, `shoulder_flex`, `shoulder_abd`, `shoulder_rot`, `wrist_flex_ext`, `drinking_cup`).
   - By converting signals into **12-channel orientation-invariant features** (accelerometer magnitude, gyroscope magnitude, jerk magnitude, angular acceleration magnitude, and inter-sensor dot products / angles) resampled to 100.0 Hz, these datasets can be harmonized without coordinate frame conflicts.

---

## 4. Canonical 12-Channel Schema & Explicit SI Units

All incoming dual-IMU streams (including future live ESP32 Bluetooth packets) are mapped to this exact 12-channel order:

```text
Index 0:  ua_acc_x_ms2  (Upper Arm Accelerometer X in m/s^2)
Index 1:  ua_acc_y_ms2  (Upper Arm Accelerometer Y in m/s^2)
Index 2:  ua_acc_z_ms2  (Upper Arm Accelerometer Z in m/s^2)
Index 3:  ua_gyr_x_rads (Upper Arm Gyroscope X in rad/s)
Index 4:  ua_gyr_y_rads (Upper Arm Gyroscope Y in rad/s)
Index 5:  ua_gyr_z_rads (Upper Arm Gyroscope Z in rad/s)
Index 6:  fa_acc_x_ms2  (Forearm Accelerometer X in m/s^2)
Index 7:  fa_acc_y_ms2  (Forearm Accelerometer Y in m/s^2)
Index 8:  fa_acc_z_ms2  (Forearm Accelerometer Z in m/s^2)
Index 9:  fa_gyr_x_rads (Forearm Gyroscope X in rad/s)
Index 10: fa_gyr_y_rads (Forearm Gyroscope Y in rad/s)
Index 11: fa_gyr_z_rads (Forearm Gyroscope Z in rad/s)
```

### Explicit Unit Conversion Rules:
- **Accelerometer:**
  - If unit is $g$: multiply by $9.80665\text{ m/s}^2$.
  - If unit is $\text{mG}$ or $\text{mm/s}^2$: multiply by $(9.80665 / 1000.0)$.
  - If unit is $\text{m/s}^2$: identity pass-through.
- **Gyroscope:**
  - If unit is $\text{deg/s}$: multiply by $(\pi / 180.0)\text{ rad/s}$.
  - If unit is $\text{rad/s}$: identity pass-through.
- **Target Joint Angles:**
  - If unit is radians: multiply by $(180.0 / \pi)^\circ$.
  - If unit is degrees: identity pass-through.

---

## 5. Subject-Independent Splits & Leakage Prevention

Subject partitioning strictly enforces that **no subject appears in more than one partition**. Validation subjects are chosen strictly from the training cohort:

| Dataset | Total Subjects | Train Subjects | Val Subjects | Test Subjects | Train + Val % | Test % | Partition Isolation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **PHYTMO** | 30 | 20 | 5 | 5 | 83.33% | 16.67% | **100% Disjoint (0 overlap)** |
| **Optotrak (Classification)** | 15 | 9 | 3 | 3 | 80.00% | 20.00% | **100% Disjoint (0 overlap)** |
| **Optotrak (Angle Regr.)** | 14 | 8 | 3 | 3 | 78.57% | 21.43% | **100% Disjoint (0 overlap)** |
| **JU-IMU** | 41 | 29 | 6 | 6 | 85.37% | 14.63% | **100% Disjoint (0 overlap)** |
| **ULTra-MoCap** | 13 | 9 | 2 | 2 | 84.62% | 15.38% | **100% Disjoint (0 overlap)** |
| **StrokeRehab** | 20 | 14 | 3 | 3 | 85.00% | 15.00% | **100% Disjoint (0 overlap)** |
| **ULF_in_ADL** | 25 | 17 | 4 | 4 | 84.00% | 16.00% | **100% Disjoint (0 overlap)** |
| **VIDIMU** | 16 | 10 | 3 | 3 | 81.25% | 18.75% | **100% Disjoint (0 overlap)** |

### Strict Anti-Leakage Safeguards Enforced:
1. **Zero Subject Overlap:** $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
2. **Boundary-Isolated Windowing:** `extract_trial_windows()` extracts windows solely within a single trial array of length $T$. Windows never span across trials or across subjects.
3. **Train-Only Normalization Fitting:** `ChannelStandardScaler.fit()` enforces `split_name == "train"`. Attempting to fit on validation or test raises a `ValueError`. Mean and standard deviation are computed exclusively from training data and frozen into `configs/stage1/normalization_params.json`.
4. **ULF_in_ADL P28 Safeguard:** Excludes unsegmented recordings $>1,500$ frames ($>5\text{ MB}$) from test evaluation, resolving the anomaly where subject P28 constituted $99.07\%$ of test windows.

---

## 6. Preprocessing Parity for Live ESP32 Bluetooth Data

When real-time IMU packets stream over Bluetooth from the ESP32 (dual 6-axis IMU), `process_live_esp32_sample()` executes the exact same pipeline:
1. Re-orders incoming channels into canonical `[UA_acc_XYZ, UA_gyr_XYZ, FA_acc_XYZ, FA_gyr_XYZ]`.
2. Converts raw units into standard SI ($\text{m/s}^2$ and $\text{rad/s}$).
3. Interpolates/resamples to uniform 100.0 Hz.
4. Normalizes using the frozen training parameters (`mean_`, `std_`, `eps = 1e-8`) loaded from `configs/stage1/normalization_params.json`.

---

## 7. Verification Test Results

We executed the full test suite using `pytest`:
```bash
python -m pytest tests/
```
**Results:**
- `tests/test_stage1_dataset_preparation.py`: **10 passed**
  - Manifest & config existence & schema validation
  - Canonical 12-channel order & sensor specifications
  - Pairwise disjoint subject partitions across all datasets
  - ~80/20 train/test subject ratios
  - Boundary-isolated window extraction (zero cross-trial leakage)
  - Train-only normalization fitting and frozen test transformation
  - Dataset compatibility rules (blocking incompatible physical merges)
  - Live ESP32 sample preprocessing parity
  - Completeness of documented exclusions
  - Task specification completeness
- `tests/test_pipeline_and_splits.py`: **8 passed**
- `tests/test_stage10_readiness_and_preview.py`: **8 passed**
- `tests/test_supervisor_demo.py`: **27 passed**
- **TOTAL:** **53 passed in 15.01s (0 failed, 0 warnings)**

---

## 8. Summary of Files Created and Modified

1. **Created:**
   - [`src/data/standardized_dataset.py`](file:///d:/me/project/src/data/standardized_dataset.py): Master Stage 1 dataset curation, schema enforcement, anti-leakage windowing, and ESP32 preprocessing module.
   - [`data/processed/stage1/stage1_standardized_manifest.json`](file:///d:/me/project/data/processed/stage1/stage1_standardized_manifest.json): Complete Stage 1 manifest.
   - [`configs/stage1/normalization_params.json`](file:///d:/me/project/configs/stage1/normalization_params.json): Serialized frozen training normalization parameters.
   - [`configs/stage1/standardized_preprocessing_spec.json`](file:///d:/me/project/configs/stage1/standardized_preprocessing_spec.json): Standardized preprocessing specification.
   - [`tests/test_stage1_dataset_preparation.py`](file:///d:/me/project/tests/test_stage1_dataset_preparation.py): Stage 1 test suite (10 tests).
   - [`docs/STAGE1_DATASET_STANDARDIZATION_REPORT.md`](file:///d:/me/project/docs/STAGE1_DATASET_STANDARDIZATION_REPORT.md): This report.
2. **Preserved / Unchanged:**
   - All 64 saved model checkpoints in `models/checkpoints/`.
   - All 13 raw public datasets in `datasets/raw/`.
   - All previous benchmark reports, logs, and supervisor demo backend code.

---

## 9. Stage 1 Verification Status & Next Steps

Stage 1 is **100% complete and verified**. All requirements have been executed and tested. 

Per your instructions:
- **No models have been trained or retrained yet.**
- **No ESP32 firmware deployment has occurred yet.**
- We are ready to proceed to **Stage 2 (Model Architecture & Benchmark Standardization)** upon your explicit approval.
