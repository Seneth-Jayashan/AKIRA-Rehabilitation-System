# Stage 10.1 — Dataset & Task Compatibility Matrix (`docs/STAGE10_DATASET_TASK_MATRIX.md`)

**Project:** AI-Based Elbow Rehabilitation Assistance System After Olecranon Fracture ORIF  
**Workspace Root:** `D:\me\project`  
**Raw Datasets Root:** `D:\me\project\datasets\raw` (`13` datasets, `68.23 GB` total)  
**Audit Date:** 2026-10-19

---

## 1. Master Task Compatibility Matrix Across All 13 Raw Datasets

Each dataset in `D:\me\project\datasets\raw` was evaluated against the **5 core post-olecranon ORIF rehabilitation tasks** plus the **auxiliary neurological/stroke impairment task**:

1. **Task 1 — Elbow Flexion/Extension Angle Prediction (`deg`)**
2. **Task 2 — Forearm Pronation/Supination Angle Prediction (`deg`)**
3. **Task 3 — Full-Repetition Range-of-Motion (`ROM`) Estimation (`deg`)**
4. **Task 4 — Rehabilitation Exercise Classification**
5. **Task 5 — Correct vs. Incorrect Movement Execution Classification**
6. **Task 6 (Auxiliary) — Stroke vs. Non-Disabled / Clinical `FMA-UE` Impairment Analysis**

| # | Dataset Name | Stage 10 Role Classification | Task 1: Elbow Flex/Ext (`deg`) | Task 2: Forearm Pro/Sup (`deg`) | Task 3: Full-Rep ROM (`deg`) | Task 4: Exercise Class | Task 5: Correct vs. Incorrect | Task 6 (Aux): Stroke / FMA-UE | Subject-Disjoint Split? |
| :-: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **`Optotrak vs DOT`** | **Training + Sample Inference Preview + Real-World Validation** | **YES (Primary)**<br>`Elbow[:, 2]` ISB Optical GT | **YES (Primary)**<br>`Elbow[:, 1]` ISB Optical GT | **YES (Primary)**<br>Optical peak-to-trough ROM | **YES (Primary)**<br>10-class Rehab & ADL (`1A`) + 7-class Transfer (`5A-1`) | No (Healthy normal executions only) | No (15 Healthy subjects) | **YES** (`10 / 2 / 3` subjects; `9 / 2 / 3` on `3A`) |
| **2** | **`ULTra-MoCap`** | **Training + Sample Inference Preview + Real-World Validation** | **YES (Primary)**<br>OpenSim IK `elbow_flex_r` | **YES (Primary)**<br>OpenSim IK `pro_sup_r` | **YES (Primary)**<br>Multi-speed OpenSim ROM | **YES (Secondary)**<br>3-class exercise (`Track 5B`) | No (Healthy normal executions only) | No (13 Healthy subjects) | **YES** on `Track 3C` (`9 / 2 / 2` subj); Speed-split on `5B` |
| **3** | **`PHYTMO`** | **Training + Sample Inference Preview + Real-World Validation** | No continuous angle GT | No continuous angle GT | No continuous angle GT | **YES (Primary)**<br>3-class (`EFE, EAH, SQZ`, `Track 1C`) + Transfer (`5A-2`) | **YES (Exclusive Primary)**<br>Binary `EFE0/EFE1` (`2A`) & 6-class (`2B`) | No (30 Healthy volunteers) | **YES** (`20 / 5 / 5` subjects on `1C/2B`; `18 / 4 / 5` on `2A`) |
| **4** | **`JU-IMU`** | **Training + Sample Inference Preview + Auxiliary Stroke Analysis** | No continuous angle GT | No continuous angle GT | No continuous angle GT | **YES (Primary)**<br>14-class upper-limb rehab (`Track 1B`) + Transfer (`5A-1`) | **NO** *(Never fabricate quality labels from Stroke vs ND)* | **YES (Primary Aux)**<br>`Stroke` (14) vs `ND` (27) (`Track 4A`) | **YES** (`29 / 6 / 6` subjects stratified by cohort) |
| **5** | **`VIDIMU`** | **Auxiliary Analysis Only (8-Ch Quaternion Baseline)** | **YES (Secondary)**<br>OpenSim IK `elbow_flex_ext` | **YES (Secondary)**<br>OpenSim IK `elbow_pro_sup` | **YES (Secondary)**<br>OpenSim IK ROM (`8.73°` MAE) | Supports 9 upper-limb activity codes (`A05..A13`) | No | No (16 Healthy subjects) | **YES** (`10 / 3 / 3` subjects) |
| **6** | **`StrokeRehab`** | **Auxiliary Analysis Only (Healthy Control Subset)** | **YES (Auxiliary)**<br>Control `_angles.csv` `elbow_Z` | **Weak / Aux**<br>Measures `wrist_supination` without raw gyro | **Auxiliary Only**<br>Healthy Control ADL ROM | Has 9 complex ADL tasks (no Stage 8 cls track) | No | Excluded (`51` Stroke patients lack `_angles.csv`) | **YES** (`14 / 3 / 3` healthy control subjects) |
| **7** | **`ULF_in_ADL`** | **Auxiliary Analysis Only (Requires `P28` Filtering)** | **Conditional Aux**<br>Xsens MVN `jRightElbow[:, 2]` *(after `P28` filter)* | **NO (Unreliable)**<br>Unscripted ADL $R^2 < 0$ | **Conditional Aux**<br>After filtering `>1,500`-frame files | 17 unscripted ADL tasks | No | **Excluded (`Track 4B`)**<br>`Macro-F1 = 0.36` & `P28` skew | **YES** (`17 / 4 / 4` subjects, but `P28` skews raw test split) |
| **8** | **`REHAB-120`** | **Auxiliary Analysis Only (Clinical `FMA-UE` Session Split)** | No continuous angle GT | No continuous angle GT | No continuous angle GT | No (Single shoulder flexion task) | No | **Auxiliary Only**<br>3-class `FMA-UE` (`0, 1, 2`, `Track 4C`) | **NO** (Session split `154 / 34 / 34`; no patient ID map) |
| **9** | **`EMG elbow dataset`** | **Excluded** | No | No | No | No | No | No | Excluded (Surface EMG only; no 6-axis IMU) |
| **10** | **`mm-fit`** | **Excluded** | No | No | No | No | No | No | Excluded (Earbud/Smartwatch only; no upper-arm + forearm IMU) |
| **11** | **`MoCap and IMU Dataset...`** | **Excluded** | Unusable | Unusable | Unusable | No | No | No | Excluded (Uncalibrated raw integer ticks + unmapped markers) |
| **12** | **`Smartwatch IMU Dataset...`** | **Excluded** | No | No | No | No | No | No | Excluded (Single wrist smartwatch IMU only; no upper-arm IMU) |
| **13** | **`UCI Biceps Curl`** | **Excluded** | No | No | No | No | **Invalid** (100% subject-class confounded) | No | Excluded (Each of 5 subjects performed only 1 class `A..E`) |

---

## 2. Detailed Dataset Audit Profiles (8 Active Benchmark Datasets)

### 2.1 `Optotrak vs DOT` — Primary Angle/ROM & 10-Class Exercise Dataset
- **Exact Path:** [`d:\me\project\datasets\raw\Optotrak vs DOT`](file:///d:/me/project/datasets/raw/Optotrak%20vs%20DOT) (`606` `.mat` files, `0.34 GB`).
- **Subjects & Recordings:** `15` healthy subjects (`subject 1` .. `subject 15`); `14` subjects in `Track 3A` (`subject 10` lacks synchronized Optotrak optical angle trajectories). `~287` dual-IMU trials across `10` rehabilitation and ADL tasks (`Task 11` Elbow Flexion-Extension through `Task 20` Reaching).
- **Sensor Placement & Channels:** Xsens DOT wearable IMUs on **Upper Arm (`Sensor 2`)** and **Forearm (`Sensor 3`)**; 12 channels mapped to `CANONICAL_DUAL_IMU_12CH`.
- **Sampling Rate & Units:** `120.0 Hz` native (`W = 240`, `S = 120`). **Important physical unit note:** `Acc` is recorded as Xsens DOT **free acceleration** (`m/s²`, gravity subtracted by onboard strapdown integration); `Gyr` is converted from `deg/s` to `rad/s`.
- **Ground-Truth Labels & Units:**
  - **Classification (`Track 1A`, `Track 5A-1`, `Track 5A-2`):** 10 exercise classes (`elbow_flex_ext`, `forearm_pro_sup`, `wrist_flex_ext`, `shoulder_flex`, `shoulder_abd`, `shoulder_int_ext_rot`, `drinking_cup`, `box_off_shelf`, `drawing_circles`, `reaching`).
  - **Regression (`Track 3A`):** Certus Optotrak 3D optical motion capture solved via ISB anatomical conventions (`Elbow[:, 2]` = Elbow Flexion-Extension in `deg`, `Elbow[:, 1]` = Forearm Pronation-Supination in `deg`).
- **Preprocessing Required:** `load_optotrak_dot_trial_12ch()` in [`src/data/loaders.py`](file:///d:/me/project/src/data/loaders.py#L121-L185): internal NaN interpolation (`max_gap=15`), clipping extreme numerical spikes to $\pm 100\text{ m/s}^2$, `deg/s -> rad/s`, and train-only `ChannelStandardScaler`.
- **Subject-Independent Splits:**
  - `Track 1A`: Train (`10` subj, `2,668` win) / Val (`2` subj: `s6, s12`, `947` win) / Test (`3` subj: `s5, s11, s15`, `731` win).
  - `Track 3A`: Train (`9` subj, `2,387` win) / Val (`2` subj: `s6, s12`, `947` win) / Test (`3` subj: `s5, s11, s15`, `731` win).
- **Known Data-Quality Issues:** (1) `subject 10` lacks optical ground truth (`Track 3A` automatically excludes `subject 10`); (2) `subject 5` exhibits a static `~35°` offset in forearm pronation-supination (`ProSup MAE = 35.00°` vs `Flexion MAE = 11.64°`) due to uncalibrated forearm sensor heading placement; (3) free acceleration lacks static gravity vector unless high-pass matched when transferring to proper-acceleration datasets.
- **Stage 10 Suitability:** **`Training`**, **`Sample Inference Preview`**, and **`Real-World Validation`**.

---

### 2.2 `ULTra-MoCap` — Primary Multi-Speed Angle/ROM & Cross-Speed Dataset
- **Exact Path:** [`d:\me\project\datasets\raw\ULTra-MoCap`](file:///d:/me/project/datasets/raw/ULTra-MoCap) (`698` files across `.csv`, `.mat`, `.c3d`, `9.29 GB`).
- **Subjects & Recordings:** `13` healthy subjects (`S1..S13`), performing 3 upper-limb exercise categories (`Elbow_Flexion`, `Forearm_Pronation`, `Reaching`) across 4 speed regimes (`Slow`, `Normal`, `Fast`, `VeryFast`).
- **Sensor Placement & Channels:** Upper-arm IMU (`IMU2` / `UA`) and Forearm IMU (`IMU3` / `FA`); 12 channels mapped to `CANONICAL_DUAL_IMU_12CH`.
- **Sampling Rate & Units:** Raw IMU at `100.0 Hz` (`acc` in `mm/s²` converted to `m/s²` via $\div 1000.0$; `gyr` in `rad/s`), synchronized with `60.0 Hz` Vicon optical MoCap OpenSim Inverse Kinematics (IK) resampled to `100.0 Hz` (`W = 200`, `S = 100`).
- **Ground-Truth Labels & Units:**
  - **Regression (`Track 3C`):** OpenSim IK continuous joint angles (`elbow_flex_r` and `pro_sup_r` in `deg`).
  - **Classification (`Track 5B`):** 3 exercise classes (`elbow_flex_ext`, `forearm_pro_sup`, `reaching_adl`).
- **Preprocessing Required:** `load_ultra_mocap_trial_12ch()` in [`src/data/loaders.py`](file:///d:/me/project/src/data/loaders.py#L276-L326): `mm/s² -> m/s²`, anti-aliased cubic/polyphase resampling of `60 Hz` OpenSim IK angles to `100 Hz`, and train-only `ChannelStandardScaler`.
- **Subject-Independent Splits:**
  - `Track 3C` (Subject-Independent): Train (`9` subj, `5,384` win) / Val (`2` subj, `1,418` win) / Test (`2` subj, `1,382` win).
  - `Track 5B` (Cross-Speed Split): Train/Val on `Normal` speed (`1,427 / 312` win) $\rightarrow$ Test on `Slow, Fast, VeryFast` (`4,099` win). *Note: Same 13 subjects appear across speeds in Track 5B.*
- **Known Data-Quality Issues:** Raw accelerations are in `mm/s²` (already handled in `load_ultra_mocap_trial_12ch`); `Track 5B` tests speed generalization rather than unseen subjects.
- **Stage 10 Suitability:** **`Training`**, **`Sample Inference Preview`**, and **`Real-World Validation`** (`Track 3C`); **`Auxiliary Analysis Only`** (`Track 5B` cross-speed).

---

### 2.3 `PHYTMO` — Primary Exercise Classification & Exclusive Execution Quality Dataset
- **Exact Path:** [`d:\me\project\datasets\raw\PHYTMO`](file:///d:/me/project/datasets/raw/PHYTMO) (`1,656` `.csv` files, `1.54 GB`).
- **Subjects & Recordings:** `30` healthy volunteers (`A01..G03`) in 3-class/6-class tracks (`1C, 2B`); `27` subjects in binary `EFE` quality track (`2A`).
- **Sensor Placement & Channels:** Shimmer3 IMUs on **Upper Arm (`UA`)** and **Forearm (`FA`)**; 12 channels mapped to `CANONICAL_DUAL_IMU_12CH`.
- **Sampling Rate & Units:** `100.0 Hz` (`W = 200`, `S = 100`). Accelerometer in `g` converted to `m/s²` ($\times 9.80665$); Gyroscope in `deg/s` converted to `rad/s` ($\times \pi/180$).
- **Ground-Truth Labels & Units:**
  - **`Track 1C`:** 3 rehabilitation exercises: `EFE` (Elbow Flexion-Extension), `EAH` (Elevation of Arm / Shoulder), `SQZ` (Isometric Hand/Forearm Squeeze).
  - **`Track 2A`:** Binary Elbow Flexion-Extension execution quality: `EFE0_Correct` (`0`) vs. `EFE1_Incorrect` (`1`).
  - **`Track 2B`:** 6-class joint Exercise $\times$ Quality (`EFE0_Correct`, `EFE1_Incorrect`, `EAH0_Correct`, `EAH1_Incorrect`, `SQZ0_Correct`, `SQZ1_Incorrect`).
  - **No continuous joint angle ground truth** (optical markers in `PHYTMO` are uncalibrated 3D positions without elbow/forearm anatomical angle inverse kinematics).
- **Preprocessing Required:** `load_phytmo_trial_12ch()` in [`src/data/loaders.py`](file:///d:/me/project/src/data/loaders.py#L71-L118): deduplicate timestamps, interpolate up to 15 internal NaNs, convert `g -> m/s²` and `deg/s -> rad/s`, and apply train-only `ChannelStandardScaler`.
- **Subject-Independent Splits:**
  - `Tracks 1C & 2B`: Train (`20` subj, `8,815` win) / Val (`5` subj, `3,117` win) / Test (`5` subj: `A03, B02, C02, D02, E02`, `2,503` win).
  - `Track 2A`: Train (`18` subj, `3,705` win) / Val (`4` subj, `1,483` win) / Test (`5` subj: `A03, B02, C02, D02, E02`, `1,238` win).
- **Known Data-Quality Issues:** (1) Contains internal NaN dropouts and occasional duplicate timestamps (handled by loader); (2) `Incorrect` (`1`) labels represent healthy volunteers performing scripted compensatory errors rather than post-olecranon ORIF surgical patients.
- **Stage 10 Suitability:** **`Training`**, **`Sample Inference Preview`**, and **`Real-World Validation`** for Exercise Classification (`Task 4`) and Correct/Incorrect Execution Quality (`Task 5`).

---

### 2.4 `JU-IMU` — Primary 14-Class Exercise & Auxiliary Stroke vs. Non-Disabled Dataset
- **Exact Path:** [`d:\me\project\datasets\raw\JU-IMU`](file:///d:/me/project/datasets/raw/JU-IMU) (`3,205` `.csv` files, `0.36 GB`).
- **Subjects & Recordings:** `41` subjects total (`27` Non-Disabled `ND` controls + `14` chronic `Stroke` patients), performing `14` upper-limb rehabilitation exercises (`ElbFlex`, `SupPro`, `FrontSupPro`, `WrstFlex`, `WrstExt`, `ShFlex90`, `ShFlex180`, `ShAbd90`, `ShAbd180`, `ShExt`, `ShRot`, `PickCup`, `Write`, `DoorKnock`).
- **Sensor Placement & Channels:** Custom 6-axis IMUs on **Upper Arm (`S2` / `UA`)** and **Forearm (`S3` / `FA`)**; 12 channels mapped to `CANONICAL_DUAL_IMU_12CH`.
- **Sampling Rate & Units:** `80.0 Hz` native (`W = 160`, `S = 80`); resampled to `100.0 Hz` (`W = 200`) in cross-dataset Tracks `5A-1` and `5A-2`. Accelerometer in `mG` converted to `m/s²` ($\times 9.80665/1000$); Gyroscope in `deg/s` converted to `rad/s`.
- **Ground-Truth Labels & Units:**
  - **`Track 1B`:** 14-class exercise label (`0..13`).
  - **`Track 4A`:** Binary clinical cohort (`0 = ND`, `1 = Stroke`).
  - **No continuous joint angle ground truth** and **no execution quality labels**.
- **Subject-Independent Splits:** Train (`29` subj: `19 ND + 10 Stroke`, `4,305` win) / Val (`6` subj: `4 ND + 2 Stroke`, `1,160` win) / Test (`6` subj: `ND8, ND18, ND25, ND31, Stroke6, Stroke13`, `979` win).
- **Known Data-Quality Issues:** Severe cohort performance disparity on 14-class recognition (`97.56%` test accuracy on `ND` subjects vs. `34.98%` on `Stroke` subjects because severely paretic `Stroke6` exhibits abnormal flexion synergy).
- **Stage 10 Suitability:** **`Training`** and **`Sample Inference Preview`** for 14-class Exercise Classification (`Task 4`) and **`Auxiliary Analysis Only`** for Stroke vs. ND impairment (`Task 6`).

---

### 2.5 `VIDIMU` — Auxiliary Quaternion-to-Angle Regression Dataset
- **Exact Path:** [`d:\me\project\datasets\raw\VIDIMU`](file:///d:/me/project/datasets/raw/VIDIMU) (`1,341` `.csv`/`.mot`/`.mp4` files, `2.65 GB`).
- **Subjects & Recordings:** `16` healthy subjects (`S01..S56` subset with complete IMU + OpenSim `.mot` pairs), performing upper-limb activities `A05..A13`.
- **Sensor Placement & Channels:** Upper-arm and forearm IMUs providing **8-channel unit quaternions ONLY** (`[ua_q0..q3, fa_q0..q3]`). Does **not** contain raw 12-channel linear acceleration (`m/s²`) and angular velocity (`rad/s`).
- **Sampling Rate & Units:** `50.0 Hz` (`W = 100`, `S = 50`).
- **Ground-Truth Labels & Units:** OpenSim IK continuous `elbow_flex_ext_deg` and `forearm_pro_sup_deg` (`Track 3E`).
- **Subject-Independent Splits:** Train (`10` subj, `4,054` win) / Val (`3` subj, `1,187` win) / Test (`3` subj, `1,110` win).
- **Stage 10 Suitability:** **`Auxiliary Analysis Only`** (useful to benchmark onboard DMP/AHRS quaternion fusion where `TCN` achieves `13.98°` MAE and `8.73°` ROM MAE, but incompatible with the 12-channel raw SI pipeline).

---

### 2.6 `StrokeRehab` — Auxiliary Healthy Control ADL Angle Regression Dataset
- **Exact Path:** [`d:\me\project\datasets\raw\StrokeRehab`](file:///d:/me/project/datasets/raw/StrokeRehab) (`6,419` files, `16.48 GB`).
- **Subjects & Recordings:** `20` Healthy Control subjects (`C1..C20`) with synchronized `_imu.csv` and `_angles.csv` across 9 complex ADLs (`deodorant`, `drinking`, `feeding`, `glasses`, `hair`, `otti`, `shelf`, `table`, `teeth`). (`51` Stroke patients in `.parquet` format lack `_angles.csv` and are excluded from `Track 3B`).
- **Sensor Placement & Channels:** Upper-arm and forearm IMUs at `100.0 Hz` (`W = 200`, `S = 100`). Raw files store linear acceleration (`g`) and 3D Euler orientation angles (`deg`); `load_strokerehab_trial_12ch()` differentiates unwrapped Euler angles to approximate angular velocity (`rad/s`).
- **Ground-Truth Labels & Units:** `elbow_Z` (Elbow Flexion-Extension in `deg`) and `wrist_supination` (Forearm/Wrist Supination in `deg`).
- **Subject-Independent Splits:** Train (`14` subj, `8,181` win) / Val (`3` subj, `2,147` win) / Test (`3` subj, `2,170` win).
- **Stage 10 Suitability:** **`Auxiliary Analysis Only`** (synthetic differentiated gyro + complex unscripted ADLs).

---

### 2.7 `ULF_in_ADL` — Auxiliary Unscripted ADL Dataset (Requires `P28` Anomaly Filter)
- **Exact Path:** [`d:\me\project\datasets\raw\ULF_in_ADL`](file:///d:/me/project/datasets/raw/ULF_in_ADL) (`545` `.mvnx` XML files, `12.69 GB`).
- **Subjects & Recordings:** `25` subjects (`5` Healthy `H01..H05` + `20` Stroke `P01..P28` subset).
- **Sensor Placement & Channels:** Xsens MVN full-body suit; `RightUpperArm` and `RightForeArm` extracted as 12 SI channels (`m/s²`, `rad/s`) at `60.0 Hz` (`W = 120`, `S = 60`).
- **Ground-Truth Labels & Units:** Xsens MVN biomechanical model `jRightElbow[:, 2]` (Flexion-Extension `deg`) and `jRightElbow[:, 1]` (Pronation-Supination `deg`) (`Track 3D`), plus 3-class limb status (`Healthy`, `Non-Paretic`, `Paretic`, `Track 4B`).
- **Known Data-Quality Issues:** **Critical `P28` File-Duration Skew:** `511 / 545` `.mvnx` files across the dataset exceed `1,500` frames (`25 s`), and test subject `P28` has 13 continuous multi-minute `.mvnx` files (`299.85 MB`, `~39,935` frames each), generating `7,839 / 8,848` (`88.6%`) of all test windows (`582 train / 143 val / 8,848 test`). In addition, `Track 4B` has near-random 3-class accuracy (`Macro-F1 = 0.3091–0.3620`).
- **Stage 10 Suitability:** **`Excluded`** for `Track 4B`; **`Auxiliary Analysis Only`** for `Track 3D` Elbow Flexion-Extension **only after filtering out `.mvnx` files `> 1,500` frames**.

---

### 2.8 `REHAB-120` — Auxiliary Clinical `FMA-UE` Session-Split Dataset
- **Exact Path:** [`d:\me\project\datasets\raw\REHAB`](file:///d:/me/project/datasets/raw/REHAB) (`223` files, `0.06 GB`).
- **Subjects & Recordings:** `222` numbered sessions from `120` stroke patients performing a single shoulder flexion rehabilitation task (`50.0 Hz`, `W = 150`, `S = 75`, 12 SI channels).
- **Ground-Truth Labels & Units:** 3-class clinical Fugl-Meyer Assessment Upper-Extremity (`FMA-UE`) item score (`0 = Cannot Perform`, `1 = Partial`, `2 = Full`, `Track 4C`).
- **Known Data-Quality Issues:** No mapping exists between the `222` session IDs and the `120` patient IDs; therefore splits (`154 / 34 / 34` sessions) are session-disjoint rather than guaranteed patient-disjoint.
- **Stage 10 Suitability:** **`Auxiliary Analysis Only`**.
