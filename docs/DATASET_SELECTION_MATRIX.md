# Dataset Selection Matrix (`docs/DATASET_SELECTION_MATRIX.md`)

**Project:** AI-Based Elbow Rehabilitation Assistance System (Post-Olecranon ORIF Recovery)  
**Workspace Root:** `D:\me\project`  
**Raw Datasets Directory:** `D:\me\project\datasets\raw` (`13` datasets, `~68.23 GB`, `55,792` files)  
**Target Models:** `TCN`, `1D CNN`, `LSTM`, `GRU`  
**Target Hardware Alignment:** Dual 6-Axis IMU (`IMU 1`: Upper Arm Acc XYZ + Gyr XYZ; `IMU 2`: Forearm Acc XYZ + Gyr XYZ $\rightarrow 12\text{ channels}$)

---

## 1. Task Definitions (Tasks A–F)

Every dataset in `D:\me\project\datasets\raw` has been evaluated against six research tasks:

- **Task A — Rehabilitation Exercise Classification:** Classify discrete upper-limb rehabilitation exercises and functional ROM tasks from wearable IMU time series.
- **Task B — Correct-vs.-Incorrect Exercise Execution Classification:** Distinguish proper rehabilitation exercise form from compensatory or erroneous execution using verified ground-truth quality labels.
- **Task C — Continuous Elbow Flexion-Extension Angle & ROM Estimation:** Regress continuous elbow flexion-extension angle $\theta_{\text{flex}}(t)$ (in degrees) and estimate trial/window Range of Motion ($\text{ROM}_{\text{flex}} = \max \theta_{\text{flex}} - \min \theta_{\text{flex}}$).
- **Task D — Continuous Forearm Pronation-Supination Angle & ROM Estimation:** Regress continuous forearm pronation-supination angle $\theta_{\text{pro/sup}}(t)$ (in degrees) and estimate $\text{ROM}_{\text{pro/sup}}$.
- **Task E — Stroke-Related or Impaired Upper-Limb Movement Analysis:** Analyze hemiparetic/impaired vs. non-disabled/unaffected movement patterns or predict clinical motor impairment scores (`FMA-UE` / `FMMA-UE`).
- **Task F — Generalization Across Subjects, Datasets, Sensor Configurations, and Movement Conditions:** Evaluate out-of-subject generalization, cross-dataset zero-shot/transfer performance on genuinely equivalent exercises, and robustness across movement speeds (`Slow`, `Normal`, `Fast`, `VeryFast`) and target ROMs.

---

## 2. Master Dataset Selection Matrix (All 13 Datasets × Tasks A–F)

**Legend:**
- **PRIMARY (`[P]`)**: Directly supports the task with verified Upper Arm + Forearm dual-IMU signals, clean subject IDs, and ground-truth labels. Included in core 12-channel dual-IMU benchmarks.
- **SECONDARY / SEPARATE TRACK (`[S]`)**: Supports the task with verified ground-truth labels, but uses an alternative sensor representation (e.g., quaternions only in `VIDIMU`, Noraxon Acc + quaternion vector in `StrokeRehab`, unmapped longitudinal sessions in `REHAB`, or unnamed 6-IMU indices in `ULTra-MoCap`). Included in dedicated separate benchmark tracks.
- **EXCLUDED (`[X]`)**: Unsuitable for the task due to missing target labels, missing upper-arm IMU, missing raw IMU channels, or severe subject-class confounding.

| # | Dataset Name | Priority Group | Subj / Sessions | Subj-Indep. Split? | Dual-IMU (UA + FA)? | Task A: Exercise Classif. | Task B: Correct vs Incorrect | Task C: Elbow Flex/Ext Angle & ROM | Task D: Forearm Pron/Sup Angle | Task E: Impaired / Stroke Analysis | Task F: Cross-Subj / Cond / Dataset Gen. | Assigned Benchmark Track(s) |
| :- | :--- | :--- | :--- | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :--- |
| **1** | **`PHYTMO`** | Priority 8 | 30 subj (`342` upper-limb trials) | **Yes** (`A01..E05`) | **Yes** (`Rarm`+`Rforearm`, `Larm`+`Lforearm` 9-DOF) | **`[P]`** (3 upper exercises: `EFE, EAH, SQZ`) | **`[P]`** (`0` Correct vs `1` Incorrect on `EFE, EAH, SQZ`) | `[X]` (Optical rigid bodies have 33–55% occlusion NaNs) | `[X]` (No pron/sup task or GT angle) | `[X]` (Healthy cohorts `A–E` only) | **`[P]`** (Subj-indep + Cross-dataset `EFE`/`EAH`) | **Track 1C, Track 2A, Track 2B, Track 5A** |
| **2** | **`Optotrak vs DOT`** | Priority 8 | 16 subj (`320` trials: 160 calib + 160 eval) | **Yes** (`subject 1..16`) | **Yes** (`RUA` + `RLA` Xsens DOT 9-DOF + Quat) | **`[P]`** (10 rehab/ADL eval tasks `11..20`) | `[X]` (Healthy normal execution only) | **`[P]`** (120 Hz Optotrak GT `Elbow[:,2]` in deg) | **`[P]`** (120 Hz Optotrak GT `Elbow[:,1]` in deg) | `[X]` (Healthy adults only) | **`[P]`** (Subj-indep + Cross-dataset w/ `JU-IMU` & `PHYTMO`) | **Track 1A, Track 3A, Track 5A** |
| **3** | **`JU-IMU`** | Priority 8 | 41 subj (`27 ND`, `14 Stroke`; `574` trials) | **Yes** (`ND4..33`, `Stroke2..16`) | **Yes** (`sensor4` RUA + `sensor1` RLA; `sensor5` LUA + `sensor2` LLA) | **`[P]`** (14 upper-limb ROM/ADL exercises) | `[X]` (No form error labels; cohort labels only) | `[X]` (No ground-truth joint angle column) | `[X]` (Has `SupPro` class, but no GT angle col) | **`[P]`** (`27 ND` vs `14 Stroke` across 14 ROM tasks) | **`[P]`** (Subj-indep, Cross-cohort, Cross-dataset w/ `Optotrak`) | **Track 1B, Track 4A, Track 5A** |
| **4** | **`StrokeRehab`** | Priority 8 | 20 healthy subj locally (`C4..C32`; `1,426` trials) | **Yes** (`C4..C32`) | **Yes** (Noraxon `LT/RTUpperarm` + `LT/RTForearm`) | **`[S]`** (9 ADL tasks + frame-level primitives) | `[X]` (Healthy controls only) | **`[S]`** (100 Hz GT `ElbowFlexionLT/RTdeg`) | **`[S]`** (100 Hz GT `WristSupinationLT/RTdeg`) | `[X]` (Stroke subfolder absent on disk; only `healthy controls` present) | **`[S]`** (Subj-indep across 20 subjects & 9 ADLs) | **Track 3B (Separate Noraxon Acc+Quat/Euler Track)** |
| **5** | **`ULF_in_ADL`** | Priority 8 | 25 subj (`20 Stroke`, `5 Healthy`; `4,492` `.mvnx`) | **Yes** (`P02..P30`, `H01..H05`) | **Yes** (Xsens MVN `Right/LeftUpperArm` + `ForeArm`) | **`[S]`** (30 ADL tasks `T01..T30`) | `[X]` (No binary form labels) | **`[P]`** (60 Hz GT `jRight/LeftElbow` flexion in deg) | **`[P]`** (60 Hz GT `jRight/LeftElbow` pron/sup in deg) | **`[P]`** (`20 Stroke` [`FMMA-UE` 21–61, paretic vs non-paretic] vs `5 Healthy`) | **`[P]`** (Cross-subject & Healthy-to-Stroke angle generalization) | **Track 3D, Track 4B** |
| **6** | **`ULTra-MoCap`** | Priority 8 | 13 subj (`P01..P13`; `217` IMU, `208` IK trials) | **Yes** (`P01..P13`) | **Yes (Distal `1..3` + Proximal `4..6` 6-DOF IMUs)** | **`[S]`** (5 upper-limb exercises) | `[X]` (Variations are speed & target ROM, not errors) | **`[S]`** (100 Hz OpenSim IK `elbow_flex_r/l` in deg) | **`[S]`** (100 Hz OpenSim IK `pro_sup_r/l` in deg) | `[X]` (Healthy subjects only) | **`[P]`** (Generalization across 4 speeds `Slow..VeryFast` & 4 ROMs `45°..Max`) | **Track 3C, Track 5B** |
| **7** | **`VIDIMU`** | Priority 8 | 53 subj (`16` IMU+Video `S40..S57`; `37` Video-only) | **Yes** (`S40..S57`) | **Quaternions Only** (`humerus_r/l_imu` + `radius_r/l_imu` in `A05..A13`) | **`[S]`** (9 upper-limb tasks `A05..A13` via Quats) | `[X]` (Healthy standard execution only) | **`[S]`** (50 Hz OpenSim IK `elbow_flex_r/l` from Quats) | **`[S]`** (50 Hz OpenSim IK `pro_sup_r/l` from Quats) | `[X]` (Healthy adults only) | **`[S]`** (Subj-indep across 16 IMU subjects) | **Track 3E (Separate 8-Ch Dual-IMU Quaternion Track)** |
| **8** | **`REHAB` (REHAB-120)** | Priority 8 | 222 sessions (`120` stroke patients in lit.) + `4,616` exercise trials | **Partial / Session-Only** (No local patient-to-session map) | **Yes** (Stream `_1` & `_2` 9-DOF IMUs on paretic arm) | **`[S]`** (27 FMA-UE items / 16 rehab exercises) | `[X]` (Clinical FMA-UE `0,1,2` score, not form error) | `[X]` (No optical GT angle; only onboard Euler) | `[X]` (No optical GT pron/sup angle) | **`[S]`** (3-class FMA-UE score `0, 1, 2` across 16 proximal arm tasks) | `[X]` (Missing patient-to-session ID mapping prevents strict subject split) | **Track 4C (Session-Split Clinical FMA-UE Track w/ Leakage Caveat)** |
| **9** | `EMG elbow dataset` | Auxiliary 5 | 10 subj (`subject 1..10`; `120` files) | Yes (`1..10`) | **No** (2-ch sEMG + 1 scalar angle col; **0 raw IMU cols**) | `[X]` (No IMU channels) | `[X]` (No quality labels) | `[X]` (No raw IMU input to predict angle) | `[X]` (No raw IMU input) | `[X]` (Healthy adults) | `[X]` (sEMG modality only) | **Excluded** (No raw accelerometer/gyroscope signals) |
| **10** | `mm-fit` | Auxiliary 5 | 21 workouts (`w00..w20`; 10 subj in lit.) | **No** (No local workout-to-subject map) | **No** (Wrists, earbud, pocket phones; **no upper-arm IMU**) | `[X]` (Gym fitness + no upper-arm IMU + unmapped subjects) | `[X]` (Repetition counts only) | `[X]` (No upper-arm IMU) | `[X]` (No pron/sup exercise) | `[X]` (Healthy gym workouts) | `[X]` (Unmapped subject workouts + 5.25M duplicate timestamps) | **Excluded** (Missing upper-arm IMU & missing subject ID mapping) |
| **11** | `MoCap and IMU Dataset...` | Auxiliary 5 | 6 takes (`Take1..6`; subj count unknown locally) | **No** (Only 6 takes, all `Skeleton 002`; unknown subject IDs) | **No** (Chest + R/L wrists; **only 1 IMU per arm**) | `[X]` (No discrete exercise labels) | `[X]` (No quality labels) | `[X]` (Lacks upper-arm IMU & lacks subject IDs) | `[X]` (Lacks upper-arm IMU & subject IDs) | `[X]` (Healthy takes) | `[X]` (Cannot verify subject-independent generalization) | **Excluded** (Single wrist IMU per arm + unknown subject IDs across 6 takes) |
| **12** | `Smartwatch IMU Dataset...` | Auxiliary 5 | 3 subj (`u01, u04, u05`) | **Too Few Subj ($N=3$)** | **No** (Single Apple Watch on wrist; **no upper-arm IMU**) | `[X]` ($N=3$ subjects & wrist-only) | `[X]` (No execution quality labels) | `[X]` (No ground-truth joint angles) | `[X]` (No ground-truth joint angles) | `[X]` (Only 1 weak-arm subject `u05`) | `[X]` ($N=3$ subjects insufficient for deep benchmark) | **Excluded** (Wrist-only sensor & only $N=3$ subjects) |
| **13** | `UCI Biceps Curl` | Auxiliary 5 | 5 subj (`4,024` rows total) | **Impossible** (Subject & class confounded in excerpt) | **Yes** (`arm` + `forearm` 9-DOF) | `[X]` (Single exercise: biceps curl) | `[X]` (**Confounded:** 3 of 5 subjects only have Class `E`) | `[X]` (No ground-truth optical elbow angle) | `[X]` (No pronation/supination task) | `[X]` (Healthy young adults) | `[X]` (4,024-row excerpt prevents subject-independent split) | **Excluded** (Local file is a 4,024-row excerpt with fatal subject-class confounding) |

---

## 3. Detailed Evaluation of the 8 Priority Candidate Datasets

### 3.1 `PHYTMO` (`D:\me\project\datasets\raw\PHYTMO`)
- **Subjects & Sessions:** **30 subjects** across 5 age cohorts (`A01–A10`, `B01–B05`, `C01–C05`, `D01–D05`, `E01–E05`). In `inertial\upper\`, there are **342 complete 4-sensor exercise trials** (`1,368` CSVs) plus **29 static calibration trials** (`116` CSVs; `A02Calib` absent).
  - *Verified Per-Subject Class Coverage:* 26 of 30 subjects have all 6 upper-limb exercise-quality conditions (`EFE0, EFE1, EAH0, EAH1, SQZ0, SQZ1`). `A02` lacks `EAH0/EAH1`; `A06`, `A07`, and `B03` lack `EFE0/EFE1`.
- **Subject-Independent Split Feasibility:** **Yes (High confidence).** Clean subject IDs (`A01`–`E05`) allow stratified subject-independent `train` (20 subjects) / `val` (5 subjects) / `test` (5 subjects) splits where all `val` and `test` subjects are drawn from the 26 complete subjects.
- **Sensors, Placement, Orientation & Signals:**
  - 4 × 9-axis NGIMUs in `inertial\upper\`: `Rarm` (Right Upper Arm), `Rforearm` (Right Forearm), `Larm` (Left Upper Arm), `Lforearm` (Left Forearm).
  - Verified static gravity vector on `Rarm` points along `+X` (`[1.013, 0.005, 0.013] g`).
  - Signals: 3D Gyroscope (`deg/s`), 3D Accelerometer (`g`), 3D Magnetometer (`uT`).
- **Sampling Rate:** Calculated median **100.89 Hz** (nominal **100.0 Hz**).
- **Available Targets:**
  - Exercise class: `EFE` (Elbow Flexion-Extension, 108 trials), `EAH` (Extension of Arms Overhead, 116 trials), `SQZ` (Squeezing, 118 trials).
  - Execution quality label: **`0` = Correct execution** (`171` trials) vs. **`1` = Incorrect/compensatory execution** (`171` trials).
- **Data Quality Issues & Handling:**
  - `0` NaNs across all `1,484` upper-limb inertial CSVs.
  - `98` of `1,484` CSVs have duplicate consecutive timestamps (`252` duplicate samples total) and slight rate drift (`98.7–101.5 Hz`) $\rightarrow$ handled by timestamp deduplication (`keep='first'`), common time-range intersection across the sensors of each trial, and linear interpolation to an exact uniform **100.0 Hz** grid.
  - Optical CSVs (`optical interp\rigid_bodies\upper\`) have missing cells (`,,`) in `33%` of files (`55%` in `optical raw`) and template marker names (`shin`/`thigh`), making optical joint-angle derivation unreliable.
- **Inclusion Decision:** **INCLUDED (`PRIMARY`)** in **Track 1C** (Exercise Classification), **Track 2A/2B** (Correct vs. Incorrect Execution Quality Classification), and **Track 5A** (Cross-Dataset Generalization).

---

### 3.2 `Optotrak vs DOT` (`D:\me\project\datasets\raw\Optotrak vs DOT`)
- **Subjects & Sessions:** **16 subjects** (`subject 1`–`subject 16`), each performing **20 trials** (`Trials 1–10` calibration; `Trials 11–20` functional/ROM evaluation tasks, 5 reps/trial = `320` total trials).
  - *Verified Subject Details:*
    - `subject 2`: Sensors were mounted on the **left arm** due to a right-side injury (documented in `Readme.docx`). Excluded from right-arm benchmarks to preserve coordinate-frame consistency.
    - `subject 8`: Has complete, NaN-free `B_dotComplete` IMU data for all 20 trials (valid for Track 1A Exercise Classification!), but lacks `H_Joint_Angles_Aligned_FINAL` because the medial epicondyle (`ME`) landmark was not digitized (excluded from Track 3A Angle Regression).
    - All remaining **14 right-arm subjects** (`subject 1, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16`) have **100% sample-aligned, zero-NaN `B_dotComplete` (`RUA` + `RLA`) and `H_Joint_Angles_Aligned_FINAL` (`opto` 3D `Elbow` angles) across all 10 evaluation tasks (`Tasks 11–20`)**!
- **Subject-Independent Split Feasibility:** **Yes (High confidence).**
  - Track 1A (15 right-arm subjects): `9 train / 3 val / 3 test` subjects.
  - Track 3A (14 right-arm subjects with Optotrak GT): `8 train / 3 val / 3 test` subjects.
- **Sensors, Placement, Orientation & Signals:**
  - 5 × Xsens DOT IMUs (`TRK`, `RSC`, `RUA` [Right Upper Arm], `RLA` [Right Forearm], `RHD` [Right Hand]) + 5 Optotrak Certus 3-marker optical clusters.
  - Verified static gravity vector on `RUA` points along `+X` (`[9.45, -0.96, 2.33] m/s²`).
  - Signals: Quaternion (`W,X,Y,Z`), 3D Accelerometer (`m/s²`), 3D Gyroscope (`deg/s`), 3D Magnetometer (normalized), and `ZYX` Euler angles (`deg` in `B_dotComplete`).
- **Sampling Rate:** Verified **120.0048 Hz** ($\Delta t = 8333\ \mu\text{s}$; exact 1-to-1 frame lock with 120 Hz Optotrak in `B_dotComplete` and `H_Joint_Angles_Aligned_FINAL`).
- **Available Targets:**
  - **Task A (10 Evaluation Tasks `11–20`):** `11` Elbow Flexion/Extension, `12` Forearm Pronation/Supination, `13` Shoulder Flexion, `14` Shoulder Abduction, `15` Shoulder Int/Ext Rotation, `16` Wrist Flexion/Extension, `17` Wrist Radial/Ulnar Deviation, `18` Drinking from Cup, `19` Box off Shelf, `20` Drawing Circles.
  - **Task C & Task D (Continuous Ground-Truth Angles in `deg`):** From `H_Joint_Angles_Aligned_FINAL['j_aligned_eu']['opto'][trial]['Elbow']`:
    - `Elbow[:, 2]`: **Continuous Elbow Flexion-Extension Angle (`deg`)** (verified range `1.4°` to `136.2°` in Task 11).
    - `Elbow[:, 1]`: **Continuous Forearm Pronation-Supination Angle (`deg`)** (verified range `-62.0°` to `+69.2°` in Task 12).
- **Data Quality Issues & Handling:**
  - `69` backup CSV files in `delete` and `Fist attempt before deleting RLA` subfolders $\rightarrow$ strictly excluded (or use `B_dotComplete` which already curates the active trials and removes row 0).
  - Row 0 (`PacketCounter == 0`) zero artifact in raw DOT CSVs (`Acc/Gyr == 0`) $\rightarrow$ already removed in `B_dotComplete` (`trk(1,:) = []`) and explicitly stripped if reading raw CSVs.
  - Calibration `trial 5` of `subject 3` has `1,761` NaNs in `opto['Elbow']` $\rightarrow$ excluded from angle regression if calibration trials are used; all 150 evaluation trials (`Tasks 11–20` across 15 subjects) have `0` NaNs.
- **Inclusion Decision:** **INCLUDED (`PRIMARY`)** in **Track 1A** (10-Class Exercise Classification), **Track 3A** (Continuous Elbow Flexion-Extension & Forearm Pronation-Supination Angle & ROM Regression), and **Track 5A** (Cross-Dataset Generalization).

---

### 3.3 `JU-IMU` (`D:\me\project\datasets\raw\JU-IMU\ROM`)
- **Subjects & Sessions:** **41 subjects** (`27` Non-Disabled `ND4–ND33` and `14` Hemiparetic Stroke patients `Stroke2–Stroke16`), each performing **14 upper-limb ROM/ADL activities** (`574` CSV files total).
- **Subject-Independent Split Feasibility:** **Yes (High confidence).** Stratified by cohort (`ND` vs. `Stroke`): `29 train (19 ND, 10 Stroke) / 6 val (4 ND, 2 Stroke) / 6 test (4 ND, 2 Stroke)` subjects.
- **Sensors, Placement & Signals:**
  - 5 × 9-axis IMUs (`sensor1`: Right Wrist/Forearm, `sensor2`: Left Wrist/Forearm, `sensor3`: Trunk/Chest, `sensor4`: Right Upper Arm, `sensor5`: Left Upper Arm).
  - Signals: 3D Accelerometer (`m/s²`), 3D Gyroscope (`rad/s`), 3D Magnetometer (`uT`).
- **Sampling Rate:** Documented **80.0 Hz** (no timestamp column in CSVs).
- **Available Targets:**
  - **14 Exercise Classes:** `ElbFlex` (Elbow Flexion/Extension), `SupPro` (Forearm Supination/Pronation), `FrontSupPro`, `WrstFlex`, `WrstExt`, `FrontWrstFlex`, `FrontWrstExt`, `ShFlex90`, `ShExt`, `ShAbd90`, `Scaption90`, `ShHorAdd`, `ShRot`, `PickCup`.
  - **Cohort / Impairment Target:** Binary `ND` (`0`) vs. `Stroke` (`1`).
- **Data Quality:** `0` NaNs and `0` duplicate rows across all `574` CSVs.
- **Inclusion Decision:** **INCLUDED (`PRIMARY`)** in **Track 1B** (14-Class Exercise Classification), **Track 4A** (Stroke vs. Non-Disabled Impaired Movement Analysis), and **Track 5A** (Cross-Dataset Generalization with `Optotrak vs DOT` & `PHYTMO`).

---

### 3.4 `StrokeRehab` (`D:\me\project\datasets\raw\StrokeRehab`)
- **Subjects & Sessions:** **20 healthy control subjects** (`C4–C32`) performing **9 ADL/rehab activities** in left-focus (`LT`, `716` CSVs) and right-focus (`RT`, `710` CSVs) folders = **`1,426` CSV files**. *(Note: Only the `healthy controls` branch is present on disk; no stroke patient folder exists locally).*
- **Subject-Independent Split Feasibility:** **Yes** (`14 train / 3 val / 3 test` subjects across the 20 subjects).
- **Sensors, Placement & Signals (Verified Anomaly):**
  - 9 × Noraxon MyoMotion IMUs (`Upperspine`, `Lowerspine`, `Pelvis`, `LT/RTUpperarm`, `LT/RTForearm`, `LT/RTHand`).
  - **Critical Signal Verification:** The CSVs provide linear accelerometer (`UpperarmAccelSensorX/Y/Z_LT/RT_mG`, `ForearmAccelSensorX/Y/Z_LT/RT_mG` in `mG`), segment orientation Euler angles (`course, pitch, roll` in `deg` in 1,418 Schema-A files), and `RotX, RotY, RotZ` (verified to have vector norm $\le 1.0$, representing unit quaternion vector components rather than raw gyroscope angular velocity!). However, continuous 3D angular velocity $\boldsymbol{\omega} = (\omega_x, \omega_y, \omega_z)$ in $\text{rad/s}$ can be deterministically differentiated from the 100 Hz segment orientations (`course, pitch, roll` / quaternion `(qw, qx, qy, qz)`).
- **Sampling Rate:** Verified exact **100.00 Hz** ($\Delta t = 0.01\text{ s}$).
- **Available Targets:**
  - Continuous `ElbowFlexionLTdeg`, `ElbowFlexionRTdeg` and `WristSupinationLTdeg`, `WristSupinationRTdeg` (`deg`).
  - 9 trial-level ADL classes + frame-level functional movement primitive labels (`MarkerNames`).
- **Data Quality Issues & Handling:**
  - `2` files (`LT\C5\drinking\C5_RT_SB_mod_drinking 5.csv` and `RT\C5\drinking\C5_SB_mod_drinking 5.csv`) have massive NaNs in `RotX..Z` $\rightarrow$ excluded.
  - `8` files (`3` in `C12` Schema B, `5` in `C26` Schema C) omit segment `course/pitch/roll` columns (and `C26` omits `UpperArmAccelSensorX/Y/Z_LT_mG`) $\rightarrow$ excluded so all retained `1,416` Schema-A files share an identical 106-column schema.
  - Folder spelling/case splits (`deodrant` vs `deodorant`, `shelf` vs `Shelf`) $\rightarrow$ normalized to canonical lowercase labels.
- **Inclusion Decision:** **INCLUDED (`SECONDARY / SEPARATE TRACK`)** in **Track 3B** (Continuous Elbow Flexion & Forearm Supination Angle Regression during ADLs). Not mixed with raw-gyroscope datasets because angular velocity is derived from Noraxon orientation filters rather than raw MEMS gyro output.

---

### 3.5 `ULF_in_ADL` (`D:\me\project\datasets\raw\ULF_in_ADL`)
- **Subjects & Sessions:** **25 subjects** (`20` post-stroke patients `P02–P30` with clinical `FMMA-UE` scores `21–61` and affected-side annotations in `ParticipantCharacteristics.xlsx`, plus `5` healthy controls `H01–H05`), performing **30 ADL tasks (`T01–T30`)** across left and right arms (`4,492` `.mvnx` XML files, `24.66 GB`).
- **Subject-Independent Split Feasibility:** **Yes** (`17 train [14 Stroke, 3 Healthy] / 4 val [3 Stroke, 1 Healthy] / 4 test [3 Stroke, 1 Healthy]`).
- **Sensors, Placement & Signals:**
  - 17 Xsens MVN IMUs including bilateral **`RightUpperArm` + `RightForeArm`** and **`LeftUpperArm` + `LeftForeArm`**.
  - Each `.mvnx` normal frame provides segment `acceleration` ($\text{m/s}^2$) and `angularVelocity` ($\text{rad/s}$), sensor `sensorFreeAcceleration` ($\text{m/s}^2$) and `sensorOrientation` (quaternions), and 3D anatomical joint angles `jointAngle` (`ZXY` and `XZY` in $\text{deg}$) for `jRightElbow` and `jLeftElbow` (continuous elbow flexion/extension and pronation/supination).
- **Sampling Rate:** Verified **60.00 Hz** (`frameRate="60"`).
- **Data Quality & Scale:** `0` NaNs in sampled `.mvnx` files; `~2%` of files omit the optional `<sensors>` block while retaining full `<segments>` (`acceleration`, `angularVelocity`) and `<joints>` (`jointAngle`) streams. Because parsing all `4,492` XML files (`24.66 GB`) takes significant disk I/O, a curated subset of canonical reaching/drinking/feeding tasks (e.g., `T01–T05`) or lazy/cached extraction is used.
- **Inclusion Decision:** **INCLUDED (`PRIMARY` for Task E & Task C/D in Stroke)** in **Track 4B** (Paretic vs. Non-Paretic vs. Healthy Limb Classification & Clinical Impairment Analysis) and **Track 3D** (Continuous Elbow Angle Estimation in Stroke ADLs).

---

### 3.6 `ULTra-MoCap` (`D:\me\project\datasets\raw\ULTra-MoCap`)
- **Subjects & Sessions:** **13 healthy subjects** (`P01–P13`), performing **5 upper-limb exercises** (`Elbow Flexion`, `Shoulder Rotation`, `Crossbody Reach`, `Arm Swing`, `Overhead Reach`) across multiple speeds (`Slow`, `Normal`, `Fast`, `VeryFast`) and target ROM elevations (`45°`, `90°`, `180°`, `Max`): `217` IMU CSVs, `217` EMG CSVs, `208` OpenSim IK CSVs/MOTs.
- **Subject-Independent Split Feasibility:** **Yes** (`9 train / 2 val / 2 test` subjects).
- **Sensors, Placement & Signals:**
  - 6 × Delsys Trigno 6-DOF IMUs (`ACCX1..GYROZ6` at **2,000 Hz**; Acc in $\text{mG}$, Gyro in $\text{deg/s}$) + 3 sEMGs (`IM EMG4..6`) + 100 Hz Vicon 60-marker OpenSim Inverse Kinematics (`elbow_flex_r/l` and `pro_sup_r/l` in $\text{deg}$).
  - *Verified Kinematic Placement:* Empirical gyro variance during pure `ElbowFlexion` confirms `IMU 1, 2, 3` are mounted on the moving distal forearm/hand segment ($\sigma_{\text{gyr}} \approx 64\text{–}70\text{ deg/s}$), whereas `IMU 4, 5, 6` (paired with `EMG 4, 5, 6`) are mounted on the stationary proximal upper-arm/shoulder muscles ($\sigma_{\text{gyr}} \approx 6\text{–}21\text{ deg/s}$). Because the CSV headers label sensors `1..6` without explicit anatomical strings, `ULTra-MoCap` is placed in its own dedicated track rather than merged with `Optotrak vs DOT` or `PHYTMO`.
- **Data Quality Issues & Handling:**
  - `9` trials (`P04–P12` `overheadreach_45`) exist in `IMU`/`EMG` but lack OpenSim IK (`Processed IK`/`Raw IK`) $\rightarrow$ excluded from Track 3C angle regression (retained for IMU-only classification if needed).
  - `8` trials (`P05–P12` `overheadreach_90`) have case differences (`overheadreach_90` in IMU vs `OverheadReach_90` in IK) $\rightarrow$ matched case-insensitively.
  - 2,000 Hz IMU signals must be low-pass anti-alias filtered (4th-order Butterworth, cutoff $40\text{ Hz}$) before decimation by $20\times$ to match the **100.0 Hz** OpenSim IK ground-truth frames (`Sub Frame == 0`).
- **Inclusion Decision:** **INCLUDED (`SECONDARY / SEPARATE TRACK`)** in **Track 3C** (Multi-Speed / Multi-ROM Continuous Elbow Angle Regression) and **Track 5B** (Cross-Speed & Cross-ROM Condition Generalization).

---

### 3.7 `VIDIMU` (`D:\me\project\datasets\raw\VIDIMU\dataset`)
- **Subjects & Sessions:** **53 subjects total**: `16` subjects (`S40–S57`) in `videoandimus` with 5 IMUs + OpenSim IK + Video across 13 activities (`208` trials total; **`144` upper-limb trials in activities `A05–A13`**), plus `37` subjects (`S01–S39`) in `videonly` with video only.
- **Subject-Independent Split Feasibility:** **Yes** (`10 train / 3 val / 3 test` subjects among the 16 IMU subjects `S40–S57`).
- **Sensors, Placement & Signals:**
  - In upper-limb activities `A05–A13`: 5 IMUs explicitly named in OpenSim `.sto` headers as `torso_imu`, **`humerus_r_imu` (Right Upper Arm)**, **`radius_r_imu` (Right Forearm)**, **`humerus_l_imu` (Left Upper Arm)**, and **`radius_l_imu` (Left Forearm)** at **50.0 Hz** (`.sto`) and **333.33 Hz** (`.raw`).
  - **Critical Signal Limitation:** `.sto` and `.raw` store **only 4D unit quaternions (`w, x, y, z`)**; raw linear accelerometer (`Acc_X/Y/Z`) is not present (though 3D body-frame angular velocity $\boldsymbol{\omega} = 2 \dot{q} \otimes q^*$ can be computed from the quaternions).
  - Ground-truth OpenSim IK `.mot` files at **50.0 Hz** provide `elbow_flex_r`, `pro_sup_r`, `elbow_flex_l`, `pro_sup_l` in **degrees**.
- **Data Quality Issues & Handling:**
  - **Verified Cross-Subject Contamination in `S48`:** `videoandimus\S48` contains 13 unprefixed files (`S48_A01..A13.mot` at 60 Hz) whose internal header reads `name=ik_S53_A01..A13` (copied from Subject `S53`). These 13 files are **strictly excluded**; only the valid `ik_S48_A01..A13.mot` (50 Hz) files are used.
  - Lower-limb activities `A01–A04` use lower-limb IMU placements (`pelvis, femur, tibia`) and are **strictly excluded**.
- **Inclusion Decision:** **INCLUDED (`SECONDARY / SEPARATE TRACK`)** in **Track 3E** (Quaternion-to-Elbow/Forearm Joint Angle Regression & 9-Class Upper-Limb Activity Classification on `A05–A13`).

---

### 3.8 `REHAB` (`REHAB-120`, `D:\me\project\datasets\raw\REHAB`)
- **Subjects & Sessions:** `222` clinical assessment sessions (`Rehab_assessment\001..222`, each with 27 FMA-UE tasks) and `4,616` training trials across 16 rehab exercises (`Rehab_exercise\000..015`).
- **Subject-Independent Split Feasibility:** **No (Session-Level Split Only).** While literature states the 222 sessions come from 120 post-stroke patients, **no local mapping file links longitudinal sessions (`001..222`) to unique patient IDs**, and `Rehab_exercise` pools trials without patient IDs.
- **Sensors, Placement & Signals:**
  - In `Rehab_assessment` proximal arm tasks (`000–014` and `026`, 16 tasks): `XXX_1.npy` `(T, 12)` and `XXX_2.npy` `(T, 13)` contain dual upper-limb 9-DOF IMU streams (3D Euler $\text{deg}$, 3D Acc $g$, 3D Mag raw, 3D Gyro $\text{deg/s}$, plus 1 joint angle in `_2.npy`), while `XXX_3.npy` (glove) is zero. In distal hand tasks (`015–025`), `_1` and `_2` are all zeros.
- **Data Quality Issues & Handling:**
  - **Corrupted File:** `Rehab_exercise\d02_processed_data\014_1.npy` is corrupted by UTF-8 replacement bytes (`\xef\xbf\xbd`) $\rightarrow$ **strictly excluded** (use `d01_raw_data\014_1.npy` if needed).
  - **Duplicate Subfolder:** `Rehab_assessment\d02_processed_data\195\2021-03-03 11-00-11\` (`81` extra files) $\rightarrow$ **strictly excluded**.
  - **Mismatched Raw Sensor Lengths:** In `Rehab_assessment\d01_raw_data`, `_1.npy` and `_2.npy` often differ in sample count $T_1 \neq T_2$; `d02_processed_data` resamples both to fixed $T = 150$.
- **Inclusion Decision:** **INCLUDED (`SECONDARY / CAVEATED TRACK`)** in **Track 4C** (Clinical FMA-UE Item Score `0, 1, 2` Classification on Proximal Upper-Limb Tasks `000–014, 026`), explicitly reporting that splits are session-independent rather than guaranteed patient-independent due to missing patient-session mapping metadata.

---

## 4. Justification for Excluding the Remaining 5 Datasets from Core Benchmarks

1. **`EMG elbow dataset`**: Excluded because the measurement files contain **zero raw IMU channels** (only 4 sEMG columns in $\text{mV}$ + 1 scalar joint angle column in $\text{deg}$), and splits are intra-subject.
2. **`mm-fit`**: Excluded because (a) **no upper-arm IMU exists** (sensors are wrist smartwatches, earbud, and trouser-pocket smartphones), (b) no local mapping links the 21 workout folders (`w00–w20`) to the 10 participants, and (c) sensor arrays contain `5,256,856` batch-duplicated millisecond timestamps.
3. **`MoCap and IMU Dataset for Upper Limb Joint Angle Estimation`**: Excluded from the core dual-IMU benchmark because (a) **only 1 IMU is placed on each arm** (wrist/forearm `Rarm`, `Larm` + `chest`; no upper-arm IMU exists), and (b) only 6 continuous recording takes (`Take1–Take6`, all using `Skeleton 002`) exist without subject ID metadata, making subject-independent splitting impossible to verify.
4. **`Smartwatch IMU Dataset for Upper-Limb Movement Ana`**: Excluded because (a) **only a single wrist-worn Apple Watch is used** (no upper-arm IMU), (b) only **$N = 3$ subjects** (`u01, u04, u05`) exist, (c) `all_users_valid.csv` contains embedded duplicate header rows, and (d) raw `GyroX/Y/Z` is completely missing (`NaN`) for subject `u01`.
5. **`UCI Biceps Curl`**: Excluded because the local CSV is only the **4,024-row example subset** of Velloso et al. (2013), in which **subject identity and execution class (`A–E`) are completely confounded** (`adelmo`, `eurico`, and `jeremy` only have Class `E`; `pedro` only has `A, B, C`; `carlitos` only has `A, D, E`), making a leak-free subject-independent split across the 5 classes mathematically impossible.
