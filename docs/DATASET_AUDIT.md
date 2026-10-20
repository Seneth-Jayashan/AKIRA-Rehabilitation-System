# Initial Dataset Audit (`docs/DATASET_AUDIT.md`)

**Audit Date:** 2026-10-19  
**Project Root:** `d:\me\project`  
**Raw Datasets Location:** `d:\me\project\datasets\raw\`  
**Inspection Mode:** Strictly read-only (0 original dataset files modified, moved, or deleted)

---

## 1. Executive Summary & Workspace Overview

An exhaustive read-only inspection of `d:\me\project` identified **13 raw public datasets** stored under `d:\me\project\datasets\raw\`, totaling **`68.23 GB` (`73,286,447,650` bytes)** across **`55,792` files**.

 throughout this audit, **File-Verified Facts** (measured directly from local file headers, array shapes, timestamps, and values) are explicitly distinguished from **Literature-Verified Context / Missing Local Documentation** (information documented in the original dataset publications that is not stored as a local README file).

### High-Level Summary Matrix Across All 13 Datasets

| # | Dataset Folder Name | Size | Files | Formats | Upper Arm + Forearm Dual-IMU? | Sampling Rate (Verified) | Subjects | Elbow Flex/Ext? | Forearm Pron/Sup? | Continuous Joint Angles / ROM? | Exercise / Activity Classes? | Correct vs. Incorrect Labels? |
| :- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | `EMG elbow dataset` | 304.30 MB | 130 | `.txt` | **No** (2-ch sEMG + 1 scalar angle col; no raw IMU cols) | Unknown locally (~1024 Hz lit.) | 10 (`subject 1..10`) | **Yes** (`flex`) | **Yes** (`pronsup`) | **Yes** (Col 4 angle in deg) | **Yes** (2 movements × 3 loads) | **No** |
| **2** | `JU-IMU` | 383.21 MB | 574 | `.csv` | **Yes** (5 × 9-axis IMUs: R/L wrist, R/L upper arm, trunk) | ~80 Hz (lit.; no timestamp col) | 41 (`27 ND`, `14 Stroke`) | **Yes** (`ElbFlex`) | **Yes** (`SupPro`, `FrontSupPro`) | **No** (Raw IMU only; no GT angle col) | **Yes** (14 ROM/rehab tasks) | **Cohort only** (`ND` vs `Stroke`) |
| **3** | `mm-fit` | 3.53 GB | 302 | `.npy`, `.csv` | **No** (L/R wrist smartwatches, earbud, pocket phones) | ~100 Hz watch, 30 Hz 3D pose | 21 workouts (`w00..w20`, 10 subj lit.) | **Yes** (`bicep_curls`, `tricep_extensions`) | **No** | **Indirect** (3D skeleton `pose_3d.npy`) | **Yes** (10 gym exercises + rep counts) | **No** |
| **4** | `MoCap and IMU Dataset for Upper Limb Joint Angle Estimation` | 1.45 GB | 378 | `.csv` | **IMUs: No** (Chest + R/L wrists); **MoCap: Yes** | **50.0 Hz** (100 Hz raw Quat/Eul decimated to 50 Hz) | 6 takes (`Take1..6`; subj count undocumented locally) | **Yes** (`L/R_elbow_z`) | **Yes** (`L/R_elbow_y`) | **Yes** (10 continuous GT angles in `df_angles.csv`) | **No** (Continuous regression only) | **No** |
| **5** | `Optotrak vs DOT` | 5.43 GB | 7,711 | `.csv`, `.c3d`, `.xls`, `.mat` | **Yes** (5 Xsens DOTs: Trunk, Scapula, R Upper Arm `RUA`, R Forearm `RLA`, R Hand) | **120.00 Hz** ($\Delta t = 8333\ \mu\text{s}$) | 16 (`subject 1..16`; 13 with full `I_tasks_split`) | **Yes** (Trial 2 & Task 11) | **Yes** (Trial 3 & Task 12) | **Yes** (3D Optotrak & IMU `Elbow` angles in deg) | **Yes** (10 calibration + 10 functional/ROM tasks) | **No** |
| **6** | `PHYTMO` | 11.68 GB | 7,076 | `.csv`, `.m` | **Yes (Bilateral)** (`Larm`, `Lforearm`, `Rarm`, `Rforearm` 9-axis NGIMUs) | **100.89 Hz** median (100.0 Hz optical) | 30 (`A01..E05`) | **Yes** (`EFE`: Elbow Flexion-Extension) | **No** | **Indirect** (OptiTrack 3D rigid body quats/markers) | **Yes** (3 upper-limb + 6 lower-limb exercises) | **YES** (`0` = Correct vs `1` = Incorrect execution) |
| **7** | `REHAB` (REHAB-120) | 1.07 GB | 36,110 | `.npy` | **Yes** (Dual upper-limb 9-axis IMUs + data glove) | ~50 Hz (lit.; no timestamp col) | 222 sessions (`001..222` from 120 stroke patients lit.) | **Yes** (Included in 27 FMA-UE & 16 rehab exercises) | **Yes** (Included in 27 FMA-UE items) | **Partial** (Onboard IMU Euler angles in deg) | **Yes** (27 assessment tasks + 16 training exercises) | **Clinical Score** (`Label.npy`: FMA-UE `0, 1, 2`) |
| **8** | `Smartwatch IMU Dataset for Upper-Limb Movement Ana` | 576.41 MB | 11 | `.csv`, `.md`, `.json` | **No** (Wrist Apple Watch Series 6 only) | **99.35 Hz** (96.4% rows) & **20.0 Hz** (3.6% rows) | 3 (`u01`, `u04`, `u05`) | **Yes** (`el-exfl`) | **Yes** (`wr-prsu`) | **No** (Watch orientation only; no GT elbow angle) | **Yes** (`el-exfl`, `sh-exfl`, `wr-prsu`) | **Cohort only** (`Target=1` weak arm `u05` vs `0`) |
| **9** | `StrokeRehab` | 5.80 GB | 1,426 | `.csv` | **Yes (Bilateral)** (9 Noraxon IMUs incl. L/R Upper Arm & Forearm) | **100.00 Hz** ($\Delta t = 0.01\text{ s}$) | 20 healthy controls (`C4..C32`) | **Yes** (`ElbowFlexionLT/RTdeg`) | **Yes** (`WristSupinationLT/RTdeg`) | **Yes** (22 joint angles + 27 segment Euler angles in deg) | **Yes** (9 ADL tasks + frame-level primitives) | **No** (Only healthy controls branch exists locally) |
| **10** | `UCI Biceps Curl` | 1.59 MB | 1 | `.csv` | **Yes (Unilateral)** (4 × 9-DOF IMUs: `belt`, `arm`, `forearm`, `dumbbell`) | **49.55 Hz** median (45 Hz nominal) | 5 (`pedro`, `carlitos`, `adelmo`, `eurico`, `jeremy`) | **Yes** (Biceps curl) | **No** | **Partial** (Sensor Euler angles; no MoCap GT) | **Single exercise** (Biceps curl variations) | **YES** (`A` = Correct vs `B, C, D, E` = 4 mistakes) |
| **11** | `ULF_in_ADL` | 24.66 GB | 4,493 | `.mvnx`, `.xlsx` | **Yes (Bilateral)** (17 Xsens MVN IMUs incl. L/R UpperArm & ForeArm) | **60.00 Hz** (`frameRate="60"`) | 25 (`20 Stroke P02..P30`, `5 Healthy H01..H05`) | **Yes** (`jRight/LeftElbow` 3D angles) | **Yes** (`jRight/LeftElbow` 3D angles) | **Yes** (22 3D joints in deg + 17 IMU streams) | **Yes** (30 ADL tasks `T01..T30`) | **Clinical** (Paretic vs non-paretic + `FMMA-UE` 21–61) |
| **12** | `ULTra-MoCap` | 12.29 GB | 1,102 | `.csv`, `.trc`, `.mot`, `.osim`, `.h5` | **Yes** (6 × 6-DOF Delsys IMUs + 3 sEMGs + 60 Vicon markers) | **2,000 Hz** IMU/EMG; **100.0 Hz** MoCap/IK | 13 (`P01..P13`) | **Yes** (`Elbow Flexion` task + `elbow_flex_r/l`) | **Yes** (`pro_sup_r/l` in OpenSim IK) | **Yes** (54 OpenSim IK coordinates in deg at 100 Hz) | **Yes** (5 exercises across speeds & target ROMs) | **No** (Speed `Slow..VeryFast` & ROM `45..Max`) |
| **13** | `VIDIMU` | 1.38 GB | 2,005 | `.csv`, `.sto`, `.mot`, `.raw` | **Yes (Bilateral in `A05-A13`)** (`torso`, `humerus_r/l`, `radius_r/l` IMUs) | **333.3 Hz** `.raw`; **50.0 Hz** `.sto`/`.mot`; ~30 Hz video | 53 (`16 IMU+Video S40..S57`, `37 Video-only S01..S39`) | **Yes** (`elbow_flex_r/l` in `.mot`) | **Yes** (`pro_sup_r/l` in `.mot`) | **Yes** (39 OpenSim IK angles in deg at 50 Hz) | **Yes** (13 activities `A01..A13`; `A05..A13` upper limb) | **No** |

---

## 2. Detailed Dataset-by-Dataset Technical Audit

### Dataset 1: `EMG elbow dataset`
- **Location & Size:** `d:\me\project\datasets\raw\EMG elbow dataset` (`304.30 MB`, `130` `.txt` files across `subject 1` to `subject 10`).
- **Documentation & Source:**
  - *File-Verified:* Each subject folder contains `subject_info.txt` (`id`, `age` 18–26 yrs, `height` 150–180 cm, `weight` 50–77 kg, `arm_length`, `hand_length`, `sex`: 6 males, 4 females). No README or license file exists locally.
  - *Literature Context:* Published on Zenodo by Toro-Ossaba & Tejada (Shimmer3 sEMG and angle dataset).
- **Schema, Units & Sampling Rate:**
  - 120 headerless, tab-separated `.txt` files (`<id>_<movement>_<split>_<load>.txt`) with **5 float columns** (`3,307,572` total rows; `19,767` to `63,701` rows/file):
    - `Col 0–1`: Raw 2-channel sEMG ($\text{mV}$).
    - `Col 2–3`: Bandpass-filtered 2-channel sEMG ($\text{mV}$, zero-mean).
    - `Col 4`: Continuous joint angle ($\text{deg}$): spans `[0.21°, 135.55°]` in `flex` files and `[-69.74°, +67.87°]` in `pronsup` files.
  - *Timestamps / Sampling Rate:* No timestamp column exists locally. Literature documents Shimmer3 sEMG acquisition at ~1,024 Hz (10–500 Hz bandpass).
- **Sensor Types & Placement:** 2 surface EMG channels (biceps/triceps or pronator/biceps) + 1 Shimmer3 derived scalar angle. **No raw accelerometer or gyroscope columns exist.**
- **Subjects & Labels:** 10 subjects (`1..10`), 2 movements (`flex` = elbow flexion/extension, `pronsup` = forearm pronation/supination), 3 external loads (`0g`, `1360g`, `2270g`), pre-partitioned into `train` and `test` files per subject.
- **Data Quality:** `0` NaNs across all 120 files; `71` exact duplicate rows (`<0.003%`).
- **Suitability:** Cannot be used for raw dual-IMU modeling because raw IMU channels are absent (only sEMG + scalar angle).

---

### Dataset 2: `JU-IMU`
- **Location & Size:** `d:\me\project\datasets\raw\JU-IMU\ROM` (`383.21 MB`, `574` `.csv` files).
- **Documentation & Source:**
  - *File-Verified:* No README or license file exists locally.
  - *Literature Context:* Jeonju University IMU Dataset (`youngminoh7/JU-IMU`), collected from non-disabled controls and hemiparetic stroke patients.
- **Schema, Units & Sampling Rate:**
  - All 574 files (`ROM_<SubjectID>_<Activity>.csv`) have **45 columns** (`240` to `3,715` rows/file, median `911.5` rows):
    - `sensor1.accx..magz` through `sensor5.accx..magz` (5 sensors × 9 axes: 3D accelerometer in $\text{m/s}^2$ [norm $\approx 9.8\text{ m/s}^2$], 3D gyroscope in $\text{rad/s}$, 3D magnetometer in $\mu\text{T}$).
  - *Timestamps / Sampling Rate:* No timestamp column in the CSVs. Literature specifies a mean sampling rate of **80 Hz**.
- **Sensor Placement:**
  - *Literature Context:* `sensor1` = Right wrist/forearm, `sensor2` = Left wrist/forearm, `sensor3` = Center of trunk (chest), `sensor4` = Right upper arm, `sensor5` = Left upper arm. Thus **bilateral Upper Arm + Forearm dual-IMUs exist** (`sensor4` + `sensor1` for right arm; `sensor5` + `sensor2` for left arm).
- **Subjects & Labels:**
  - **41 unique subjects:** `27` Non-Disabled (`ND4`–`ND33`, with numbering gaps) and `14` Stroke patients (`Stroke2`–`Stroke16`, with `Stroke1` and `Stroke11` absent).
  - **14 upper-limb ROM / rehab activities** (41 files each, perfectly balanced across activities): `ElbFlex` (elbow flexion/extension), `SupPro` (forearm supination/pronation), `FrontSupPro`, `WrstFlex`, `WrstExt`, `FrontWrstFlex`, `FrontWrstExt`, `ShFlex90`, `ShExt`, `ShAbd90`, `Scaption90`, `ShHorAdd`, `ShRot`, and `PickCup`.
- **Data Quality:** `0` NaNs and `0` duplicate rows across all 574 files. Cohort split is imbalanced (`27 ND` vs. `14 Stroke`). No ground-truth joint angle column is stored.

---

### Dataset 3: `mm-fit`
- **Location & Size:** `d:\me\project\datasets\raw\mm-fit` (`3.53 GB`, `302` files: `281` `.npy` and `21` `.csv` across `w00`–`w20`).
- **Documentation & Source:**
  - *File-Verified:* No local README or license file.
  - *Literature Context:* Strömbäck et al., *MM-Fit: Multimodal Deep Learning for Automatic Exercise Logging Across Sensing Devices*, IMWUT 2020.
- **Schema, Units & Sampling Rate:**
  - Inertial `.npy` arrays of shape `(N, 5)`: `[frame_index, unix_timestamp_ms, x, y, z]` (`acc` in $\text{m/s}^2$, `gyr` in $\text{rad/s}$, `mag` in $\mu\text{T}$).
  - Pose `.npy` arrays: `pose_2d.npy` `(2, F, 19)` and `pose_3d.npy` `(3, F, 18)` at **30 Hz**.
  - `wXX_labels.csv`: `[start_frame, end_frame, repetitions, activity_label]`.
  - *Calculated Sampling Rates:* Left/Right smartwatches (`sw_l`, `sw_r`): **~100 Hz** (median $\Delta t = 10.0\text{ ms}$); Left earbud (`eb_l`): **~91.6 Hz**; Right pocket smartphone (`sp_r`): **~217–500 Hz**.
- **Sensor Placement:** Wrists (`sw_l`, `sw_r`), ear (`eb_l`), trouser pockets (`sp_r`, `sp_l`). **No upper-arm IMU exists.**
- **Subjects & Labels:** 21 workouts (`w00`–`w20`; subject-to-workout mapping is not stored locally). 10 gym exercises (`616` sets, `6,160` reps total), including `bicep_curls` (`59` sets) and `tricep_extensions` (`64` sets).
- **Data Quality:** `0` NaNs, but `5,256,856` duplicate millisecond timestamps across sensor arrays due to Bluetooth packet batching, and missing sensor modalities (`sp_l` only in 6/21 workouts, `sw_r_hr` in 14/21, `sw_l_hr` in 18/21).

---

### Dataset 4: `MoCap and IMU Dataset for Upper Limb Joint Angle Estimation`
- **Location & Size:** `d:\me\project\datasets\raw\MoCap and IMU Dataset for Upper Limb Joint Angle Estimation` (`1.45 GB`, `378` `.csv` files across `Take1`–`Take6`, 63 files/take).
- **Documentation & Source:**
  - *File-Verified:* No local README/license; `Toma*.csv` headers document OptiTrack Motive `Format Version 1.25`, `Capture Frame Rate 50.000000`, `2025-06-20` to `2025-06-23`.
  - *Literature Context:* Niño-Tejada et al., *Electronics* 2025 (Zenodo `10.5281/zenodo.15778949`).
- **Schema, Units & Sampling Rate:**
  - `df_final.csv` (`162,064` total rows across 6 takes at **50.0 Hz**, `27` columns): 3 IMUs (`chest`, `Rarm`, `Larm`) × 9 channels (`yaw, pitch, roll` in $\text{deg}$, `wx, wy, wz` in $\text{deg/s}$, `ax, ay, az` in $g$).
  - `df_angles.csv` (`162,064` rows at **50.0 Hz**, `10` columns in $\text{deg}$): `L_elbow_z`, `L_elbow_y`, `L_shoulder_z`, `L_shoulder_y`, `L_shoulder_x`, `R_elbow_z`, `R_elbow_y`, `R_shoulder_z`, `R_shoulder_y`, `R_shoulder_x`.
  - Raw/intermediate subfolders (`Acc`, `Gyro` at 50 Hz; `Eul`, `Quat` raw at 100 Hz subsampled to 50 Hz) and full 47-bone OptiTrack MoCap (`Toma15..20_Z.csv`).
- **Sensor Placement:** 3 IMUs (`chest`, `Rarm` [right wrist], `Larm` [left wrist]). **Only 1 IMU per arm (no upper-arm IMU)**, although `Toma*.csv` has full optical upper-arm (`LUArm`, `RUArm`) and forearm (`LFArm`, `RFArm`) bone trajectories.
- **Subjects & Labels:** 6 continuous takes (`Take1`–`Take6`, all using `Skeleton 002`; subject count undocumented locally). Continuous 10-DOF joint angle regression targets (`L/R_elbow_z` flexion/extension and `L/R_elbow_y`), with no discrete exercise class labels.
- **Data Quality:** `0` NaNs and `0` duplicate timestamps across all 378 files.

---

### Dataset 5: `Optotrak vs DOT`
- **Location & Size:** `d:\me\project\datasets\raw\Optotrak vs DOT` (`5.43 GB`, `7,711` files).
- **Documentation & Source:**
  - *File-Verified:* Includes `Readme.docx` (`37,547` B), per-subject `Info.txt` / `Notes.txt`, and the complete MATLAB pipeline (`Codes\A_Optotrack_sortData.m` through `J_calculateStatisticsV2.m`). Collected at the University Medical Centre Groningen (UMCG), January 2023, following ISB upper-limb kinematic standards (Wu et al., 2005).
- **Schema, Units & Sampling Rate:**
  - **Raw Xsens DOT CSVs (`1,600` active files = 16 subjects × 20 trials × 5 sensors):** 15 columns (`PacketCounter`, `SampleTimeFine` in $\mu\text{s}$, `Quat_W..Z`, `Acc_X..Z` in $\text{m/s}^2$, `Gyr_X..Z` in $\text{deg/s}$, `Mag_X..Z` normalized). Verified sampling rate: **120.0048 Hz** ($\Delta t = 8333\ \mu\text{s}$).
  - **MATLAB Struct Files (saved as binary `.mat` v5 files without file extension, readable via `scipy.io.loadmat`):**
    - `B_dotComplete`: 20-trial struct per subject with fields `TRK`, `RSC`, `RUA`, `RLA`, `RHD`, each `(T, 18)` (15 raw columns + 3 `ZYX` Euler angles in $\text{deg}$).
    - `A_OptotrackComplete`: 20-trial struct per subject with 3D marker coordinates (`mm`) at **120 Hz** for 15 cluster markers + 19 bony landmarks + `GHJC`.
    - `H_Joint_Angles_Aligned_FINAL`: Aligned continuous 3D Euler joint angles (`(T, 3)` in $\text{deg}$) at **120 Hz** for 5 joints (`Elbow`, `GlenoHumeral`, `HumeroThoracic`, `ScapuloThoracic`, `Wrist`) across ground-truth Optotrak (`opto`) and 4 IMU calibration models (`nPose`, `func`, `manual`, `oneAxis`).
    - `I_tasks_split`: Repetition-segmented joint angles for the 10 functional/ROM tasks (5 reps per task).
- **Sensor Placement (Verified):**
  - **5 Xsens DOT IMUs + 5 Optotrak 3-marker rigid clusters:**
    1. `TRK`: Thorax / Sternum
    2. `RSC`: Right Scapula
    3. `RUA`: **Right Upper Arm**
    4. `RLA`: **Right Lower Arm (Forearm)**
    5. `RHD` / `RHA`: Right Hand
  - **Matches Target Dual-IMU Placement (`RUA` Upper Arm + `RLA` Forearm)!**
- **Subjects & Labels:**
  - **16 subjects** (`subject 1`–`subject 16`; 6M/10F, ages 20–44). Per `Readme.docx`, 13 subjects have complete `I_tasks_split` (`subject 2` recorded on left arm due to right-side injury; `subject 4` had hand IMU issue on task 20; `subject 8` lacked medial epicondyle landmark digitization for optical elbow angle).
  - **20 Trials per Subject:**
    - Trials 1–10 (Calibration): `1` N-Pose, `2` Elbow Flex/Ext (15°–50°), `3` Elbow Pron/Sup, `4` Wrist Flex/Ext, `5` Wrist Rad/Uln Dev, `6` Shoulder Int/Ext Rot, `7` Shoulder Flex/Ext, `8` Shoulder Abd/Add, `9` Scapula Elevation, `10` Scapula Protraction.
    - Trials 11–20 (10 Evaluation Tasks, 5 reps each): `11` **Elbow Flexion/Extension (full ROM)**, `12` **Forearm Pronation/Supination (full ROM)**, `13` **Shoulder Flexion**, `14` **Shoulder Abduction**, `15` **Shoulder Internal/External Rotation**, `16` **Wrist Flexion/Extension**, `17` **Wrist Radial/Ulnar Deviation**, `18` **Drinking from a Cup**, `19` **Box off Shelf**, `20` **Drawing Circles**.
- **Data Quality:**
  - `0` NaNs and `0` duplicate timestamps across all 1,600 raw DOT CSVs and all 16 `B_dotComplete` files.
  - Known quirk 1: Row 0 (`PacketCounter == 0`) in raw DOT CSVs has zeroed `Acc/Gyr` (`0,0,0,0,0,0`) and must be dropped (as `B_sortDotData.m` does).
  - Known quirk 2: 5 of 1,600 raw CSVs omit the first `sep=,` line.
  - Known quirk 3: 69 discarded backup CSVs exist inside `delete` / `Fist attempt before deleting RLA` subfolders and must be excluded by path filtering.

---

### Dataset 6: `PHYTMO`
- **Location & Size:** `d:\me\project\datasets\raw\PHYTMO` (`11.68 GB`, `7,076` files: `7,075` `.csv` + `1` `.m`).
- **Documentation & Source:**
  - *File-Verified:* `features_extraction.m` cites García-de-Villa, Jiménez-Martín, & García-Domínguez (Zenodo / *Scientific Data* 9, 266, 2022: *"A database of physical therapy exercises with variability of execution collected by wearable sensors"*).
- **Schema, Units & Sampling Rate:**
  - `inertial\upper\` (`1,484` CSVs = 371 trial recordings × 4 upper-limb IMUs) and `inertial\lower\` (`3,384` CSVs):
    - **10 Columns:** `Time (s)`, `Gyroscope X (deg/s)`, `Gyroscope Y (deg/s)`, `Gyroscope Z (deg/s)`, `Accelerometer X (g)`, `Accelerometer Y (g)`, `Accelerometer Z (g)`, `Magnetometer X (uT)`, `Magnetometer Y (uT)`, `Magnetometer Z (uT)`.
    - **Calculated Sampling Rate:** Median **100.89 Hz** ($\Delta t \approx 9.91\text{ ms}$; nominal **100 Hz**). Magnetometer updates at ~20 Hz (zero-order held every 5 samples).
  - `optical raw\` & `optical interp\`: OptiTrack Motive CSVs (`84` columns of rigid-body quaternions and 3D marker coordinates in meters at **100.0 Hz**).
- **Sensor Placement (Verified):**
  - `inertial\upper\`: **4 simultaneous 9-axis NGIMUs:** `Larm` (Left Upper Arm), `Lforearm` (Left Forearm), `Rarm` (Right Upper Arm), `Rforearm` (Right Forearm).
  - **Matches Target Dual-IMU Placement Bilaterally (`Larm`+`Lforearm` and `Rarm`+`Rforearm`)!**
- **Subjects & Labels:**
  - **30 subjects** across 5 age/cohort groups (`A01–A10`, `B01–B05`, `C01–C05`, `D01–D05`, `E01–E05`).
  - **Upper-Limb Exercise Classes (`inertial\upper`):**
    1. `EFE`: **Elbow Flexion-Extension** (`54` `EFE0` trials + `54` `EFE1` trials × 4 sensors = `432` CSVs)
    2. `EAH`: **Extension of Arms over Head** (`58` `EAH0` trials + `58` `EAH1` trials × 4 sensors = `464` CSVs)
    3. `SQZ`: **Squeezing** (`59` `SQZ0` trials + `59` `SQZ1` trials × 4 sensors = `472` CSVs)
    4. `Calib`: Static/calibration trials (`29` trials × 4 sensors = `116` CSVs).
  - **Correct vs. Incorrect Execution Labels (Verified):**
    - **`0` = Correct execution** (`EFE0`, `EAH0`, `SQZ0`: 171 upper-limb trials / 684 sensor CSVs)
    - **`1` = Incorrect / compensatory execution** (`EFE1`, `EAH1`, `SQZ1`: 171 upper-limb trials / 684 sensor CSVs).
- **Data Quality:**
  - `0` NaNs across all `1,484` upper-limb inertial CSVs and `3,384` lower-limb inertial CSVs.
  - `98` of `1,484` upper-limb inertial CSVs have occasional duplicate consecutive timestamps (`252` duplicate steps total), easily handled via timestamp deduplication/resampling to uniform 100 Hz.
  - Optical CSVs (`optical raw` and `optical interp`) have missing cells (`,,`) from marker occlusions (`55%` of raw optical files, `33%` of interpolated optical files).

---

### Dataset 7: `REHAB` (REHAB-120)
- **Location & Size:** `d:\me\project\datasets\raw\REHAB` (`1.07 GB`, `36,110` `.npy` files).
- **Documentation & Source:**
  - *File-Verified:* No local README/license file.
  - *Literature Context:* Zhou et al. (*Scientific Data* / *J. NeuroEng. Rehabil.* 2025, REHAB-120 dataset of 120 post-stroke patients across 222 assessment sessions).
- **Schema, Units & Sampling Rate:**
  - `Rehab_assessment\Label.npy`: `(222, 27)` `int64` clinical FMA-UE item scores (`0`, `1`, `2`).
  - `Rehab_assessment\d01_raw_data\001..222` (variable length $T$) & `d02_processed_data\001..222` (interpolated to fixed $T = 150$):
    - Tasks `000–014` & `026` (16 proximal arm tasks): `XXX_1.npy` `(T, 12)` (3 Euler deg, 3 Acc $g$, 3 Mag raw, 3 Gyr deg/s) and `XXX_2.npy` `(T, 13)` (1 joint angle deg + 3 Euler deg, 3 Acc $g$, 3 Mag raw, 3 Gyr deg/s) are active; `XXX_3.npy` `(T, 5)` is all zeros.
    - Tasks `015–025` (11 hand/wrist tasks): `XXX_3.npy` `(T, 5)` (data glove) is active; `XXX_1.npy` and `XXX_2.npy` are all zeros.
  - `Rehab_exercise\d01_raw_data` & `d02_processed_data`: `000_1..015_1.npy` and `000_2..015_2.npy` of shape `(N_trials, 880, 6)` (`4,616` total trials across 16 exercises).
  - *Sampling Rate:* 50 Hz (literature; no timestamps in `.npy`).
- **Data Quality & Critical Issues:**
  1. **Corrupted Binary File:** `Rehab_exercise\d02_processed_data\014_1.npy` (`26.98 MB`) is corrupted by UTF-8 replacement bytes (`\xef\xbf\xbd`) in the `.npy` magic header (`d01_raw_data\014_1.npy` is intact).
  2. **Duplicate Nested Folder:** `Rehab_assessment\d02_processed_data\195\2021-03-03 11-00-11\` contains 81 extra `.npy` files.
  3. **Missing Patient-to-Session Mapping (Leakage Risk):** The 222 folders in `Rehab_assessment` come from 120 stroke patients (some evaluated multiple times), and `Rehab_exercise` aggregates trials across patients without subject IDs. Splitting by session index `001..222` or trial index risks **subject leakage** because no patient-ID lookup table is included locally.

---

### Dataset 8: `Smartwatch IMU Dataset for Upper-Limb Movement Ana`
- **Location & Size:** `d:\me\project\datasets\raw\Smartwatch IMU Dataset for Upper-Limb Movement Ana` (`576.41 MB`, `11` files).
- **Documentation & Source:**
  - *File-Verified:* `4 - docs\README_data.md`, `columns.md`, `provenance.json`. Benachour et al., *Engineering Applications of Artificial Intelligence* 156:111194 (2025). License: **CC BY 4.0**.
- **Schema, Units & Sampling Rate:**
  - `1 - raw\all_users_{train,valid,test}.csv` (`188,717`, `105,043`, and `101,674` rows × `40` columns): Apple Watch Series 6 `AccelroX, AceelroY, AceelroZ` ($g$), `GyroX..Z` ($\text{rad/s}$), CoreMotion `DMUAccelX..Z` ($g$), `DMRotX..Z` ($\text{rad/s}$), `DMGrvX..Z` ($g$), `DMPitch, DMRoll, DMYaw` ($\text{rad}$), `DMQuatW..Z`, `TimeStamp`, `Hertz`, `UID`, `MoveType`, `Side`, `Wrist`, `SessionID`.
  - *Sampling Rate Discrepancy:* Docs state `20 Hz`, but **96.4% of rows have `Hertz == 100` (measured median `99.35 Hz`)** and only **3.6% have `Hertz == 20` (`20.00 Hz`)**.
- **Sensor Placement:** Single smartwatch on wrist (`lt` or `rt`). **No upper-arm sensor.**
- **Subjects & Labels:** Only **3 subjects** (`u01`, `u04`, `u05`), 3 movements (`el-exfl` elbow flexion/extension, `sh-exfl` shoulder flexion/extension, `wr-prsu` forearm pronation/supination).
- **Data Quality Issues:**
  - `all_users_valid.csv` contains **2 embedded header rows** (`UID == 'UID'`) mid-file.
  - Raw `GyroX..Z` is `NaN` for subject `u01` (`85,819` NaN rows across splits), though CoreMotion `DMRotX..Z` is 100% complete.
  - All 3 subjects appear in all 3 pre-made `train`, `valid`, and `test` files (not subject-independent as pre-split, and $N=3$ subjects is too small for robust subject-independent evaluation).

---

### Dataset 9: `StrokeRehab`
- **Location & Size:** `d:\me\project\datasets\raw\StrokeRehab` (`5.80 GB`, `1,426` `.csv` files under `files\healthy controls\{LT,RT}\<Subject>\<Activity>\`).
- **Documentation & Source:** No local README or license file. Recorded with a **9-sensor Noraxon MyoMotion IMU system**.
- **Schema, Units & Sampling Rate:**
  - **Schema A (`1,418` of `1,426` files, `106` columns at exact `100.00 Hz`):**
    - `Times` ($\text{s}$), **22 anatomical joint angles** in $\text{deg}$ (including **`ElbowFlexionLTdeg`**, **`ElbowFlexionRTdeg`**, **`WristSupinationLTdeg`**, **`WristSupinationRTdeg`**, shoulder 3-DOF, wrist 2-DOF, lumbar/thoracic 3-DOF), **27 segment orientation Euler angles** (`course, pitch, roll` in $\text{deg}$ for 9 segments), **27 linear accelerations** (`mG` for 9 sensors), **27 rotation vectors (`RotX..Z`)**, and frame-level primitive labels (`Markers`, `MarkerNames`).
  - **Schema B (`3` files in `C12`) & Schema C (`5` files in `C26`):** `80` and `84` columns respectively (use underscore naming `ElbowFlexionLT_deg`, omit the 27 segment `course/pitch/roll` columns, and Schema C omits `UpperArmAccelSensorX/Y/Z_LT_mG`).
- **Sensor Placement (Verified):** **9 IMUs** including bilateral **Upper Arm + Forearm dual-IMUs** (`LTUpperarm` + `LTForearm` and `RTUpperarm` + `RTForearm`), plus `LTHand`, `RTHand`, `Upperspine`, `Lowerspine`, `Pelvis`.
- **Subjects & Labels:**
  - **20 healthy control subjects** (`C4`–`C32`; note that the stroke patients branch is not present on disk).
  - **9 upper-extremity ADL / rehab activities** (`feeding`, `brushing`, `face wash`, `drinking`, `glasses`, `combing`, `deodrant`/`deodorant`, `RTT left/right side`, `shelf`/`Shelf left/right side`) plus frame-level movement primitives (`reach_prox`, `reposition_prox`, `transport_prox`, `stabilize`, `retract_prox`, `idle`, `rest`).
- **Data Quality:** `0` duplicates; exact `100.0 Hz` rate; `2` files (`C5` drinking trial 5 in LT/RT) have NaNs in `RotX..Z`; folder names have minor spelling/case variants (`deodrant` vs `deodorant`, `shelf` vs `Shelf`).

---

### Dataset 10: `UCI Biceps Curl`
- **Location & Size:** `d:\me\project\datasets\raw\UCI Biceps Curl\Example_WearableComputing_weight_lifting_exercises_biceps_curl_variations.csv` (`1.59 MB`, `1` `.csv` file).
- **Documentation & Source:** Line 0 header notes this is the **4,024-row example excerpt** of the PUC-Rio Weight Lifting Exercises dataset (Velloso et al., 2013). Requires `skiprows=3` when reading.
- **Schema, Units & Sampling Rate:**
  - `4,024` rows × `159` columns (`6` metadata/timestamps + `52` instantaneous 9-DOF IMU & Euler channels across 4 sensors [`belt`, `arm`, `dumbbell`, `forearm`] + `100` window summary statistics + `1` target column `classe`).
  - Calculated median sampling rate: **49.55 Hz** (nominal 45 Hz).
- **Sensor Placement:** Includes `arm` (upper arm) + `forearm` 9-DOF IMUs (`roll, pitch, yaw, total_accel, gyros_x..z, accel_x..z, magnet_x..z`).
- **Subjects & Labels:** 5 subjects (`pedro`, `carlitos`, `adelmo`, `eurico`, `jeremy`) performing biceps curls across 5 execution classes (`A` = correct, `B, C, D, E` = 4 execution errors).
- **Critical Limitation / Leakage Risk:** Because this file is only a 4,024-row excerpt, **subject and class are completely confounded**: `adelmo`, `eurico`, and `jeremy` only have Class `E`; `pedro` only has `A, B, C`; `carlitos` only has `A, D, E`. A subject-independent split is **mathematically impossible** across all 5 classes on this excerpt file.

---

### Dataset 11: `ULF_in_ADL`
- **Location & Size:** `d:\me\project\datasets\raw\ULF_in_ADL` (`24.66 GB`, `4,492` `.mvnx` XML files + `1` `ParticipantCharacteristics.xlsx`).
- **Documentation & Source:** SoftPro / University Hospital Zurich Upper Limb Function in ADL dataset. `ParticipantCharacteristics.xlsx` provides participant demographics, time since stroke, affected side, and clinical `FMMA-UE` scores.
- **Schema, Units & Sampling Rate:**
  - Xsens MVN `.mvnx` XML format at **60.0 Hz** (`<subject frameRate="60">`): `23` segments (quaternions, 3D positions $\text{m}$, velocities $\text{m/s}$, accelerations $\text{m/s}^2$, angular velocities $\text{rad/s}$), **`17` raw/fused IMU sensors** (`sensorFreeAcceleration` in $\text{m/s}^2$, `sensorMagneticField`, `sensorOrientation` quaternions), and **`22` 3D anatomical joints** (`jointAngle` ZXY and `jointAngleXZY` in $\text{deg}$).
- **Sensor Placement:** 17 Xsens IMUs including bilateral **`RightUpperArm` + `RightForeArm`** and **`LeftUpperArm` + `LeftForeArm`**, providing both IMU streams and continuous 3D elbow joint angles (`jRightElbow`, `jLeftElbow`: flexion/extension and pronation/supination).
- **Subjects & Labels:** **25 subjects** (`20` Stroke patients `P02`–`P30` with `FMMA-UE` scores `21–61`; `5` Healthy controls `H01`–`H05`), performing **30 ADL tasks (`T01`–`T30`)** bilaterally (3 reps per arm).
- **Data Quality:** `0` NaNs and `0` duplicate timestamps in sampled `.mvnx` files; `1/50` sampled files lacked the optional `<sensors>` block while retaining segment/joint kinematics; parsing `24.66 GB` of XML `.mvnx` files is I/O intensive unless converted to binary/Parquet.

---

### Dataset 12: `ULTra-MoCap`
- **Location & Size:** `d:\me\project\datasets\raw\ULTra-MoCap` (`12.29 GB`, `1,102` files across `ULTra-MoCap-processed`, `ULTra-MoCap-raw-0`, `ULTra-MoCap-raw-1`).
- **Schema, Units & Sampling Rate:**
  - `IMU\*.csv` (`217` files, `38` cols at **2,000 Hz**): `Frame`, `Sub Frame`, plus 6 × 6-DOF Delsys Trigno IMUs (`ACCX1..GYROZ6`; Acc in $\text{mm/s}^2$ or $\text{mG}$, Gyro in $\text{deg/s}$).
  - `EMG\*.csv` (`217` files, `5` cols at **2,000 Hz**): `Frame`, `Sub Frame`, `IM EMG4..6` ($\text{V}$).
  - `Processed IK\*.csv` & `Raw IK\*.mot` (`208` files each, `55` cols at **100.0 Hz**): `time` + **54 OpenSim joint angles in degrees** including **`elbow_flex_r`**, **`pro_sup_r`**, **`elbow_flex_l`**, **`pro_sup_l`**, and 3-DOF shoulder/wrist angles.
  - `Raw Marker\*.trc` (`221` files at **100.0 Hz**): 60 3D Vicon optical markers ($\text{mm}$).
  - `All_subjects_data.h5` (`2.09 GB` consolidated HDF5 file).
- **Subjects & Labels:** **13 healthy subjects (`P01`–`P13`)** performing **5 upper-limb exercises** (`Elbow Flexion`, `Shoulder Rotation`, `Crossbody Reach`, `Arm Swing`, `Overhead Reach`) across multiple speeds (`Slow`, `Normal`, `Fast`, `VeryFast`) and target ROM elevations (`45°`, `90°`, `180°`, `Max`).
- **Data Quality:** `0` NaNs and `0` duplicates in sampled CSVs. Note: `9` trials (`P04–P12` `overheadreach_45`) exist only in `IMU`/`EMG` (not in `IK`/`Marker`), and `8` trials (`P05–P12` `overheadreach_90`) differ in filename capitalization between `IMU`/`EMG` and `IK`. Also, the 6 IMU columns are numbered `1..6` without anatomical segment names inside the CSV header.

---

### Dataset 13: `VIDIMU`
- **Location & Size:** `d:\me\project\datasets\raw\VIDIMU\dataset` (`1.38 GB`, `2,005` files).
- **Schema, Units & Sampling Rate:**
  - `.raw` (`242` files at **333.33 Hz**): Raw 5-IMU quaternions (`w, x, y, z, timestamp`).
  - `.sto` (`208` files at **50.0 Hz**): Resampled OpenSim 5-IMU quaternions (`torso_imu`, **`humerus_r_imu`**, **`radius_r_imu`**, **`humerus_l_imu`**, **`radius_l_imu`** in upper-limb activities `A05–A13`).
  - `.mot` (`255` files at **50.0 Hz**): 39 OpenSim IK joint angles in **degrees**, including **`elbow_flex_r`**, **`pro_sup_r`**, **`elbow_flex_l`**, **`pro_sup_l`**.
  - `.csv` (`873` trial files + `217` `_Npose.csv` at **~30 Hz**): 34 3D video pose landmarks ($\text{mm}$).
- **Sensor Placement:** In activities `A05–A13`, 5 IMUs are placed on the torso and bilaterally on the **upper arm (`humerus_r/l_imu`) + forearm (`radius_r/l_imu`)**. However, the IMU files store **only 4D orientation quaternions** (no raw accelerometer or gyroscope channels).
- **Subjects & Labels:** `16` subjects with IMU + Video (`S40`–`S57`) and `37` subjects with Video only (`S01`–`S39`), across `13` activities (`A01–A04` lower limb, `A05–A13` upper limb).
- **Data Quality Issues:** `videoandimus\S48` contains `13` extra unprefixed `S48_A01..13.mot` files at 60 Hz whose internal header shows they are actually mislabeled copies of `ik_S53_A01..13` (`ik_S48_A01..13.mot` are the valid 50 Hz files for `S48`).

---

## 3. License and Usage Restrictions

- **Verifiable Local License:**
  - **`Smartwatch IMU Dataset for Upper-Limb Movement Ana`**: Explicitly licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)** in `4 - docs\README_data.md` and `provenance.json`.
- **Literature / Repository Licenses (No Local `LICENSE` File Stored in Folder):**
  - The remaining 12 datasets were downloaded from academic open-access repositories (Zenodo, PhysioNet/Figshare, GitHub, PUC-Rio, UMCG) accompanying peer-reviewed publications (e.g., *Scientific Data*, *Electronics*, *IMWUT*). No local `LICENSE` text file is present in their directories; usage in academic research requires standard citation of the respective original publications, and redistribution of raw data outside local storage is prevented via `.gitignore`.

---

## 4. Capability Analysis: What the Data Can and Cannot Support

### A. Tasks Genuinely Supported by Existing Local Datasets

1. **Rehabilitation Exercise Classification Using Upper-Arm + Forearm Dual-IMU Signals (Strongly Supported):**
   - **`Optotrak vs DOT`**: 16 subjects, 120 Hz raw 9-axis/quaternion Xsens DOT IMUs on **Right Upper Arm (`RUA`) + Right Forearm (`RLA`)**, 10 well-defined upper-limb rehabilitation & functional tasks (`Elbow Flex/Ext`, `Forearm Pron/Sup`, `Shoulder Flex`, `Shoulder Abd`, `Shoulder Int/Ext Rot`, `Wrist Flex/Ext`, `Wrist Rad/Uln Dev`, `Drinking`, `Box off Shelf`, `Drawing Circles`).
   - **`JU-IMU`**: 41 subjects (`27 ND` + `14 Stroke`), 80 Hz 9-axis IMUs on **Upper Arm (`sensor4/5`) + Forearm/Wrist (`sensor1/2`)**, 14 upper-limb ROM and rehab exercises (`ElbFlex`, `SupPro`, `FrontSupPro`, wrist and shoulder ROM tasks, `PickCup`).
   - **`PHYTMO` (`inertial\upper`)**: 30 subjects, 100 Hz 9-axis NGIMUs on **Upper Arm (`Larm/Rarm`) + Forearm (`Lforearm/Rforearm`)**, 3 upper-limb exercise classes (`EFE` Elbow Flexion-Extension, `EAH` Extension of Arms Overhead, `SQZ` Squeezing).

2. **Correct vs. Incorrect Exercise Execution Assessment Using Dual-IMU Signals (Supported on Specific Exercises):**
   - **`PHYTMO` (`inertial\upper`)**: **Directly supports binary Correct (`0`) vs. Incorrect/Compensatory (`1`) execution classification** across **30 subjects** with bilateral **Upper Arm + Forearm 9-axis IMUs at 100 Hz**, specifically including **Elbow Flexion-Extension (`EFE0` vs. `EFE1`)** as well as `EAH0` vs. `EAH1` and `SQZ0` vs. `SQZ1` (balanced 1:1 across 342 upper-limb exercise trials).
   - **`UCI Biceps Curl`**: Contains 5-class execution quality labels (`A`–`E`) for biceps curls with `arm` + `forearm` IMUs, **but** the local file is only a 4,024-row excerpt where subject ID and class label are confounded (3 of 5 subjects only have Class `E`), making subject-independent training/evaluation invalid on the local excerpt.
   - **`JU-IMU`, `ULF_in_ADL`, and `REHAB`**: Do not label individual repetitions as "correct vs. incorrect form," but provide **clinical impairment labels** (`ND` vs. `Stroke` in `JU-IMU`; paretic vs. non-paretic limb and `FMMA-UE` score `21–61` in `ULF_in_ADL`; item-level FMA-UE `0, 1, 2` in `REHAB`).

3. **Continuous Elbow Motion & Range of Motion (ROM) Estimation — Flexion/Extension and Pronation/Supination (Strongly Supported):**
   - **`Optotrak vs DOT`**: Provides synchronized **120 Hz** raw `RUA` (upper arm) + `RLA` (forearm) IMU streams alongside gold-standard **Optotrak 3D `Elbow` joint angles** (flexion/extension and pronation/supination in degrees) across 16 subjects and 20 trials.
   - **`StrokeRehab`**: Provides synchronized **100 Hz** bilateral Upper Arm + Forearm IMU streams alongside continuous `ElbowFlexionLT/RTdeg` and `WristSupinationLT/RTdeg` across 20 subjects (`1,426` trials).
   - **`ULF_in_ADL`**: Provides **60 Hz** Xsens MVN bilateral Upper Arm + Forearm IMU streams and continuous 3D elbow joint angles (`jRightElbow`, `jLeftElbow`) across 25 subjects (`20 Stroke`, `5 Healthy`, `4,492` trials).
   - **`VIDIMU` (`A05–A13`)**: Provides **50 Hz** bilateral `humerus` + `radius` IMU quaternions and OpenSim IK `elbow_flex_r/l` & `pro_sup_r/l` across 16 subjects (`144` upper-limb trials).

### B. Tasks That Cannot Yet Be Performed Reliably with Existing Data

1. **Multi-Exercise Correct vs. Incorrect Form Classification Across All Elbow/Forearm Rehab Movements (e.g., Pronation/Supination Form Errors):**
   - Only `PHYTMO` has subject-independent, full-trial dual-IMU correct/incorrect labels (`EFE`, `EAH`, `SQZ`), and it does **not** include a forearm pronation/supination exercise. None of the datasets that include forearm pronation/supination (`Optotrak vs DOT`, `JU-IMU`, `StrokeRehab`, `ULTra-MoCap`, `VIDIMU`) contain "incorrect execution" trials.
2. **Direct Clinical Validation on Post-Olecranon ORIF Surgery Patients:**
   - None of the 13 public datasets were collected from patients recovering from **olecranon ORIF fracture surgery**. The patient cohorts present in the workspace (`JU-IMU`, `REHAB`, `ULF_in_ADL`) are **post-stroke hemiparetic patients**, while `Optotrak vs DOT`, `PHYTMO`, `ULTra-MoCap`, `VIDIMU`, and the local `StrokeRehab` branch consist of **healthy volunteers**.

---

## 5. Potential Data Leakage Risks Identified

1. **Window-Overlap Leakage (Critical across all continuous time-series datasets):**
   - Extracting overlapping sliding windows (e.g., 50% overlap) *before* splitting into `train`/`val`/`test` causes nearly identical adjacent windows from the same trial to appear in both training and test sets, artificially inflating accuracy and F1-scores near 99–100%. **All splits must be performed at the Subject ID level first.**
2. **Subject Leakage in Pre-Split or Unmapped Datasets:**
   - In `Smartwatch IMU Dataset`, the author-provided `all_users_train.csv`, `all_users_valid.csv`, and `all_users_test.csv` contain data from **all 3 subjects (`u01, u04, u05`) in every split**.
   - In `EMG elbow dataset`, `train` and `test` files exist within each of the 10 subjects (intra-subject split).
   - In `REHAB`, the 222 sessions come from 120 patients without a local patient-ID mapping table, meaning random session splitting leaks longitudinal visits of the same patient across splits.
   - In `mm-fit`, 21 workouts (`w00`–`w20`) come from 10 participants without a local workout-to-subject table.
3. **Subject-Class Confounding in `UCI Biceps Curl`:**
   - Because 3 of 5 subjects in the 4,024-row excerpt only perform Class `E`, any model trained without strict awareness of this confounding can memorize subject sensor bias rather than movement quality.
4. **Normalization / Filtering Leakage:**
   - Computing channel-wise mean/standard deviation ($z$-score) or min-max bounds across the full dataset before splitting leaks test distribution statistics (e.g., the pre-computed `Normalized` subfolders in `MoCap and IMU Dataset` were min-max scaled per take). Normalization parameters must be fitted **exclusively on the training subjects**.
5. **Cross-Subject File Contamination in `VIDIMU`:**
   - As discovered during this audit, `VIDIMU\dataset\videoandimus\S48\` contains 13 unprefixed files (`S48_A01..13.mot`) that are actually copies of Subject `S53` (`name=ik_S53_A01..13`). If `S48` is placed in `train` and `S53` in `test` without filtering out those 13 unprefixed files, `S53` ground-truth trajectories leak directly into `S48`.

---

## 6. Limitations for Applying Public Dataset Results to Post-Olecranon ORIF Patients

1. **Pathomechanical Differences (Orthopedic Post-ORIF vs. Healthy / Post-Stroke):**
   - Post-olecranon ORIF patients experience mechanical articular block, surgical hardware constraints, triceps insertion tenderness, joint capsule stiffness, and pain-guarded terminal extension/flexion—a biomechanical profile distinct from both healthy adults (who achieve full, smooth $0^\circ\text{–}145^\circ$ ROM) and post-stroke patients (who exhibit spasticity, abnormal flexor synergy, and paresis).
2. **Restricted Range of Motion & Slower Angular Velocities:**
   - Early-to-mid stage post-ORIF rehabilitation typically operates within restricted safe ROM arcs (e.g., $30^\circ\text{–}100^\circ$ flexion, limited active extension against resistance) and at much slower angular velocities than healthy benchmark trials.
3. **Sensor Mounting & Soft-Tissue Artifacts:**
   - Post-operative edema, dressings, or splints around the posterior elbow can alter forearm/upper-arm strap placement and soft-tissue skin motion compared to laboratory Xsens DOT or NGIMU mounting on healthy limbs.
4. **Definition of "Correct vs. Incorrect" Execution:**
   - In `PHYTMO`, "incorrect" execution (`EFE1`) was simulated by healthy subjects performing predefined compensatory movements. In post-ORIF clinical care, "incorrect" execution includes exceeding surgeon-prescribed ROM/load limits, shoulder hiking/trunk compensation due to elbow stiffness, or jerky movement velocity.

---

## 7. Recommended Next Step

Proceed to **Stage 2 review** using [`docs/DATA_PREPARATION_PLAN.md`](DATA_PREPARATION_PLAN.md), which proposes a clean, subject-independent data preparation and 4-model benchmark plan (`TCN`, `1D CNN`, `LSTM`, `GRU`) anchored on our verified dual-IMU datasets (**`PHYTMO`** for exercise + execution quality classification, and **`Optotrak vs DOT`** / **`JU-IMU`** for multi-exercise rehabilitation classification and elbow ROM/angle estimation), while keeping all raw datasets untouched.
