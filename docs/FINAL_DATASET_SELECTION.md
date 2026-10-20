# Final Dataset Selection & Task Suitability Decision (`docs/FINAL_DATASET_SELECTION.md`)

**Project:** AI-Based Elbow Rehabilitation Assistance System (Post-Olecranon ORIF Recovery)  
**Stage:** Stage 9 — Final Post-Benchmark Dataset Audit and Selection  
**Target Hardware:** Dual 6-Axis IMU (`IMU 1`: Upper Arm Acc XYZ + Gyr XYZ; `IMU 2`: Forearm Acc XYZ + Gyr XYZ $\rightarrow 12\text{ SI channels}$)

---

## 1. Executive Summary of Dataset Decisions

Following the execution of all 14 benchmark tracks in Stage 8 ([`experiments/reports/MODEL_COMPARISON_REPORT.md`](file:///d:/me/project/experiments/reports/MODEL_COMPARISON_REPORT.md)) and the Stage 9 forensic audit ([`experiments/reports/STAGE9_AUDIT_REPORT.md`](file:///d:/me/project/experiments/reports/STAGE9_AUDIT_REPORT.md)), each of the **8 benchmark datasets** (plus the 5 previously excluded datasets) has been evaluated for final model training across our six target capabilities:

1. **Continuous Elbow Flexion/Extension Angle Estimation ($\theta_{\text{flex}}(t)$ in `deg`)**
2. **Continuous Forearm Pronation/Supination Angle Estimation ($\theta_{\text{pro/sup}}(t)$ in `deg`)**
3. **Range-of-Motion (`ROM`) Measurement (`deg`)**
4. **Rehabilitation Exercise Classification**
5. **Correct vs. Incorrect Exercise Execution Classification**
6. **Stroke-Related / Impaired Upper-Limb Movement Analysis**

### Master Final Dataset Decision Table (8 Benchmark Datasets)

| # | Dataset | Final Decision | Approved Tasks & Exact Target Labels | Sensor Placement, Channels & Rate | Ground-Truth Source | Subject Count & Key Limitations | Can Combine With Other Datasets? |
| :- | :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | **`Optotrak vs DOT`** | **KEEP (`PRIMARY` for Angles, ROM & Exercise Classif.)** | • **Elbow Flex/Ext Angle & ROM:** `Elbow[:, 2]` (`deg`)<br>• **Forearm Pro/Sup Angle & ROM:** `Elbow[:, 1]` (`deg`)<br>• **Rehab Exercise Classif.:** 10 tasks (`Trials 11..20`) | Right Upper Arm (`RUA`) + Right Forearm (`RLA`) Xsens DOT; 12-ch 6-DOF Acc (`m/s²`) + Gyr (`deg/s` $\rightarrow$ `rad/s`) at **`120.0 Hz`** | 120 Hz Optotrak Certus 3D optical marker clusters + ISB bony landmarks (`H_Joint_Angles_Aligned_FINAL`) | `16` healthy subjects (`15` right-arm in Track 1A; `14` with Optotrak GT in Track 3A; `subj 2` left-arm, `subj 8` missing `ME` landmark). Healthy adults only. | **Classification only:** Can combine with `JU-IMU` (7 shared classes) and `PHYTMO` (2 shared classes) **only** via orientation-invariant features at `100 Hz`. **Never** combine raw angle targets with OpenSim IK datasets due to zero-offset differences. |
| **2** | **`PHYTMO`** | **KEEP (`PRIMARY` for Execution Quality & Exercise Classif.)** | • **Correct vs. Incorrect Execution:** Binary `0` (`EFE0` Correct) vs `1` (`EFE1` Incorrect)<br>• **Exercise Classif.:** 3 classes (`EFE, EAH, SQZ`) & 6-class joint (`EFE0..SQZ1`) | Bilateral Upper Arm (`Rarm, Larm`) + Forearm (`Rforearm, Lforearm`) NGIMUs; 12-ch 6-DOF Acc (`g` $\rightarrow$ `m/s²`) + Gyr (`deg/s` $\rightarrow$ `rad/s`) at **`100.89 Hz`** ($\rightarrow$ `100.0 Hz`) | Clinical physical therapy protocol labels (`0` = proper form, `1` = compensatory form) | `30` healthy subjects (`A01..E05`, `342` active upper-limb trials). `A06, A07, B03` lack `EFE`; `A02` lacks `EAH`. `98` CSVs have duplicate timestamps (repaired). Optical marker files have `33–55%` occlusion NaNs (unusable for continuous angle GT). | **Keep separate for Track 2A/2B quality.** Can combine `EFE0` and `EAH0` with `Optotrak vs DOT` + `JU-IMU` in Track 5A-2 via orientation-invariant features. |
| **3** | **`ULTra-MoCap`** | **KEEP FOR SPECIFIC TASKS ONLY (`Separate Multi-Speed Angle & Speed-Generalization Track`)** | • **Elbow Flex/Ext Angle & ROM:** `elbow_flex_r` (`deg`)<br>• **Forearm Pro/Sup Angle & ROM:** `pro_sup_r` (`deg`)<br>• **Cross-Speed Exercise Classif.:** 5 exercises across `Slow, Normal, Fast, VeryFast` | 6 × 6-DOF Delsys Trigno IMUs (`IMU 4` proximal upper arm + `IMU 1` distal forearm/hand); Acc (`mm/s²` $\rightarrow$ `m/s²`) + Gyr (`deg/s` $\rightarrow$ `rad/s`) at **`2,000 Hz`** decimated $20\times$ to **`100.0 Hz`** | 100 Hz 60-marker Vicon optical motion capture + OpenSim Inverse Kinematics (`Processed IK`) | `13` healthy subjects (`P01..P13`, `208` paired IMU+IK trials; `9` `overheadreach_45` trials lack IK). Sensor columns are numbered `1..6` without anatomical header strings; `veryfast` speed only exists for `Arm Swing`. | **Do NOT combine** with `Optotrak vs DOT` or `PHYTMO` because `IMU 1` is distal forearm/hand and OpenSim `pro_sup_r` uses a different neutral offset (`149°–175°`). Keep as a dedicated benchmark track. |
| **4** | **`JU-IMU`** | **KEEP FOR SPECIFIC TASKS ONLY (`Exercise Classif. & Stroke vs. ND Impairment Analysis`)** | • **Rehab Exercise Classif.:** 14 upper-limb ROM/ADL classes (`ElbFlex, SupPro, ...`)<br>• **Stroke Movement Analysis:** Binary cohort (`0` = `ND`, `1` = `Stroke`) + cohort-stratified exercise analysis | Right Upper Arm (`sensor4`) + Right Wrist/Forearm (`sensor1`) (plus Left `sensor5`+`sensor2`); 12-ch 6-DOF Acc (`m/s²`) + Gyr (`rad/s`) at **`80.0 Hz`** | Protocol activity labels & clinical cohort diagnosis (`27 ND`, `14 Stroke`) | `41` subjects (`27 ND`, `14 Stroke`; `574` trials). **No continuous ground-truth joint angles exist.** Severe stroke subjects (e.g., `Stroke6`) exhibit strong compensatory synergies that degrade standard 14-class recognition (`35.0%` vs `97.6%` in `ND`). | **Classification only:** Can combine 7 shared classes (`ElbFlex, SupPro, ShFlex90, ShAbd90, ShRot, WrstFlex/Ext, PickCup`) with `Optotrak vs DOT` at `100 Hz` using orientation-invariant features. |
| **5** | **`ULF_in_ADL`** | **KEEP FOR SPECIFIC TASKS ONLY (`Stroke Angle Regression Only — Requires File-Length Filtering`; EXCLUDE from 3-Class Limb Classification)** | • **Stroke Elbow Flex/Ext & Pro/Sup Angle Regression:** `jRight/LeftElbow[:, [2, 1]]` (`deg`)<br>• *(Exclude from 3-class `Healthy/Non-Paretic/Paretic` window classification)* | Bilateral Xsens MVN `Right/LeftUpperArm` + `Right/LeftForeArm`; 12-ch segment Acc (`m/s²`) + Gyr (`rad/s`) at **`60.0 Hz`** | 17-sensor full-body Xsens MVN kinematic model (`jRight/LeftElbow` 3D `jointAngle` in `deg`) + `ParticipantCharacteristics.xlsx` (`FMMA-UE` 21–61) | `25` subjects (`20 Stroke P02..P30`, `5 Healthy H01..H05`, `4,489` valid `.mvnx`). **Critical Stage 9 Audit Finding:** `511` `.mvnx` files (including `13` `299.85 MB` / `40,517`-frame files in test subject `P28`) are unsegmented full-session recordings that dominated `99.07%` of test windows in Tracks 3D/4B. | **Do NOT combine** with optical MoCap angle datasets because angles come from Xsens biomechanical solver (`60 Hz`) rather than optical markers, and unscripted ADLs have weak pronation/supination observability ($R^2 < 0$). |
| **6** | **`StrokeRehab`** | **EXCLUDE FROM FINAL CORE MODEL (`Keep as Secondary Benchmark Reference Only`)** | • **Secondary ADL Angle Regression Reference:** `ElbowFlexionRT/LTdeg`, `WristSupinationRT/LTdeg` (`deg`) across 9 ADLs | Bilateral Noraxon `RT/LTUpperarm` + `RT/LTForearm` at **`100.0 Hz`**; linear Acc (`mG` $\rightarrow$ `m/s²`) + differentiated Euler angles (`course, pitch, roll` $\rightarrow$ `rad/s`) | 9-sensor Noraxon MyoMotion biomechanical model (`100 Hz`) | `20` **healthy control** subjects (`C4..C32`, `1,404` Schema-A CSVs; **stroke patient branch is missing on disk**). Raw gyroscope channels are absent (`RotX..Z` are unit quaternion vector parts; angular velocity had to be differentiated from Euler angles). | **Do NOT combine** with raw MEMS gyroscope datasets (`Optotrak vs DOT`, `PHYTMO`, `JU-IMU`), and **cannot** be used for stroke impairment analysis because only healthy controls exist locally. |
| **7** | **`VIDIMU`** | **EXCLUDE FROM FINAL 6-DOF RAW IMU MODEL (`Keep as Quaternion-Input Reference Track 3E Only`)** | • **Quaternion-to-Angle Regression Reference:** `elbow_flex_r`, `pro_sup_r` (`deg`) across 9 upper-limb tasks (`A05..A13`) | Right Upper Arm (`humerus_r_imu`) + Right Forearm (`radius_r_imu`) at **`50.0 Hz`**; **8-channel unit quaternions only** (`q_w, q_x, q_y, q_z`) | 50 Hz OpenSim Inverse Kinematics (`.mot` files) | `16` IMU subjects (`S40..S57`, `144` upper-limb trials; `13` cross-contaminated `S48` `.mot` files and `64` lower-limb `A01..A04` trials excluded). **Raw 3D accelerometer and gyroscope signals are absent** in `.sto`/`.raw`. | **Cannot be combined** with 12-channel 6-DOF `[Acc, Gyr]` datasets because `.sto` files provide only 4D orientations (`8` channels total). |
| **8** | **`REHAB` (`REHAB-120`)** | **EXCLUDE FROM SUBJECT-INDEPENDENT BENCHMARKS (`Session-Split Exploratory Track 4C Only`)** | • **Exploratory Clinical FMA-UE Score Classification:** 3-class item score (`0, 1, 2`) on 16 proximal arm tasks (`000..014, 026`) | Dual 9-DOF IMUs (`_1.npy` and `_2.npy`) on paretic arm at **`50.0 Hz`** (`T = 150` fixed length, 12-ch Acc + Gyr) | Clinical Fugl-Meyer Assessment (`Label.npy`: `(222, 27)` scores `0, 1, 2`) | `222` sessions from `120` stroke patients. **Fatal Split Limitation:** No local mapping links the `222` session folders (`001..222`) to the `120` unique patient IDs, so session splits cannot prevent longitudinal subject leakage. Also has `1` corrupted `.npy` (`014_1.npy`) and `1` duplicate folder (`195`). | **Do NOT combine** with any subject-independent dataset due to unmapped longitudinal patient sessions and fixed 150-sample pre-normalized segments. |

---

## 2. Detailed Dataset-by-Dataset Evaluation (1–8)

### 2.1 `PHYTMO` (`D:\me\project\datasets\raw\PHYTMO`)
1. **Decision:** **KEEP (`PRIMARY` for Correct-vs.-Incorrect Execution Assessment [Track 2A/2B] and 3-Class Rehabilitation Exercise Classification [Track 1C]).**
2. **Exact Tasks & Labels Supported:**
   - **Correct vs. Incorrect Exercise Execution (Primary Task):** Binary `0` (`Correct`) vs `1` (`Incorrect` / compensatory execution) on Elbow Flexion-Extension (`EFE0` vs `EFE1`, Track 2A) and 6-class joint `Exercise × Quality` (`EFE0, EFE1, EAH0, EAH1, SQZ0, SQZ1`, Track 2B).
   - **Rehabilitation Exercise Classification:** 3-class upper-limb rehabilitation exercises (`0: EFE`, `1: EAH`, `2: SQZ`, Track 1C).
   - *Unsuitable for:* Continuous elbow flexion/extension or pronation/supination angle regression (optical marker files have `33–55%` occlusion NaNs) and stroke analysis (healthy volunteers only).
3. **Sensor Placement, Channels, Sampling Rate & Ground-Truth Source:**
   - Bilateral Upper Arm (`Rarm`, `Larm`) and Forearm (`Rforearm`, `Lforearm`) 9-axis NGIMUs (`Acc` in $g \rightarrow \text{m/s}^2$, `Gyr` in $\text{deg/s} \rightarrow \text{rad/s}$), resampled from `100.89 Hz` to uniform **`100.0 Hz`** after timestamp deduplication. Ground truth comes from supervised physical therapy trial protocols.
4. **Subject Count & Data-Quality Limitations:**
   - `30` subjects (`A01..E05`, `342` active 4-sensor upper-limb trials; `20 train / 5 val / 5 test` subjects).
   - `A06, A07, B03` lack `EFE` trials (`27` subjects in Track 2A); `A02` lacks `EAH` trials. `98` CSVs contain `252` duplicate consecutive timestamps (repaired deterministically in [`load_phytmo_trial_12ch`](file:///d:/me/project/src/data/loaders.py#L71-L112)).
5. **Combinability:** Cannot be combined in raw 12-channel coordinates with `Optotrak vs DOT` or `JU-IMU` without orientation-invariant transformation (`compute_orientation_invariant_12ch`), and its execution-quality labels (`EFE0/EFE1`) are unique to `PHYTMO`.
6. **Reason for Decision:** `PHYTMO` is the **only** dataset in the workspace with valid, subject-independent (`N = 27–30` subjects) dual-IMU ground-truth labels for **correct vs. incorrect elbow rehabilitation exercise execution**, achieving `0.8925` Test Macro-F1 (`92.20%` incorrect-execution recall) with `TCN`.

---

### 2.2 `Optotrak vs DOT` (`D:\me\project\datasets\raw\Optotrak vs DOT`)
1. **Decision:** **KEEP (`PRIMARY` Gold-Standard Dataset for Continuous Elbow Flexion-Extension Angle, Forearm Pronation-Supination Angle, ROM Estimation [Track 3A], and 10-Class Exercise Classification [Track 1A]).**
2. **Exact Tasks & Labels Supported:**
   - **Continuous Elbow Flexion-Extension Angle & ROM (`deg`):** `H_Joint_Angles_Aligned_FINAL['j_aligned_eu']['opto'][t]['Elbow'][:, 2]` (`1.4°` to `136.2°`).
   - **Continuous Forearm Pronation-Supination Angle & ROM (`deg`):** `Elbow[:, 1]` (`-62.0°` to `+69.2°`).
   - **Rehabilitation & Functional Exercise Classification:** 10 tasks (`Trials 11..20`: `elbow_flex_ext`, `forearm_pro_sup`, `shoulder_flex`, `shoulder_abd`, `shoulder_int_ext_rot`, `wrist_flex_ext`, `wrist_rad_uln_dev`, `drinking_cup`, `box_off_shelf`, `drawing_circles`).
3. **Sensor Placement, Channels, Sampling Rate & Ground-Truth Source:**
   - Right Upper Arm (`RUA`) + Right Forearm (`RLA`) Xsens DOT IMUs (`Acc` in $\text{m/s}^2$, `Gyr` in $\text{deg/s} \rightarrow \text{rad/s}$) hardware-synchronized at **`120.0 Hz`** with Optotrak Certus 3D optical marker clusters and ISB anatomical bony landmarks.
4. **Subject Count & Data-Quality Limitations:**
   - `16` healthy subjects total: `subject 2` excluded from right-arm tracks due to left-arm sensor placement (`15` subjects / `150` trials in Track 1A); `subject 8` excluded from Track 3A due to missing medial epicondyle (`ME`) landmark digitization (`14` subjects / `140` trials in Track 3A).
   - `69` backup CSVs in `delete`/`Fist attempt` folders and `150` calibration trials (`Trials 1..10`) are excluded from evaluation tracks.
5. **Combinability:** Can be combined with `JU-IMU` (7 shared exercise classes) after resampling to `100.0 Hz` and applying orientation-invariant kinematic features (`Track 5A`). Should **not** be merged with `ULTra-MoCap` or `StrokeRehab` for raw joint-angle regression because of differing anatomical zero-angle definitions (`ISB Optotrak` vs. `OpenSim IK` vs. `Noraxon`).
6. **Reason for Decision:** Provides the highest-fidelity optical motion-capture ground truth (`120 Hz` ISB bony landmarks) paired with exact Upper Arm + Forearm 6-DOF IMUs, yielding `0.9808` Test Macro-F1 on 10-class classification (`1D_CNN`) and `12.24°` Elbow Flexion MAE (`R² = 0.8103, r = 0.9088` across all 10 multi-joint tasks; `6.17°–8.48°` MAE on isolated forearm/elbow tasks) with `GRU`.

---

### 2.3 `JU-IMU` (`D:\me\project\datasets\raw\JU-IMU\ROM`)
1. **Decision:** **KEEP FOR SPECIFIC TASKS ONLY (`PRIMARY` for 14-Class Exercise Classification [Track 1B] and Stroke vs. Non-Disabled Movement Analysis [Track 4A]).**
2. **Exact Tasks & Labels Supported:**
   - **Rehabilitation Exercise Classification:** 14 upper-limb ROM/ADL classes (`ElbFlex, SupPro, FrontSupPro, WrstFlex, WrstExt, FrontWrstFlex, FrontWrstExt, ShFlex90, ShExt, ShAbd90, Scaption90, ShHorAdd, ShRot, PickCup`).
   - **Stroke vs. Non-Disabled Impairment Analysis:** Binary cohort classification (`0` = `ND`, `1` = `Stroke`) and cohort-stratified exercise recognition breakdown.
   - *Unsuitable for:* Continuous joint-angle or ROM regression (no ground-truth angle columns exist) and execution form error detection (no form quality labels exist).
3. **Sensor Placement, Channels, Sampling Rate & Ground-Truth Source:**
   - Bilateral Upper Arm (`sensor4` Right, `sensor5` Left) + Wrist/Forearm (`sensor1` Right, `sensor2` Left) 9-DOF IMUs (`Acc` in $\text{m/s}^2$, `Gyr` in $\text{rad/s}$) at **`80.0 Hz`**. Ground truth is clinical cohort status and instructed activity.
4. **Subject Count & Data-Quality Limitations:**
   - `41` subjects (`27 ND`, `14 Stroke`, `574` trials; `29 train / 6 val / 6 test`), `0` NaNs.
   - Significant kinematic divergence between `ND` (`97.56%` test accuracy in Track 1B) and severe `Stroke` (`34.98%` test accuracy; `Stroke13: 78.03%`, `Stroke6: 5.24%`), demonstrating that severe post-stroke motor synergy alters exercise signatures.
5. **Combinability:** Harmonizes cleanly with `Optotrak vs DOT` across 7 shared upper-limb movements (`Track 5A-1`, achieving `0.7392` zero-shot Macro-F1 with `LSTM`).
6. **Reason for Decision:** Largest subject cohort (`41` subjects) with clean 6-DOF Upper Arm + Forearm signals and verified hemiparetic stroke vs. non-disabled labels (`0.8733` Macro-F1 / `0.8990` Balanced Accuracy in Track 4A with `GRU`).

---

### 2.4 `ULF_in_ADL` (`D:\me\project\datasets\raw\ULF_in_ADL`)
1. **Decision:** **KEEP FOR SPECIFIC TASK ONLY (`Secondary Stroke ADL Angle Regression [Track 3D] — ONLY After Filtering Out Unsegmented >1,500-Frame Files`; EXCLUDE from 3-Class Limb Status Classification [Track 4B]).**
2. **Exact Tasks & Labels Supported:**
   - **Stroke & Healthy Continuous Elbow Angle Regression (`Track 3D`):** `jRightElbow` / `jLeftElbow` flexion-extension (`col 2`, `deg`) and pronation-supination (`col 1`, `deg`).
   - **EXCLUDE `Track 4B` (`3-Class Healthy vs Non-Paretic vs Paretic Limb Classification`):** Unscripted 2-second ADL windows do not reliably separate unimpaired `Healthy` limbs from unimpaired `Stroke Non-Paretic` limbs (`Macro-F1 = 0.3091–0.3620`).
3. **Sensor Placement, Channels, Sampling Rate & Ground-Truth Source:**
   - Bilateral Xsens MVN `Right/LeftUpperArm` (segments `9, 13`) + `Right/LeftForeArm` (segments `10, 14`) 6-DOF kinematics (`acceleration` in $\text{m/s}^2$, `angularVelocity` in $\text{rad/s}$) and 3D elbow joint angles (`jRight/LeftElbow` in `deg`) at **`60.0 Hz`**, plus clinical `FMMA-UE` scores (`21–61`) in `ParticipantCharacteristics.xlsx`.
4. **Subject Count & Critical Data-Quality Anomaly Discovered in Stage 9 Audit:**
   - `25` subjects (`20 Stroke P02..P30`, `5 Healthy H01..H05`, `4,489` valid `.mvnx` files).
   - **Critical Anomaly:** Our Stage 9 forensic audit (`experiments/reports/STAGE9_AUDIT_REPORT.md`) discovered that **`511` of the `4,492` `.mvnx` files exceed `5 MB`**, and specifically in test subject `P28`, **13 files (`P28_T01_L1.mvnx` through `P28_T07_L1.mvnx`) are `299.85 MB` (`40,517` frames = `11.25 minutes` each) unsegmented full-session recordings** rather than trimmed 3-second (`~180`-frame) task clips! Consequently, `P28` accounted for `8,766` of the `8,848` test windows (`99.07%`) in Stage 8 Tracks 3D and 4B.
5. **Combinability:** Must **not** be combined with `Optotrak vs DOT` or `ULTra-MoCap` because joint angles are Xsens MVN filter estimates at `60 Hz` during unscripted daily tasks rather than optical motion capture.
6. **Reason for Decision:** Valuable as an auxiliary testbed for evaluating elbow flexion angle tracking in chronic stroke patients (`Subject-Equalized Elbow Flexion MAE = 18.74°` with `TCN`), provided that unsegmented `.mvnx` files (`> 1,500` frames / `> 5 MB`) are filtered out prior to windowing.

---

### 2.5 `StrokeRehab` (`D:\me\project\datasets\raw\StrokeRehab`)
1. **Decision:** **EXCLUDE FROM FINAL CORE MODEL (`Retain Only as Secondary Angle-Regression Reference Track 3B`).**
2. **Exact Tasks & Labels Supported:**
   - Continuous `ElbowFlexionRT/LTdeg` and `WristSupinationRT/LTdeg` (`deg`) across 9 ADLs (`brushing, combing, deodorant, drinking, feeding, glasses, Pageturning, RTT, shelf`).
   - **Does NOT support Stroke Movement Analysis locally** because only the `healthy controls` directory (`20` healthy subjects `C4..C32`) is present on disk (the stroke patients branch is absent).
3. **Sensor Placement, Channels, Sampling Rate & Ground-Truth Source:**
   - Bilateral Noraxon MyoMotion `RT/LTUpperarm` + `RT/LTForearm` at **`100.0 Hz`**.
   - **Critical Signal Limitation:** Raw MEMS gyroscope channels are **not** stored in the CSVs (`RotX/Y/Z` are unit quaternion vector components with norm $\le 1$, so angular velocity in `rad/s` had to be approximated by finite-differencing `course, pitch, roll` Euler angles).
4. **Subject Count & Data-Quality Limitations:**
   - `20` healthy subjects (`1,404` valid 106-column Schema-A CSVs; `2` NaN files in `C5` and `20` Schema B/C files in `C12/C26` excluded).
5. **Combinability:** Cannot be combined with true raw-gyroscope datasets (`Optotrak vs DOT`, `PHYTMO`, `JU-IMU`) because its angular rate is differentiated from onboard Kalman/complementary Euler angles.
6. **Reason for Decision:** While `TCN` achieved `13.52°` Elbow Flexion MAE (`11.50°` ROM MAE) on Track 3B, the absence of raw gyroscope signals and the absence of the stroke cohort on disk make `StrokeRehab` inferior to `Optotrak vs DOT` and `ULTra-MoCap` for training the final dual-IMU hardware model.

---

### 2.6 `ULTra-MoCap` (`D:\me\project\datasets\raw\ULTra-MoCap`)
1. **Decision:** **KEEP FOR SPECIFIC TASKS ONLY (`Dedicated Multi-Speed Continuous Angle & ROM Regression Track [Track 3C]` and `Speed-Generalization Reference [Track 5B]`).**
2. **Exact Tasks & Labels Supported:**
   - **Continuous Elbow Flexion & Pronation-Supination Angle + ROM Regression (`Track 3C`):** `elbow_flex_r` and `pro_sup_r` (`deg`) at `100.0 Hz`.
   - **Cross-Speed Generalization (`Track 5B`):** 5 upper-limb exercises across `Slow, Normal, Fast, VeryFast` speeds and `45°, 90°, 180°, Max` ROM elevations.
3. **Sensor Placement, Channels, Sampling Rate & Ground-Truth Source:**
   - 6 × 6-DOF Delsys Trigno IMUs at **`2,000 Hz`** (`ACC` in $\text{mm/s}^2 \rightarrow \text{m/s}^2$, `GYRO` in $\text{deg/s} \rightarrow \text{rad/s}$), anti-alias filtered (4th-order Butterworth, `40 Hz` cutoff) and decimated $20\times$ to **`100.0 Hz`** to match 60-marker Vicon + OpenSim Inverse Kinematics (`Processed IK`). Kinematic variance analysis confirmed `IMU 4` is proximal upper arm and `IMU 1` is distal forearm/hand.
4. **Subject Count & Data-Quality Limitations:**
   - `13` healthy subjects (`P01..P13`, `208` paired IMU+IK trials; `9 train / 2 val / 2 test` subjects). `9` `overheadreach_45` trials lack IK files (excluded); `8` `overheadreach_90` files have capitalization mismatches (repaired).
   - **Stage 9 Audit Caveat on Track 5B:** In Stage 8, `Track 5B` split trials by speed (`Normal` $\rightarrow$ `Slow, Fast, VeryFast`) across the same 13 subjects rather than holding out both unseen subjects and unseen speeds simultaneously, and `veryfast` was only recorded for `Arm Swing`.
5. **Combinability:** Must **not** be merged with `Optotrak vs DOT` because `IMU 1..6` lack explicit anatomical header strings and OpenSim's `pro_sup_r` coordinate uses a shifted neutral offset (`149°–175°`).
6. **Reason for Decision:** Provides high-precision 100 Hz OpenSim IK ground truth across varying movement speeds, where **`TCN` achieves `10.07°` Overall MAE (`9.96°` Elbow Flexion MAE, `10.19°` Pronation-Supination MAE, `8.31°` Elbow Flexion ROM MAE, $R^2 = 0.8873, r = 0.9497$)** on unseen test subjects (`P06, P12`).

---

### 2.7 `VIDIMU` (`D:\me\project\datasets\raw\VIDIMU\dataset`)
1. **Decision:** **EXCLUDE FROM FINAL 6-DOF RAW IMU MODEL (`Retain Only as Quaternion-Input Reference Track 3E`).**
2. **Exact Tasks & Labels Supported:**
   - Continuous `elbow_flex_r` and `pro_sup_r` (`deg`) across 9 upper-limb activities (`A05..A13`).
3. **Sensor Placement, Channels, Sampling Rate & Ground-Truth Source:**
   - `humerus_r_imu` + `radius_r_imu` in `.sto` at **`50.0 Hz`** paired with OpenSim IK `.mot` angles (`50.0 Hz`).
   - **Critical Limitation:** `.sto` and `.raw` files store **only 4D unit quaternions (`8` channels total)**—raw 3-axis accelerometer and 3-axis gyroscope channels are completely absent.
4. **Subject Count & Data-Quality Limitations:**
   - `16` IMU subjects (`S40..S57`, `144` valid upper-limb trials; `37` video-only subjects `S01..S39`, `64` lower-limb trials `A01..A04`, and `13` cross-subject-contaminated `S48_A01..13.mot` files excluded).
5. **Combinability:** Incompatible with 12-channel `[Acc, Gyr]` models.
6. **Reason for Decision:** Useful only if the future ESP32 + IMU firmware streams onboard DMP quaternions (`8` channels), where `TCN` achieves `13.98°` Overall MAE and **`8.73°` Window ROM MAE (`8.06°` Elbow Flexion ROM MAE, $R^2 = 0.7725$)**. Excluded from the primary 12-channel raw `[Acc, Gyr]` pipeline.

---

### 2.8 `REHAB` (`REHAB-120`, `D:\me\project\datasets\raw\REHAB`)
1. **Decision:** **EXCLUDE FROM FINAL SUBJECT-INDEPENDENT MODEL (`Retain Only as Exploratory Session-Split Reference Track 4C`).**
2. **Exact Tasks & Labels Supported:**
   - 3-class clinical Fugl-Meyer Assessment (`FMA-UE`) item score (`0` = cannot perform, `1` = partial, `2` = full) across 16 proximal arm tasks (`000..014, 026`).
3. **Sensor Placement, Channels, Sampling Rate & Ground-Truth Source:**
   - Dual upper-limb IMUs (`_1.npy` and `_2.npy`) pre-interpolated to fixed $T = 150$ samples (`3.0 s` at **`50.0 Hz`**) + `Label.npy` `(222, 27)`.
4. **Subject Count & Fatal Limitation:**
   - `222` sessions recorded from `120` stroke patients, **but no local metadata maps the `222` session folders (`001..222`) to the `120` patient IDs** (and `Rehab_exercise` pools `4,616` trials without subject IDs). Thus, **strict subject-independent splitting is impossible to guarantee locally**.
5. **Combinability:** Cannot be combined with any other dataset.
6. **Reason for Decision:** While `GRU` achieved `0.8051` Test Macro-F1 (`82.50%` accuracy) on the session-disjoint split (`Track 4C`), the unresolvable risk of longitudinal patient overlap across sessions `001..222` disqualifies `REHAB-120` from serving as a primary subject-independent benchmark.
