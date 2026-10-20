# Multi-Dataset Benchmark Plan (`docs/MULTI_DATASET_BENCHMARK_PLAN.md`)

**Project:** AI-Based Elbow Rehabilitation Assistance System (Post-Olecranon ORIF Recovery)  
**Models Under Comparison:** `TCN`, `1D CNN`, `LSTM`, `GRU`  
**Canonical Sensor Representation:** 12-Channel Dual 6-Axis IMU (`Upper Arm`: Acc XYZ + Gyr XYZ; `Forearm`: Acc XYZ + Gyr XYZ)

---

## 1. Overview of the 5 Benchmark Tracks

Because the 8 suitable public datasets differ in sensor coordinate frames, sampling rates (`50 Hz` to `2,000 Hz`), activity taxonomies, and target types (discrete classes vs. continuous joint angles in degrees vs. clinical impairment scores), **datasets are never blindly concatenated**. Instead, we organize the research into **five scientifically rigorous benchmark tracks**:

| Track | Track Name | Research Objective | Eligible Datasets & Sub-Benchmarks | Input Representation | Target Labels |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Track 1** | **Rehabilitation Exercise Classification** | Classify prescribed upper-limb ROM and functional rehab exercises | • **1A:** `Optotrak vs DOT` (15 right-arm subj, 10 classes)<br>• **1B:** `JU-IMU` (41 subj, 14 classes)<br>• **1C:** `PHYTMO` (30 subj, 3 upper-limb classes) | **12-ch 6-DOF Dual-IMU** (`UA` + `FA` Acc $\text{m/s}^2$, Gyr $\text{rad/s}$) | Discrete Exercise Class (`10`, `14`, or `3` classes) |
| **Track 2** | **Execution-Quality Classification** | Detect correct vs. compensatory/incorrect exercise execution | • **2A:** `PHYTMO` Elbow Flexion-Extension (`EFE0` vs `EFE1`, 27 subj)<br>• **2B:** `PHYTMO` All Upper-Limb (`EFE, EAH, SQZ` × `0/1`, 30 subj) | **12-ch 6-DOF Dual-IMU** (`Rarm` + `Rforearm` at `100 Hz`) | • **2A:** Binary Quality (`0`=Correct, `1`=Incorrect)<br>• **2B:** 6-Class Joint (`Exercise × Quality`) & Dual-Head |
| **Track 3** | **Joint-Angle & ROM Regression** | Estimate continuous elbow flexion/extension $\theta_{\text{flex}}(t)$, forearm pronation/supination $\theta_{\text{pro/sup}}(t)$ ($\text{deg}$), and ROM | • **3A (Primary):** `Optotrak vs DOT` (14 right-arm subj w/ Optotrak GT)<br>• **3B (Separate):** `StrokeRehab` (20 subj, 9 ADLs, Noraxon)<br>• **3C (Separate):** `ULTra-MoCap` (13 subj, OpenSim IK)<br>• **3D (Primary Stroke):** `ULF_in_ADL` (25 subj, Xsens 60 Hz)<br>• **3E (Separate Quat):** `VIDIMU` (16 subj, `A05–A13`, 50 Hz) | • **3A, 3D:** 12-ch 6-DOF Dual-IMU<br>• **3B:** 12-ch Acc + Quat-derived $\boldsymbol{\omega}$<br>• **3C:** 12-ch Distal `IMU1` + Proximal `IMU4`<br>• **3E:** 8-ch Dual-IMU Quaternions | Continuous $\theta_{\text{flex}}(t) \in \mathbb{R}$ and $\theta_{\text{pro/sup}}(t) \in \mathbb{R}$ in **degrees (`deg`)**, plus Window/Trial $\text{ROM}$ (`deg`) |
| **Track 4** | **Impaired & Stroke Movement Analysis** | Evaluate model performance on hemiparetic stroke patients and clinical `FMA-UE` severity | • **4A:** `JU-IMU` (`27 ND` vs `14 Stroke` cohort & within-Stroke 14-class)<br>• **4B:** `ULF_in_ADL` (`Paretic` vs `Non-Paretic` vs `Healthy` limb)<br>• **4C:** `REHAB` (`222` sessions, 16 proximal FMA-UE items `0, 1, 2`) | **12-ch 6-DOF Dual-IMU** | • **4A:** Binary Cohort (`ND` vs `Stroke`) & 14-class in Stroke<br>• **4B:** 3-Class Limb Status (`Healthy, Non-Paretic, Paretic`)<br>• **4C:** 3-Class FMA-UE (`0, 1, 2`) |
| **Track 5** | **Cross-Dataset & Cross-Condition Generalization** | Test zero-shot & transfer generalization across datasets, sensor coordinate frames, speeds, and ROMs | • **5A:** Cross-Dataset Transfer (`Optotrak vs DOT` $\leftrightarrow$ `JU-IMU` 7 shared classes; + `PHYTMO` 2 shared classes)<br>• **5B:** `ULTra-MoCap` Cross-Speed (`Normal` $\rightarrow$ `Slow/Fast/VeryFast`) & Cross-ROM (`90°/180°` $\rightarrow$ `45°/Max`) | • **5A:** Gravity-Aligned + Norm-Augmented Dual-IMU at `100 Hz`<br>• **5B:** 12-ch 6-DOF at `100 Hz` | • **5A:** 7-Class / 2-Class Harmonized Taxonomy<br>• **5B:** 5-Class Exercise & Continuous Elbow Angle (`deg`) |

---

## 2. Activity Taxonomy Mapping Rules (Track 1 & Track 5A)

Original dataset labels are **always preserved** in dataset-specific tracks (Tracks 1A, 1B, 1C). For **Track 5A (Cross-Dataset Generalization)**, we map **only genuinely biomechanically equivalent movements** and exclude non-overlapping classes from cross-dataset evaluation:

### Verified 7-Class Shared Upper-Limb Rehabilitation Taxonomy (`Optotrak vs DOT` $\leftrightarrow$ `JU-IMU`, plus 2-Class `PHYTMO`)

| Shared Class ID | Canonical Movement Name | `Optotrak vs DOT` Original Label | `JU-IMU` Original Label | `PHYTMO` (`inertial\upper`) Original Label | Biomechanical Equivalence Justification |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **`0`** | **`elbow_flex_ext`** | `Task 11` (`Elbow flexion/extension`) | `ElbFlex` | `EFE0` (Correct `Elbow Flexion-Extension`) | Isolated uniaxial elbow flexion and extension in sagittal plane |
| **`1`** | **`forearm_pro_sup`** | `Task 12` (`Elbow pronation/supination`) | `SupPro` | *Not present* | Forearm axial rotation (pronation/supination) |
| **`2`** | **`shoulder_flex`** | `Task 13` (`Shoulder flexion`) | `ShFlex90` | `EAH0` (`Extension of Arms Overhead`, 2-class subset only) | Sagittal-plane glenohumeral elevation / flexion |
| **`3`** | **`shoulder_abd`** | `Task 14` (`Shoulder abduction`) | `ShAbd90` | *Not present* | Coronal-plane glenohumeral abduction |
| **`4`** | **`shoulder_rot`** | `Task 15` (`Shoulder int/ext rotation`) | `ShRot` | *Not present* | Humeral internal/external axial rotation |
| **`5`** | **`wrist_flex_ext`** | `Task 16` (`Wrist flexion/extension`) | `WrstFlex` + `WrstExt` | *Not present* | Sagittal wrist flexion and extension |
| **`6`** | **`drinking_cup_adl`** | `Task 18` (`Drinking from a cup`) | `PickCup` | *Not present* | Functional reach-to-grasp-and-drink cup ADL |

*Non-equivalent classes kept strictly in dataset-specific tracks:*
- `Optotrak vs DOT`: `Task 17` (`Wrist radial/ulnar deviation`), `Task 19` (`Box off shelf`), `Task 20` (`Drawing circles`).
- `JU-IMU`: `FrontSupPro`, `FrontWrstFlex`, `FrontWrstExt`, `ShExt`, `Scaption90`, `ShHorAdd`.
- `PHYTMO`: `SQZ0`/`SQZ1` (`Squeezing`) and compensatory trials (`EFE1`, `EAH1`).

---

## 3. Responsible Sensor Input Standardization (Stage 3)

### 3.1 Canonical 12-Channel 6-DOF Dual-IMU Representation
For all primary dual-IMU datasets (`PHYTMO`, `Optotrak vs DOT`, `JU-IMU`, `ULF_in_ADL`), each time step $t$ is represented as a 12-dimensional vector in standardized SI units:

$$\mathbf{x}(t) = \big[\underbrace{a_{x}^{\text{UA}}, a_{y}^{\text{UA}}, a_{z}^{\text{UA}}}_{\text{Upper Arm Acc }(\text{m/s}^2)}, \underbrace{\omega_{x}^{\text{UA}}, \omega_{y}^{\text{UA}}, \omega_{z}^{\text{UA}}}_{\text{Upper Arm Gyr }(\text{rad/s})}, \underbrace{a_{x}^{\text{FA}}, a_{y}^{\text{FA}}, a_{z}^{\text{FA}}}_{\text{Forearm Acc }(\text{m/s}^2)}, \underbrace{\omega_{x}^{\text{FA}}, \omega_{y}^{\text{FA}}, \omega_{z}^{\text{FA}}}_{\text{Forearm Gyr }(\text{rad/s})}\big]^{\top} \in \mathbb{R}^{12}$$

### 3.2 Dataset-Specific Unit Conversions & Channel Mappings
| Dataset | Upper Arm Sensor | Forearm Sensor | Raw Acc Unit $\rightarrow$ $\text{m/s}^2$ | Raw Gyr Unit $\rightarrow$ $\text{rad/s}$ | Angle Target Unit | Native Rate $\rightarrow$ Target Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`PHYTMO`** | `Rarm` (`Larm` opt.) | `Rforearm` (`Lforearm` opt.) | $g \times 9.80665$ | $\text{deg/s} \times \frac{\pi}{180}$ | N/A | $100.89\text{ Hz} \rightarrow \mathbf{100.0\text{ Hz}}$ (dedup + linear interp) |
| **`Optotrak vs DOT`** | `RUA` | `RLA` | $\text{m/s}^2 \times 1.0$ | $\text{deg/s} \times \frac{\pi}{180}$ | **Degrees (`deg`)** (`Elbow[:,2]` flex, `Elbow[:,1]` pro/sup) | $\mathbf{120.0\text{ Hz}}$ native (or $100.0\text{ Hz}$ in Track 5A) |
| **`JU-IMU`** | `sensor4` (R) / `sensor5` (L) | `sensor1` (R) / `sensor2` (L) | $\text{m/s}^2 \times 1.0$ | $\text{rad/s} \times 1.0$ | N/A | $\mathbf{80.0\text{ Hz}}$ native (or $100.0\text{ Hz}$ in Track 5A) |
| **`ULF_in_ADL`** | `RightUpperArm` / `LeftUpperArm` | `RightForeArm` / `LeftForeArm` | $\text{m/s}^2 \times 1.0$ | $\text{rad/s} \times 1.0$ | **Degrees (`deg`)** (`jRight/LeftElbow` ZXY) | $\mathbf{60.0\text{ Hz}}$ native |
| **`StrokeRehab`** *(Track 3B)* | `RTUpperarm` / `LTUpperarm` | `RTForearm` / `LTForearm` | $\text{mG} \times 0.00980665$ | Quat/Euler finite diff ($\text{rad/s}$) | **Degrees (`deg`)** (`ElbowFlexion`, `WristSupination`) | $\mathbf{100.0\text{ Hz}}$ native |
| **`ULTra-MoCap`** *(Track 3C/5B)* | `IMU 4` (`ACCX4..GYROZ4`) | `IMU 1` (`ACCX1..GYROZ1`) | $\text{mG} \times 0.00980665$ | $\text{deg/s} \times \frac{\pi}{180}$ | **Degrees (`deg`)** (`elbow_flex_r`, `pro_sup_r`) | $2000\text{ Hz} \rightarrow \mathbf{100.0\text{ Hz}}$ (40 Hz Butterworth LP + $20\times$ decimation) |
| **`VIDIMU`** *(Track 3E)* | `humerus_r_imu` | `radius_r_imu` | Quat $(q_w, q_x, q_y, q_z)$ | Quat $(q_w, q_x, q_y, q_z)$ | **Degrees (`deg`)** (`elbow_flex_r`, `pro_sup_r`) | $\mathbf{50.0\text{ Hz}}$ native (`.sto` & `.mot`) |

### 3.3 Consistent Windowing & Train-Only Normalization
- **Window Duration:** **$2.0\text{ seconds}$** across all continuous time-series benchmarks ($W = 200$ at $100\text{ Hz}$, $W = 240$ at $120\text{ Hz}$, $W = 160$ at $80\text{ Hz}$, $W = 120$ at $60\text{ Hz}$, $W = 100$ at $50\text{ Hz}$) with **$0.5\text{ second}$ stride** ($75\%$ overlap) strictly inside single trials after subject partitioning.
- **Normalization:** Channel-wise mean $\boldsymbol{\mu}_{\text{train}} \in \mathbb{R}^{12}$ and standard deviation $\boldsymbol{\sigma}_{\text{train}} \in \mathbb{R}^{12}$ are fitted **strictly on the `train` partition** of each benchmark track. Angle targets ($\theta_{\text{flex}}, \theta_{\text{pro/sup}}$) are kept in **physical degrees (`deg`)** for interpretable evaluation metrics ($\text{MAE}^\circ, \text{RMSE}^\circ$).

---

## 4. Subject-Independent Partition Manifests & Leakage Prevention (Stage 4)

Every primary and secondary dataset with subject IDs uses a **fixed, deterministic, subject-independent `train` / `val` / `test` split** (generated with `seed = 42` and verified for complete class coverage in every split):

1. **`PHYTMO` (`30` subjects: `20 train / 5 val / 5 test`):**
   - `val` (5 subjects, 1 per age group `A–E`, all having complete 6/6 classes): `['A01', 'B01', 'C01', 'D01', 'E01']`
   - `test` (5 subjects, 1 per age group `A–E`, all having complete 6/6 classes): `['A03', 'B02', 'C02', 'D02', 'E02']`
   - `train` (20 remaining subjects): `['A02', 'A04', 'A05', 'A06', 'A07', 'A08', 'A09', 'A10', 'B03', 'B04', 'B05', 'C03', 'C04', 'C05', 'D03', 'D04', 'D05', 'E03', 'E04', 'E05']` *(with `A06, A07, B03` excluded when running the `EFE`-only binary Track 2A since they have no `EFE` files).*
2. **`Optotrak vs DOT` (`15` right-arm subjects in Track 1A; `14` with Optotrak GT in Track 3A; `subject 2` excluded due to left-arm mounting):**
   - `val` (3 subjects): `['subject 3', 'subject 9', 'subject 13']`
   - `test` (3 subjects): `['subject 5', 'subject 11', 'subject 15']`
   - `train` (9 subjects for Track 1A; 8 subjects for Track 3A excluding `subject 8`): `['subject 1', 'subject 4', 'subject 6', 'subject 7', ('subject 8' in Track 1A), 'subject 10', 'subject 12', 'subject 14', 'subject 16']`
3. **`JU-IMU` (`41` subjects: `29 train [19 ND, 10 Stroke] / 6 val [4 ND, 2 Stroke] / 6 test [4 ND, 2 Stroke]`):**
   - `val` (6 subjects): `['ND6', 'ND15', 'ND22', 'ND29', 'Stroke4', 'Stroke10']`
   - `test` (6 subjects): `['ND8', 'ND18', 'ND25', 'ND31', 'Stroke6', 'Stroke13']`
   - `train` (29 remaining subjects): `19 ND` + `10 Stroke`.
4. **`StrokeRehab` (`20` subjects: `14 train / 3 val / 3 test`):**
   - `val` (3 subjects): `['C9', 'C20', 'C28']`
   - `test` (3 subjects): `['C11', 'C23', 'C30']`
   - `train` (14 remaining subjects): `['C4', 'C5', 'C7', 'C12', 'C15', 'C19', 'C22', 'C24', 'C25', 'C26', 'C27', 'C29', 'C31', 'C32']`
5. **`ULF_in_ADL` (`25` subjects: `17 train [14 Stroke, 3 Healthy] / 4 val [3 Stroke, 1 Healthy] / 4 test [3 Stroke, 1 Healthy]`):**
   - `val` (4 subjects): `['H02', 'P05', 'P16', 'P24']`
   - `test` (4 subjects): `['H04', 'P08', 'P19', 'P28']`
   - `train` (17 remaining subjects): `['H01', 'H03', 'H05', 'P02', 'P03', 'P04', 'P06', 'P09', 'P10', 'P12', 'P17', 'P20', 'P21', 'P22', 'P25', 'P29', 'P30']`
6. **`ULTra-MoCap` (`13` subjects: `9 train / 2 val / 2 test`):**
   - `val` (2 subjects): `['P03', 'P09']`
   - `test` (2 subjects): `['P06', 'P12']`
   - `train` (9 remaining subjects): `['P01', 'P02', 'P04', 'P05', 'P07', 'P08', 'P10', 'P11', 'P13']`
7. **`VIDIMU` (`16` IMU subjects `S40–S57`: `10 train / 3 val / 3 test`):**
   - `val` (3 subjects): `['S42', 'S49', 'S54']`
   - `test` (3 subjects): `['S46', 'S51', 'S56']`
   - `train` (10 remaining subjects): `['S40', 'S41', 'S44', 'S47', 'S48', 'S50', 'S52', 'S53', 'S55', 'S57']`

---

## 5. Multi-Task Learning vs. Task-Specific Models (Stage 5)

### 5.1 Scientific Assessment
- **Primary Strategy — Task-Specific Models First:**
  Classification (which benefits from temporal invariance and global window pooling) and continuous sample-by-sample joint-angle regression (which requires preserving phase, sign, and fine temporal resolution at every timestep $t = 1 \dots W$) have distinct inductive requirements. Therefore, **each benchmark track (Tracks 1–5) first evaluates dedicated task-specific `TCN`, `1D CNN`, `LSTM`, and `GRU` models** so architectural strengths for classification vs. angle regression are never confounded.
- **Secondary Multi-Task Investigation (Where Verified Simultaneous Labels Exist):**
  Two datasets genuinely provide simultaneous multi-task ground truth on every window without inventing labels:
  1. **`Optotrak vs DOT` (Joint Exercise Classification + Dual-DOF Elbow Angle Regression):** Every window in `Tasks 11–20` has both a verified exercise label $y_{\text{cls}} \in \{0..9\}$ and continuous ground-truth angles $\mathbf{Y}_{\text{ang}} = [\boldsymbol{\theta}_{\text{flex}}, \boldsymbol{\theta}_{\text{pro/sup}}] \in \mathbb{R}^{W \times 2}$.
  2. **`PHYTMO` (Joint Exercise Type + Execution Quality Classification):** Every window in `inertial\upper` has both an exercise class $y_{\text{ex}} \in \{\text{EFE}, \text{EAH}, \text{SQZ}\}$ and an execution quality label $y_{\text{qual}} \in \{0, 1\}$.
- **Masked Multi-Task Loss Formulation (When Multi-Task Head Is Enabled):**
  For a sample $i$ with availability indicators $m_{i,\text{cls}}, m_{i,\text{qual}}, m_{i,\text{ang}} \in \{0, 1\}$:
  $$\mathcal{L}_{\text{MTL}}(i) = m_{i,\text{cls}} \mathcal{L}_{\text{CE}}(\hat{y}_{i,\text{cls}}, y_{i,\text{cls}}) + \lambda_{\text{qual}} m_{i,\text{qual}} \mathcal{L}_{\text{BCE}}(\hat{y}_{i,\text{qual}}, y_{i,\text{qual}}) + \lambda_{\text{ang}} m_{i,\text{ang}} \mathcal{L}_{\text{SmoothL1}}(\hat{\mathbf{Y}}_{i,\text{ang}}, \mathbf{Y}_{i,\text{ang}})$$
  Unavailable targets have $m = 0$ and contribute zero gradient.

---

## 6. Fair Four-Model Comparison Specification (Stage 8)

### 6.1 Architectures (`TCN`, `1D CNN`, `LSTM`, `GRU`)
All four models share an identical input interface `(batch_size, W, C_in)` and task-head interface:
1. **`TCN`:** 4 dilated residual blocks ($d \in \{1, 2, 4, 8\}$, kernel size $k = 5$, 64 channels, causal/same padding, BatchNorm1d, GELU, Dropout $p=0.2$, $1\times 1$ residual projection).
2. **`1D CNN`:** 4 standard non-dilated 1D conv blocks ($d = 1$, kernel sizes $7, 5, 5, 3$, channels $64 \rightarrow 64 \rightarrow 128 \rightarrow 128$, BatchNorm1d, GELU, Dropout $p=0.2$).
3. **`LSTM`:** 2-layer stacked LSTM (hidden size $H = 64$, Dropout $p=0.2$).
4. **`GRU`:** 2-layer stacked GRU (hidden size $H = 64$, Dropout $p=0.2$).

For **sequence-level classification** (Tracks 1, 2, 4, 5A), the sequence representation $\mathbf{H} \in \mathbb{R}^{B \times W \times D}$ is pooled via Global Average + Max Pooling into a linear classifier. For **continuous angle regression** (Track 3), a pointwise linear head projects $\mathbf{H} \in \mathbb{R}^{B \times W \times D} \rightarrow \hat{\mathbf{Y}} \in \mathbb{R}^{B \times W \times 2}$ (predicting $\hat{\theta}_{\text{flex}}(t)$ and $\hat{\theta}_{\text{pro/sup}}(t)$ in degrees at every sample $t$).

### 6.2 Evaluation Metrics
- **Classification Tracks:** Macro-F1 (primary), Balanced Accuracy, Per-Class Precision & Recall, Full Confusion Matrix, and Subject-wise Macro-F1 ($\text{mean} \pm \text{SD}$).
- **Angle & ROM Regression Tracks:** $\text{MAE}$ (`deg`), $\text{RMSE}$ (`deg`), $R^2$, Pearson $r$, Trial/Window $\text{ROM Error}$ ($|\widehat{\text{ROM}} - \text{ROM}_{\text{GT}}|$ in `deg`), broken down by movement type (`Elbow Flex/Ext` vs. `Pronation/Supination` vs. `ADLs`) and by test subject.
- **Deployment & Efficiency Metrics:** Trainable parameter count, serialized checkpoint size (`KB`), CPU single-window inference latency (`ms`, p50/p95/p99), and training wall-clock time (`s`).

---

## 7. Clinical and Deployment Guardrails (Stage 9)

1. Models trained on healthy subjects (`PHYTMO`, `Optotrak vs DOT`, `StrokeRehab`, `ULTra-MoCap`, `VIDIMU`) or post-stroke hemiparetic cohorts (`JU-IMU`, `ULF_in_ADL`, `REHAB`) are **pre-clinical algorithmic benchmarks**, not clinically validated medical devices for post-olecranon-ORIF surgical patients.
2. Predictions from Track 2 (`PHYTMO` Correct vs. Incorrect execution) reflect adherence to physical therapy movement patterns in healthy volunteers and must **never** be presented as a medical safety clearance for postoperative fracture loading.
3. Stage 6 ESP32 hardware integration remains strictly deferred until all benchmark tracks have been trained, evaluated, and reviewed.
