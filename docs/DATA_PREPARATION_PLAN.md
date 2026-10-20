# Data Preparation and Four-Model Comparison Plan (`docs/DATA_PREPARATION_PLAN.md`)

**Status:** Proposed Plan (Stage 2) — Awaiting User Review and Approval  
**Target Models:** `TCN`, `1D CNN`, `LSTM`, `GRU`  
**Hardware Target Alignment:** Dual-IMU configuration (`IMU 1`: Upper Arm, `IMU 2`: Forearm)

---

## 1. Dataset & Subset Selection for the First Experiment

Based on the verified findings in [`docs/DATASET_AUDIT.md`](DATASET_AUDIT.md), no single public dataset simultaneously contains (a) 10+ multi-joint rehab exercises, (b) optical MoCap continuous elbow flexion/extension & pronation/supination angles, **and** (c) correct-vs.-incorrect execution quality labels. However, **three complementary, verified dual-IMU datasets** in `d:\me\project\datasets\raw\` directly match the target **Upper Arm + Forearm** sensor topology and together cover every required task without combining incompatible schemas:

### Recommended Primary Benchmark Options (Select One or Two Ordered Tracks for Stage 3–4)

#### Option A (Recommended Track 1 — Exercise Classification & Correct vs. Incorrect Execution Quality): `PHYTMO` (`inertial\upper`)
- **Why this dataset is ideal:**
  - **Exact Dual-IMU Placement:** Contains 9-axis IMUs on both the **Upper Arm (`Rarm` / `Larm`)** and **Forearm (`Rforearm` / `Lforearm`)** at **100 Hz** (`0` NaNs across all `1,484` upper-limb CSVs).
  - **Clean Subject IDs:** **30 distinct subjects** (`A01–A10`, `B01–B05`, `C01–C05`, `D01–D05`, `E01–E05`), enabling rigorous subject-independent train/validation/test splitting.
  - **Verified Ground-Truth Labels for Both Exercise Type & Execution Quality:**
    - **Task 1A (3-Class Exercise Classification):** `EFE` (Elbow Flexion-Extension), `EAH` (Extension of Arms Overhead), `SQZ` (Squeezing) — `342` exercise trials.
    - **Task 1B (6-Class Joint Exercise + Quality Classification, or Binary Correct vs. Incorrect on `EFE`):** Distinguishes **Correct (`0`) vs. Incorrect/Compensatory (`1`)** execution (`EFE0` vs `EFE1`, `EAH0` vs `EAH1`, `SQZ0` vs `SQZ1`), with a 1:1 balanced class distribution (`171` correct trials vs. `171` incorrect trials).

#### Option B (Recommended Track 2 — Multi-Exercise Rehab Classification + Continuous Elbow ROM Estimation): `Optotrak vs DOT`
- **Why this dataset is ideal:**
  - **Exact Dual-IMU Placement:** Xsens DOT IMUs on **Right Upper Arm (`RUA`)** and **Right Forearm (`RLA`)** at **120 Hz**.
  - **Covers Both Elbow Flexion/Extension and Forearm Pronation/Supination:** Includes dedicated full-ROM **Elbow Flexion/Extension (`Task 11`)** and **Forearm Pronation/Supination (`Task 12`)**, plus 8 additional upper-limb rehabilitation and ADL tasks (`Task 13–20`: shoulder flexion/abduction/rotation, wrist flexion/deviation, drinking from cup, box off shelf, drawing circles).
  - **Gold-Standard Optical MoCap Joint Angles:** Synchronized **120 Hz Optotrak Certus 3D `Elbow` angles** (`H_Joint_Angles_Aligned_FINAL` / `A_OptotrackComplete` + `B_dotComplete`), supporting both **10-class rehabilitation exercise classification** and **continuous elbow flexion/extension & pronation/supination angle/ROM estimation** across **15 right-handed subjects** (`subject 1` to `subject 16`, excluding left-side `subject 2`; `12–15` subjects for joint angles).

#### Option C (Alternative Multi-Cohort Rehab Exercise Classification): `JU-IMU` (`ROM`)
- **Why this dataset is a strong secondary benchmark:**
  - **41 subjects** (`27 Non-Disabled` controls + `14 Stroke` patients) with **Upper Arm (`sensor4/5`) + Forearm/Wrist (`sensor1/2`)** 9-axis IMUs at **80 Hz** (`0` NaNs across `574` files).
  - **14 upper-limb ROM/rehab exercises** (perfectly balanced: 41 trials per class), explicitly including `ElbFlex` (elbow flexion/extension), `SupPro` (forearm supination/pronation), and `FrontSupPro`.

> **Recommendation for First Experiment:**  
> Start the first 4-model comparison (`TCN`, `1D CNN`, `LSTM`, `GRU`) on **`PHYTMO` (`inertial\upper`)** (which uniquely tests both **exercise classification** and **correct vs. incorrect execution quality** on 30 subjects using Upper Arm + Forearm IMUs) **and/or** **`Optotrak vs DOT`** (which tests **10-class elbow/forearm/shoulder rehabilitation exercise classification** and **continuous elbow ROM estimation** on Upper Arm + Forearm IMUs). Both datasets remain strictly separate (never merged).

---

## 2. Input Channels and Target Labels

To ensure direct transferability to the future **ESP32 dual-IMU hardware (Stage 6)**—where 6-axis IMU streams (3-axis Accelerometer + 3-axis Gyroscope per sensor) are standard and magnetometers are prone to indoor ferromagnetic disturbances—the core model input tensor will be standardized to **12 inertial channels**:

$$\mathbf{X} \in \mathbb{R}^{W \times 12}$$

where $W$ is the window length in samples and the 12 channels are:
1. **IMU 1 (Upper Arm — 6 channels):** `ua_acc_x`, `ua_acc_y`, `ua_acc_z` ($\text{m/s}^2$), `ua_gyr_x`, `ua_gyr_y`, `ua_gyr_z` ($\text{rad/s}$ or $\text{deg/s}$, standardized before z-scoring).
2. **IMU 2 (Forearm — 6 channels):** `fa_acc_x`, `fa_acc_y`, `fa_acc_z` ($\text{m/s}^2$), `fa_gyr_x`, `fa_gyr_y`, `fa_gyr_z` ($\text{rad/s}$ or $\text{deg/s}$).

*(Note: An ablation adding magnetometer/quaternion channels can be logged separately in Stage 5, but the primary 4-model comparison will use the 12-channel 6-DOF dual-IMU representation so the winning model can be deployed directly to dual 6-axis IMUs such as MPU6050 / ICM-42688 / LSM6DSO on ESP32).*

### Supported Prediction Targets
- **On `PHYTMO` (`inertial\upper`, dominant/active right limb `Rarm` + `Rforearm`):**
  1. **6-Class Exercise + Execution Quality Target (`y_6class` $\in \{0..5\}$):** `EFE0` (Elbow Flex/Ext Correct), `EFE1` (Elbow Flex/Ext Incorrect), `EAH0` (Overhead Arm Ext Correct), `EAH1` (Overhead Arm Ext Incorrect), `SQZ0` (Squeezing Correct), `SQZ1` (Squeezing Incorrect).
  2. **Binary Execution Quality Target (`y_quality` $\in \{0, 1\}$ on `EFE` subset and across all 3 exercises):** `0` = Correct execution, `1` = Incorrect execution.
- **On `Optotrak vs DOT` (`RUA` + `RLA`, Tasks 11–20):**
  1. **10-Class Rehabilitation Exercise Target (`y_task` $\in \{0..9\}$):** Tasks `11` (`Elbow Flex/Ext`) through `20` (`Drawing Circles`).
  2. **Continuous Elbow Joint Angle / ROM Target (`y_angle` $\in \mathbb{R}^{W \times 2}$):** Synchronized 3D Optotrak `Elbow` flexion/extension and pronation/supination angles ($\text{deg}$), from which window-level and trial-level $\text{ROM} = \max(\theta) - \min(\theta)$ is computed.

---

## 3. Preprocessing, Sampling Rates, Missing Values, and Windowing

### A. Handling Sampling Rates and Missing Values
1. **Strict Raw Data Preservation:** All files in `d:\me\project\datasets\raw\` are read-only. Cleaned trial trajectories are written to `data/interim/` and windowed splits to `data/processed/`.
2. **Timestamp Deduplication & Uniform Resampling:**
   - **In `PHYTMO`:** Although nominal rate is `100.0 Hz`, `98` of `1,484` files contain duplicate consecutive timestamps (`252` steps total) and slight jitter (`98.7–101.5 Hz`). For each trial, duplicate timestamps will be deduplicated, the 4 sensor streams (`Larm`, `Lforearm`, `Rarm`, `Rforearm`) will be aligned on their common overlapping time interval $[t_{\text{start}}, t_{\text{end}}]$, and linearly interpolated onto an exact uniform **`100.0 Hz`** grid ($\Delta t = 0.010\text{ s}$).
   - **In `Optotrak vs DOT`:** Drop row 0 (`PacketCounter == 0`, where raw `Acc/Gyr` are `0.0`), handle the 5 CSVs without the `sep=,` line, and exclude the 69 backup files in `delete` / `Fist attempt...` folders. Sampling rate is already a uniform **`120.0 Hz`** ($\Delta t = 8333\ \mu\text{s}$), or can be resampled to **`100.0 Hz`** if a single unified hardware sampling frequency (100 Hz) is desired across the entire project.
3. **Missing Values:**
   - Both `PHYTMO` (`inertial\upper`) and `Optotrak vs DOT` (`B_dotComplete` / raw DOT CSVs) have **0 NaNs** in their IMU streams. Any trial with corrupted or missing sensor files (e.g., `A02Calib` in `PHYTMO`) is logged and excluded via a deterministic manifest.

### B. Normalization (Leakage-Free)
- Unit conversion to consistent physical units: Accelerometers in $\text{m/s}^2$ ($1g = 9.80665\text{ m/s}^2$), Gyroscopes in $\text{rad/s}$ ($\text{deg/s} \times \pi / 180$).
- **Channel-wise $Z$-score standardization:**
  $$\tilde{x}_{c}(t) = \frac{x_{c}(t) - \mu_{c,\text{train}}}{\sigma_{c,\text{train}} + \epsilon}$$
  where $\mu_{c,\text{train}}$ and $\sigma_{c,\text{train}}$ ($c \in \{1,\dots,12\}$) are computed **exclusively across all training-split windows** and saved to `data/processed/scaler_stats.json`. Validation and test splits are transformed using the frozen training statistics.

### C. Time-Windowing Strategy
- **Window Duration:** **$2.0\text{ seconds}$** ($W = 200$ samples at $100\text{ Hz}$, or $W = 240$ samples at $120\text{ Hz}$).
  - *Rationale:* A single upper-limb rehabilitation repetition typically takes $1.5\text{–}3.5\text{ s}$; a $2.0\text{ s}$ window captures the kinematic phase transitions (acceleration, peak flexion/extension turnaround, and deceleration) while keeping inference latency low for real-time feedback.
- **Window Stride / Overlap:** **$0.5\text{ seconds}$** ($75\%$ overlap, stride $S = 50$ samples at $100\text{ Hz}$) within each continuous trial **after** trials have already been assigned to `train`, `val`, or `test` by Subject ID.
- **Boundary Rule:** Windows are extracted strictly within single trial files; no window ever crosses a trial or subject boundary.

---

## 4. Subject-Independent Data Splitting & Leakage Prevention

To guarantee zero data leakage:

1. **Strict Subject-Level Partitioning (`GroupShuffleSplit` / Stratified Subject Split, Seed = `42`):**
   - Every subject ID belongs to **exactly one partition** (`train`, `val`, or `test`):
     $$\mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{val}} = \emptyset, \quad \mathcal{S}_{\text{train}} \cap \mathcal{S}_{\text{test}} = \emptyset, \quad \mathcal{S}_{\text{val}} \cap \mathcal{S}_{\text{test}} = \emptyset$$
   - **For `PHYTMO` (30 subjects across 5 age groups `A`–`E`):**
     Stratified by age cohort (`A`–`E`) so every split has an identical demographic distribution:
     - **Train (70% — 21 subjects):** 7 from `A`, 3–4 each from `B, C, D, E`
     - **Validation (15% — 4 subjects):** 1 from `A`, 3 across `B–E`
     - **Test (15% — 5 subjects):** 2 from `A`, 3 across `B–E`
   - **For `Optotrak vs DOT` (15 right-side subjects; 13 with full `I_tasks_split`):**
     - **Train:** 9 subjects | **Validation:** 3 subjects | **Test:** 3 subjects (with `subject 2` excluded due to left-arm placement, and `subject 8` excluded from optical angle regression due to missing `ME` landmark).
   - **For `JU-IMU` (41 subjects: 27 `ND`, 14 `Stroke`), if run:**
     Stratified by cohort (`ND` vs `Stroke`):
     - **Train (70% — 29 subjects):** 19 `ND` + 10 `Stroke`
     - **Validation (15% — 6 subjects):** 4 `ND` + 2 `Stroke`
     - **Test (15% — 6 subjects):** 4 `ND` + 2 `Stroke`
2. **Preventing Overlapping-Window Leakage:**
   - Sliding windows are generated **inside each partition after subject assignment**. Because no trial is shared between splits, zero overlapping windows can ever cross between `train`, `val`, and `test`.
3. **Automated Leakage Unit Test (`tests/test_splits.py` in Stage 3):**
   - Before any model training begins in Stage 4, an automated pytest suite will assert:
     - Disjoint subject sets across `train`, `val`, and `test`.
     - Disjoint source trial filenames across `train`, `val`, and `test`.
     - Zero NaN/Inf values in processed tensors.
     - Exact identity of input window indices and labels fed to all 4 models.

---

## 5. Fair Four-Model Comparison Protocol (Stage 4)

All four architectures will be implemented in PyTorch (`src/models/`) and trained/evaluated under identical experimental conditions:

### A. The Four Architectures
1. **TCN (Temporal Convolutional Network):**
   - Stacked residual blocks with causal/dilated 1D convolutions (dilation factors $d = 1, 2, 4, 8, 16$, kernel size $k = 3$ or $5$), weight/batch normalization, spatial dropout, global average pooling over time, and linear classification/regression head.
2. **1D CNN (One-Dimensional Convolutional Neural Network):**
   - Standard multi-stage 1D convolutional blocks ($d = 1$, kernel sizes $7 \rightarrow 5 \rightarrow 3$) with BatchNorm, ReLU/GELU, max/average pooling layers, dropout, global average pooling, and linear head.
3. **LSTM (Long Short-Term Memory):**
   - 2-layer stacked unidirectional (or bidirectional for offline window classification) LSTM with dropout, followed by temporal pooling / last-step hidden state projection and linear head.
4. **GRU (Gated Recurrent Unit):**
   - 2-layer stacked GRU matching the LSTM depth and hidden-state capacity, followed by identical pooling and linear projection head.

### B. Fairness Controls
- **Identical Data & Partitions:** All 4 models consume the exact same normalized `(N, W, 12)` `train`, `val`, and `test` tensors.
- **Comparable Parameter Budget:** Hidden dimensions will be tuned so all 4 models have a comparable parameter count (e.g., target ~`50k–150k` trainable parameters, suitable for edge/ESP32 deployment).
- **Identical Training Protocol:**
  - Optimizer: `AdamW` (initial learning rate `1e-3`, weight decay `1e-4`) with `ReduceLROnPlateau` or cosine annealing scheduler.
  - Loss Function: Class-weighted Cross-Entropy (`nn.CrossEntropyLoss`) for classification; Smooth L1 / MSE loss for joint angle regression.
  - Batch Size: `64`, Max Epochs: `100` with Early Stopping (`patience = 15` on validation macro-F1 / validation MAE).
  - Random Seeds: Run across 3 fixed random seeds (`42, 123, 999`) to report `mean ± std` and verify stability.

---

## 6. Evaluation Metrics & Edge Efficiency Profiling (Stage 5)

### A. Classification Performance Metrics (Evaluated on Held-Out Test Subjects)
- **Macro-F1 Score** (primary model selection metric)
- **Balanced Accuracy** and **Overall Accuracy**
- **Per-Class Precision, Recall, and F1-Score**
- **Full Confusion Matrix** (both raw counts and row-normalized percentages, saved to `experiments/metrics/` and plotted to `outputs/`)
- **Subject-wise Macro-F1 Breakdown** (mean $\pm$ SD across individual test subjects to quantify inter-subject generalization variance)

### B. Continuous Joint Angle / ROM Metrics (When Evaluating ROM Estimation on `Optotrak vs DOT`)
- **Mean Absolute Error ($\text{MAE}$, in degrees)** and **Root Mean Squared Error ($\text{RMSE}$, in degrees)** for elbow flexion/extension and forearm pronation/supination
- **Pearson Correlation Coefficient ($r$)** and **$R^2$**
- **Range of Motion Error ($\Delta\text{ROM} = |\widehat{\text{ROM}} - \text{ROM}_{\text{GT}}|$ in degrees)** per repetition/trial

### C. Model Size & Inference Latency Profiling (For Stage 6 ESP32 Readiness)
For each of the 4 models (`TCN`, `1D CNN`, `LSTM`, `GRU`), we will measure and report:
1. **Total Trainable Parameters** and **Serialized Checkpoint Size (KB / MB)** (FP32 and estimated INT8 quantized footprint).
2. **MACs / FLOPs per 2.0-second window.**
3. **Single-Window (`batch_size = 1`) Inference Latency (ms):**
   - Warm-up 50 iterations, then time 1,000 forward passes on CPU (`torch.inference_mode()`), reporting **mean, median (p50), p95, and p99 latency in milliseconds** and throughput (windows/sec).

---

## 7. Open Decisions Requiring Your Approval Before Stage 3

Before we write any Stage 3 data-processing code, please confirm your preferences on the following items:

1. **Primary Dataset Selection for Stage 3 & Stage 4:**
   - Do you approve starting with **`PHYTMO` (`inertial\upper`)** (30 subjects, Upper Arm + Forearm 9-axis IMUs at 100 Hz, supporting **both exercise classification and Correct vs. Incorrect execution classification**), **`Optotrak vs DOT`** (16 subjects, Upper Arm + Forearm IMUs at 120 Hz, supporting **10-class elbow/forearm/shoulder rehab classification + continuous elbow ROM estimation**), or preparing **both** as separate benchmark tracks?
2. **Sensor Channels (`12-channel 6-DOF Acc+Gyr` vs. `18-channel 9-DOF Acc+Gyr+Mag`):**
   - Do you approve using the **12-channel 6-DOF Upper Arm + Forearm (`Acc_X/Y/Z` + `Gyr_X/Y/Z` × 2 sensors)** representation as the primary input so the trained models match standard 6-axis ESP32 IMUs without magnetometer calibration drift?
3. **Raw Dataset Storage Location (`datasets/raw/` vs. `data/raw/`):**
   - All 13 raw datasets (`68.23 GB`) currently reside in `d:\me\project\datasets\raw\`. Do you want to **keep them in `d:\me\project\datasets\raw\`** and point our Stage 3 data loaders directly to `datasets/raw/` (recommended, zero disk movement), or move/junction them into `d:\me\project\data\raw\`?
