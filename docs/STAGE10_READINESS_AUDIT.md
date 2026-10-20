# Stage 10.1 — Final Model Readiness Audit (`docs/STAGE10_READINESS_AUDIT.md`)

**Project:** AI-Based Elbow Rehabilitation Assistance System After Olecranon Fracture ORIF  
**Workspace Root:** `D:\me\project`  
**Audit Date:** 2026-10-19  
**Audit Scope:** Strictly Read-Only Code, Dataset, Preprocessing, Checkpoint, and Sample-Inference Readiness Audit prior to Stage 10 Model Development.

---

## 1. Executive Summary

This audit evaluates the readiness of the `D:\me\project` repository to advance into **Stage 10**, where the four non-negotiable deep learning architectures (**`TCN`**, **`1D_CNN`**, **`LSTM`**, and **`GRU`**) will be compared on identical held-out sample recordings across all core post-olecranon ORIF rehabilitation tasks, previewed interactively in the existing FastAPI supervisor web demo, and prepared for real dual-IMU recordings.

### High-Level Readiness Verdict

| Audit Dimension | Readiness Status | Key Finding |
| :--- | :---: | :--- |
| **1. Four Model Architectures (`TCN`, `1D_CNN`, `LSTM`, `GRU`)** | **READY** | All 4 models are cleanly implemented in [`src/models/architectures.py`](file:///d:/me/project/src/models/architectures.py) with matched parameter budgets (`83.6k–116.8k`), identical input tensor contracts `(B, W, C)`, and unified `TaskOutputHead` supporting `"classification"`, `"regression"`, and `"multitask"` modes. |
| **2. Existing Checkpoints (`64` `.pt` files across `16` tracks)** | **READY (Single-Task)** | All `64` checkpoints (`16` track configurations $\times$ `4` models) in [`models/checkpoints/`](file:///d:/me/project/models/checkpoints) load cleanly on CPU, bundle frozen train-only `ChannelStandardScaler` statistics (`mean`, `std`), and support side-by-side 4-model inference on identical sample inputs. **Note:** `40/64` checkpoints (`10` tracks) are **Primary/Secondary Validated**, `16/64` (`4` tracks) are **Auxiliary/Exploratory**, and `8/64` (`Tracks 3D & 4B` on `ULF_in_ADL`) are **Unreliable as Trained** due to unsegmented `P28` file dominance (`88.6%` of test windows). |
| **3. Canonical 12-Channel Preprocessing & Unit Standardization** | **READY (Offline) / ACTION NEEDED (Streaming)** | All primary datasets map cleanly to the canonical 12-channel Upper-Arm + Forearm SI schema (`m/s²` and `rad/s`) in [`src/data/schemas.py`](file:///d:/me/project/src/data/schemas.py) and [`src/data/loaders.py`](file:///d:/me/project/src/data/loaders.py). However, `compute_orientation_invariant_12ch()` (used only in cross-dataset Tracks `5A-1` and `5A-2`) subtracts a **full-trial mean** (`ua_acc.mean(axis=0)`), which is non-causal for real-time streaming. |
| **4. Comparable 4-Model Sample Inference & Supervisor Web Demo** | **READY** | [`src/inference/engine.py`](file:///d:/me/project/src/inference/engine.py), [`src/inference/demo_samples.py`](file:///d:/me/project/src/inference/demo_samples.py), and [`src/inference/app.py`](file:///d:/me/project/src/inference/app.py) already serve **35 curated held-out test sample CSVs** ([`data/processed/demo_samples/manifest.json`](file:///d:/me/project/data/processed/demo_samples/manifest.json)) across 8 exercise categories and support instant single-model and 4-model comparison (`/api/demo/predict`, `/api/demo/compare-models`) without losing sample selection. |
| **5. Simultaneous Multi-Head Inference on a Single Recording** | **BLOCKER FOR STAGE 10.2** | Currently, each checkpoint is trained on a single track (`classification` OR `regression`). Running a single recording through **all 5 core tasks simultaneously** (Flexion/Extension angle + Pronation/Supination angle + Full-Repetition ROM + Exercise Class + Execution Quality) requires either (a) a **Stage 10 Unified Multi-Head / Composite Inference Pipeline** that runs compatible `100 Hz` 12-channel checkpoints in parallel (`Track 3C` / resampled `Track 3A` for angles/ROM + `Track 1C`/`1A` for exercise class + `Track 2A`/`2B` for execution quality) or (b) training the `TaskOutputHead(task_mode="multitask")` architecture in Stage 10. |

---

## 2. Project Structure & Reusable Components Inventory

All core modules were inspected directly in `D:\me\project`:

| Component Category | Existing Files | Reusability Assessment for Stage 10 |
| :--- | :--- | :--- |
| **Model Architectures** | [`src/models/architectures.py`](file:///d:/me/project/src/models/architectures.py) | **100% Reusable.** Defines `TaskOutputHead`, `TemporalBlock`, `TCNModel`, `CNN1DModel`, `LSTMModel`, `GRUModel`, `build_four_models()`, and `compute_masked_multitask_loss()`. |
| **Data Schemas & SI Units** | [`src/data/schemas.py`](file:///d:/me/project/src/data/schemas.py) | **100% Reusable.** Defines `CANONICAL_DUAL_IMU_12CH`, `CANONICAL_JOINT_ANGLE_TARGETS_DEG`, `SHARED_EXERCISE_TAXONOMY`, `convert_acc_to_ms2()`, `convert_gyr_to_rads()`, `check_timestamp_integrity()`, and `TrialRecord`. |
| **Dataset Loaders & Resampling** | [`src/data/loaders.py`](file:///d:/me/project/src/data/loaders.py) | **Reusable with 2 targeted Stage 10 updates:** (1) add a strict file-size/frame-count filter (`<= 1,500` frames) inside `load_ulf_in_adl_mvnx_12ch()` to exclude unsegmented `P28` recordings, and (2) add a window-local causal mode to `compute_orientation_invariant_12ch()`. |
| **Windowing & Normalization** | [`src/features/windowing.py`](file:///d:/me/project/src/features/windowing.py) | **100% Reusable.** Implements `ChannelStandardScaler` (enforcing `split_name == "train"` during `.fit()`), `extract_trial_windows()`, and `build_split_window_tensors()`. |
| **Split Generation & Validation** | [`src/data/splits.py`](file:///d:/me/project/src/data/splits.py), [`data/processed/splits/multi_dataset_split_manifests.json`](file:///d:/me/project/data/processed/splits/multi_dataset_split_manifests.json) | **100% Reusable.** Enforces zero subject overlap (`train_subjects ∩ val_subjects ∩ test_subjects == ∅`) across all subject-identified datasets. |
| **Evaluation Metrics** | [`src/evaluation/metrics.py`](file:///d:/me/project/src/evaluation/metrics.py) | **Reusable; extend in Stage 10** to add subject-equalized macro-averaging and individual repetition peak-to-trough ROM cycle segmentation alongside existing window/trial ROM metrics. |
| **Benchmark & Audit Scripts** | [`scripts/run_stage8_benchmarks.py`](file:///d:/me/project/scripts/run_stage8_benchmarks.py), [`scripts/run_stage8_remaining_tracks.py`](file:///d:/me/project/scripts/run_stage8_remaining_tracks.py), [`scripts/run_stage9_forensic_audit.py`](file:///d:/me/project/scripts/run_stage9_forensic_audit.py) | **100% Reusable** as reference training/evaluation harnesses and forensic latency/subject-equalized audit tools. |
| **Supervisor Web Demo Backend & Frontend** | [`src/inference/app.py`](file:///d:/me/project/src/inference/app.py), [`src/inference/engine.py`](file:///d:/me/project/src/inference/engine.py), [`src/inference/demo_samples.py`](file:///d:/me/project/src/inference/demo_samples.py), [`src/inference/schemas.py`](file:///d:/me/project/src/inference/schemas.py), [`src/inference/static/index.html`](file:///d:/me/project/src/inference/static/index.html), [`src/inference/static/app.js`](file:///d:/me/project/src/inference/static/app.js), [`src/inference/static/styles.css`](file:///d:/me/project/src/inference/static/styles.css) | **100% Reusable & Active.** Preserve all existing Stage 9.5 and Stage 9.6 endpoints (`/api/overview`, `/api/tracks`, `/api/samples`, `/api/predict`, `/api/demo/catalog`, `/api/demo/samples`, `/api/demo/sample-preview/{id}`, `/api/demo/predict`, `/api/demo/compare-models`). |
| **Automated Test Suite** | [`tests/test_pipeline.py`](file:///d:/me/project/tests/test_pipeline.py), [`tests/test_supervisor_demo.py`](file:///d:/me/project/tests/test_supervisor_demo.py) | **100% Passing (`35/35` tests).** Covers unit conversions, split disjointness, windowing, scaler leakage prevention, model forward/backward passes, and all FastAPI endpoints. |

---

## 3. Preprocessing Pipeline Audit & Offline vs. Real-Time Streaming Analysis

### 3.1 Sampling Rates, Window Lengths, and Strides

All datasets are windowed into fixed-duration temporal windows (2.0 seconds for Tracks `1A–4B, 5A–5B`; 3.0 seconds for Track `4C`):

| Track(s) | Dataset(s) | Sampling Rate (`Hz`) | Window Samples ($W$) | Train/Eval Stride ($S$) | Dense Trajectory Reconstruction Stride | Input Channels ($C$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`1A, 3A`** | `Optotrak vs DOT` | `120.0 Hz` | `240` (`2.0 s`) | `120` (`1.0 s`, 50% overlap) | `48` (`0.4 s`, $W/5$) | `12` (SI Raw Dual-IMU) |
| **`1C, 2A, 2B, 3B, 3C, 5B`** | `PHYTMO`, `StrokeRehab`, `ULTra-MoCap` | `100.0 Hz` | `200` (`2.0 s`) | `100` (`1.0 s`, 50% overlap) | `40` (`0.4 s`, $W/5$) | `12` (SI Raw Dual-IMU) |
| **`5A-1, 5A-2`** | Cross-Dataset (`JU-IMU`, `Optotrak`, `PHYTMO`) | `100.0 Hz` (resampled) | `200` (`2.0 s`) | `100` (`1.0 s`, 50% overlap) | N/A (Classification) | `12` (Orientation-Invariant) |
| **`1B, 4A`** | `JU-IMU` | `80.0 Hz` | `160` (`2.0 s`) | `80` (`1.0 s`, 50% overlap) | N/A (Classification) | `12` (SI Raw Dual-IMU) |
| **`3D, 4B`** | `ULF_in_ADL` | `60.0 Hz` | `120` (`2.0 s`) | `60` (`1.0 s`, 50% overlap) | `24` (`0.4 s`, $W/5$) | `12` (SI Raw Dual-IMU) |
| **`3E`** | `VIDIMU` | `50.0 Hz` | `100` (`2.0 s`) | `50` (`1.0 s`, 50% overlap) | `20` (`0.4 s`, $W/5$) | `8` (Dual-IMU Quaternions) |
| **`4C`** | `REHAB-120` | `50.0 Hz` | `150` (`3.0 s`) | `75` (`1.5 s`, 50% overlap) | N/A (Classification) | `12` (SI Raw Dual-IMU) |

### 3.2 Sensor Channel Ordering & Unit Conversions

For all 12-channel raw dual-IMU tracks (`15/16` tracks), the channel order is strictly standardized by `CANONICAL_DUAL_IMU_12CH` in [`src/data/schemas.py`](file:///d:/me/project/src/data/schemas.py#L14-L27):
1. **Upper Arm (`IMU 1`, indices `0..5`):** `[ua_acc_x_ms2, ua_acc_y_ms2, ua_acc_z_ms2, ua_gyr_x_rads, ua_gyr_y_rads, ua_gyr_z_rads]`
2. **Forearm (`IMU 2`, indices `6..11`):** `[fa_acc_x_ms2, fa_acc_y_ms2, fa_acc_z_ms2, fa_gyr_x_rads, fa_gyr_y_rads, fa_gyr_z_rads]`

**Verified Dataset-Specific Unit Conversions in [`src/data/loaders.py`](file:///d:/me/project/src/data/loaders.py):**
- **`PHYTMO`** (`load_phytmo_trial_12ch`, lines `99–102`): Accelerometer converted from `g` to `m/s²` ($\times 9.80665$); Gyroscope converted from `deg/s` to `rad/s` ($\times \pi/180$).
- **`Optotrak vs DOT`** (`load_optotrak_dot_trial_12ch`, lines `172–175`): Already in `m/s²` (free acceleration, gravity removed; clipped to $\pm 100\text{ m/s}^2$ to guard against numerical dropout spikes); Gyroscope converted from `deg/s` to `rad/s`.
- **`JU-IMU`** (`load_ju_imu_trial_12ch`, lines `207–210`): Accelerometer converted from `mG` to `m/s²` ($\times 9.80665 / 1000$); Gyroscope converted from `deg/s` to `rad/s`.
- **`StrokeRehab`** (`load_strokerehab_trial_12ch`, lines `250–266`): Accelerometer converted from `g` to `m/s²`; angular velocity reconstructed by differentiating unwrapped Euler angles (`deg` $\rightarrow$ `rad/s`) at `100 Hz`.
- **`ULTra-MoCap`** (`load_ultra_mocap_trial_12ch`, lines `306–309`): Accelerometer converted from `mm/s²` to `m/s²` ($\div 1000.0$); Gyroscope already in `rad/s`; multi-rate (`60 Hz` MoCap / `100 Hz` IMU) synchronized to `100 Hz`.
- **`ULF_in_ADL`** (`load_ulf_in_adl_mvnx_12ch`, lines `366–367`): Already in SI `m/s²` and `rad/s` in `.mvnx` XML segments (`RightUpperArm` = index 8, `RightForeArm` = index 9).
- **`VIDIMU`** (`load_vidimu_ trial_8ch_quat`, lines `396–401`): 8-channel unit quaternions `[ua_q0, ua_q1, ua_q2, ua_q3, fa_q0, fa_q1, fa_q2, fa_q3]`.
- **`REHAB-120`** (`load_rehab120_trial_12ch`, lines `425–437`): Already in SI `m/s²` and `rad/s`.

### 3.3 Missing-Value & Timestamp-Integrity Handling

- **NaN Interpolation:** Every loader in [`src/data/loaders.py`](file:///d:/me/project/src/data/loaders.py) passes signals through `_interpolate_internal_nans(arr, max_gap=15)` (lines `28–47`), which trims leading/trailing all-NaN rows and linearly interpolates internal gaps up to 15 samples (`125 ms` at `120 Hz`). If any non-finite values remain, `extract_trial_windows()` raises a `ValueError` (line `72` of [`src/features/windowing.py`](file:///d:/me/project/src/features/windowing.py#L71-L72)).
- **Duplicate Timestamp Handling:** `load_phytmo_trial_12ch()` drops duplicate timestamps via `df.drop_duplicates(subset=[time_col], keep="first")` (line `84` of `src/data/loaders.py`), and `check_timestamp_integrity()` in [`src/data/schemas.py`](file:///d:/me/project/src/data/schemas.py#L121-L157) audits monotonic timestamps and sampling rate drift.

### 3.4 Normalization (`ChannelStandardScaler`) & Angle Conventions

- **Train-Only Fitting:** [`ChannelStandardScaler.fit()`](file:///d:/me/project/src/features/windowing.py#L19-L31) explicitly asserts `if split_name != "train": raise ValueError(...)`. All 64 checkpoints store the fitted training scaler (`ckpt["scaler"]` for classification; `ckpt["x_scaler"]` and `ckpt["y_scaler"]` for regression).
- **Angle Target Conventions (`deg`):**
  - Target 0: `elbow_flex_ext_deg` (`0°` = full extension, positive = flexion, typical physiological range `0°–145°`).
  - Target 1: `forearm_pro_sup_deg` (continuous axial forearm rotation in degrees; `Optotrak vs DOT` uses ISB `Elbow[:, 1]`, `ULTra-MoCap` uses OpenSim `pro_sup_r`, `VIDIMU` uses OpenSim `elbow_pro_sup`).

### 3.5 Critical Audit Finding: Offline vs. Real-Time Streaming Operations

We inspected all preprocessing, windowing, model layers, and inference code for any operations that depend on future frames or full-trial statistics:

1. **Full-Trial Mean Subtraction in `compute_orientation_invariant_12ch()` ([`src/data/loaders.py` lines 455 & 459](file:///d:/me/project/src/data/loaders.py#L440-L486)):**
   - **Code inspected:**
     ```python
     ua_dyn_norm = np.linalg.norm(ua_acc - ua_acc.mean(axis=0, keepdims=True), axis=1)
     fa_dyn_norm = np.linalg.norm(fa_acc - fa_acc.mean(axis=0, keepdims=True), axis=1)
     ```
   - **Impact:** In both [`scripts/run_stage8_benchmarks.py`](file:///d:/me/project/scripts/run_stage8_benchmarks.py) and [`src/inference/engine.py`](file:///d:/me/project/src/inference/engine.py#L1199), `compute_orientation_invariant_12ch(raw_12ch)` is called on the **full trial array `(T, 12)` before `extract_trial_windows()`**. Consequently, `ua_acc.mean(axis=0)` and `fa_acc.mean(axis=0)` average across the entire recording duration rather than strictly within each 2.0-second window.
   - **Scope:** This affects **only the two cross-dataset tracks (`Track 5A-1` and `Track 5A-2`)**. None of the 14 single-dataset tracks (`1A, 1B, 1C, 2A, 2B, 3A, 3B, 3C, 3D, 3E, 4A, 4B, 4C, 5B`) call `compute_orientation_invariant_12ch()`.
   - **Stage 10 Remediation:** Add a `window_local: bool = True` option (or apply `compute_orientation_invariant_12ch` *inside* each 2.0-second window / via causal exponential moving average) when training Stage 10 orientation-invariant models for real-time streaming.
2. **Zero-Phase Forward-Backward Filtering in `resample_multirate_signal()` ([`src/data/loaders.py` line 63](file:///d:/me/project/src/data/loaders.py#L50-L68)):**
   - **Impact:** When downsampling (`target_hz < source_hz`), `scipy.signal.sosfiltfilt` filters forward and backward across the trial.
   - **Stage 10 Remediation:** When operating at the native `100 Hz` hardware sampling rate (or `PHYTMO` / `ULTra-MoCap` at `100 Hz`), `abs(source_hz - target_hz) < 1e-4` returns a direct copy without calling `sosfiltfilt`. If real-time downsampling is ever needed, use causal `scipy.signal.sosfilt`.
3. **Buffered 2.0-Second Window Inference (`TemporalBlock` Symmetric Padding & Global Window Pooling):**
   - In [`src/models/architectures.py`](file:///d:/me/project/src/models/architectures.py#L69-L77), `TemporalBlock` (`TCN`) and `CNN1DModel` use symmetric centered padding (`padding = (kernel_size - 1) * dilation // 2`), and `TaskOutputHead` (`classification`) pools over the 2.0-second window (`mean(dim=1)` + `amax(dim=1)`).
   - **Implication:** In both offline sample evaluation and real-time hardware deployment, all 4 architectures operate on a **sliding 2.0-second FIFO ring buffer** `(1, W, 12)` (e.g., `200` samples at `100 Hz`), emitting updated predictions every stride step (e.g., every `100–400 ms`). Within that 2.0-second buffer, centered convolutions are completely valid and deterministic.

---

## 4. Verification of Data-Quality Safeguards (Stage 1–9 Findings Preserved)

All previously identified data-quality exclusions and caveats remain strictly enforced:

| Dataset / Track | Safeguard / Finding | Current Enforcement Status |
| :--- | :--- | :--- |
| **5 Excluded Raw Datasets** (`EMG elbow dataset`, `mm-fit`, `MoCap and IMU Dataset...`, `Smartwatch IMU Dataset...`, `UCI Biceps Curl`) | Excluded due to missing forearm IMU (`mm-fit`), single wrist IMU (`Smartwatch`), EMG-only (`EMG`), 100% subject-label confounding (`UCI Biceps Curl`), or raw uncalibrated ticks (`MoCap and IMU`). | **Strictly Excluded.** None are loaded in `src/data/loaders.py` or `configs/benchmark_tracks.json`. |
| **`JU-IMU` Invalid Pseudo-Labels** | Do not fabricate correct/incorrect execution labels by contrasting `ND` vs `Stroke` cohorts. | **Strictly Enforced.** `JU-IMU` is used only for 14-class exercise recognition (`Track 1B`), valid `Stroke vs ND` cohort analysis (`Track 4A`), and 7-class cross-dataset transfer (`Track 5A-1`). |
| **`ULF_in_ADL` Subject `P28` Anomaly (`Tracks 3D & 4B`)** | Subject `P28` has 13 unsegmented multi-minute `.mvnx` files (`299.85 MB`, `~39,935` frames each), accounting for `7,839 / 8,848` (`88.6%`) of test windows. Furthermore, `Track 4B` has near-random 3-class performance (`Macro-F1 = 0.3091–0.3620`). | **Flagged in Demo UI & Docs.** `Track 4B` is marked `excluded_from_headline` in `src/inference/engine.py`; Stage 9 supplemental audit (`experiments/metrics/stage9_supplemental_audit.json`) provides Subject-Equalized MAE for `Track 3D`. For Stage 10, `load_ulf_in_adl_mvnx_12ch` must filter out `.mvnx` files `> 1,500` frames (`> 5 MB`). |
| **`StrokeRehab` Patient Parquet Exclusion (`Track 3B`)** | Stroke patient `.parquet` files lack synchronized `_angles.csv` and only 6 of 51 patients (`40/3,372` trials) have video-annotated `_labels.csv`. | **Strictly Enforced.** `Track 3B` loads only the `20` healthy control subjects (`C1..C20`) that have ground-truth `_angles.csv`. |
| **`ULTra-MoCap` Cross-Speed Split Caveat (`Track 5B`)** | `Track 5B` splits by movement speed (`Normal` $\rightarrow$ `Slow, Fast, VeryFast`) across the same 13 subjects. | **Flagged as Exploratory.** Clearly labeled in `src/inference/engine.py` and `docs/FINAL_DATASET_SELECTION.md` as within-subject cross-speed generalization, while `Track 3C` uses a strict subject-disjoint split (`9 train / 2 val / 2 test`). |
| **`REHAB-120` Session-Level Split Caveat (`Track 4C`)** | Dataset provides `222` numbered sessions from `120` patients without a session-to-patient mapping table. | **Flagged as Exploratory.** Clearly labeled as session-independent (`154 / 34 / 34` sessions), not guaranteed subject-independent. |

---

## 5. Readiness of Sample Data & Comparable 4-Model Inference

### 5.1 Curated Held-Out Sample Catalog (`data/processed/demo_samples/manifest.json`)

The project already contains **35 verified held-out test sample CSV files** in [`data/processed/demo_samples/`](file:///d:/me/project/data/processed/demo_samples/manifest.json) spanning **8 user-friendly rehabilitation exercise categories** and **5 primary benchmark tracks**:

| Category ID | Exercise Category Label | Source Dataset & Track | Sampling Rate & Channels | # Curated Held-Out Test Samples | Supported Ground-Truth Target(s) |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `elbow_flexion_extension` | Elbow Flexion-Extension (`EFE` / `Task 11` / `ElbFlex`) | `PHYTMO` (`1C`), `Optotrak vs DOT` (`1A, 3A`), `JU-IMU` (`1B`), `ULTra-MoCap` (`3C`) | `80–120 Hz`, `12 ch` | `10` | Exercise class (`1A, 1B, 1C`) OR Continuous Flexion + Pro/Sup angles & ROM (`3A, 3C`) |
| `elbow_flexion_quality` | Elbow Flexion-Extension Execution Quality (`Correct` vs `Incorrect`) | `PHYTMO` (`Track 2A`) | `100 Hz`, `12 ch` | `6` (`3 EFE0 Correct`, `3 EFE1 Incorrect`) | Binary Execution Quality (`EFE0_Correct` vs `EFE1_Incorrect`) |
| `forearm_pronation_supination` | Forearm Pronation-Supination (`Task 12` / `SupPro`) | `Optotrak vs DOT` (`1A, 3A`), `JU-IMU` (`1B`) | `80–120 Hz`, `12 ch` | `5` | Exercise class (`1A, 1B`) OR Continuous Flexion + Pro/Sup angles & ROM (`3A`) |
| `arm_elevation_shoulder` | Arm Elevation & Shoulder Flexion (`EAH` / `Task 14`) | `PHYTMO` (`1C`), `Optotrak vs DOT` (`1A, 3A`) | `100–120 Hz`, `12 ch` | `4` | Exercise class (`1A, 1C`) OR Continuous angles & ROM (`3A`) |
| `squeeze_hand_grip` | Isometric Squeeze / Hand Exercise (`SQZ`) | `PHYTMO` (`Track 1C`) | `100 Hz`, `12 ch` | `2` | 3-Class Exercise (`SQZ`) |
| `wrist_flexion_extension` | Wrist Flexion-Extension (`Task 16` / `WrstFlex`) | `Optotrak vs DOT` (`1A, 3A`), `JU-IMU` (`1B`) | `80–120 Hz`, `12 ch` | `3` | Exercise class (`1A, 1B`) OR Continuous angles & ROM (`3A`) |
| `functional_drinking_cup` | Functional ADL: Drinking from Cup (`Task 18` / `PickCup`) | `Optotrak vs DOT` (`1A, 3A`), `JU-IMU` (`1B`) | `80–120 Hz`, `12 ch` | `3` | Exercise class (`1A, 1B`) OR Continuous angles & ROM (`3A`) |
| `functional_box_reaching` | Functional ADL: Moving Box & Reaching (`Task 19` / `Reaching`) | `Optotrak vs DOT` (`3A`), `ULTra-MoCap` (`3C`) | `100–120 Hz`, `12 ch` | `2` | Continuous Flexion + Pro/Sup angles & ROM (`3A, 3C`) |

### 5.2 Can All Four Models Currently Process the Same Sample for Comparable Outputs?

- **Within Any Track (`1A–5B`): YES (`100%` Compatible).**
  - Given any sample recording in the catalog (or any uploaded/recorded dual-IMU CSV), `SupervisorDemoEngine.run_Interactive_demo_evaluation()` in [`src/inference/demo_samples.py`](file:///d:/me/project/src/inference/demo_samples.py) runs **`TCN`**, **`1D_CNN`**, **`LSTM`**, and **`GRU`** on the **exact same preprocessed windows** and returns side-by-side predictions, confidences, accuracies/errors, inference latencies (`ms`), and trajectory/probability arrays.
- **Across Multiple Tasks on the Same Sample (e.g., Running Exercise Classification + Execution Quality + Angle/ROM Regression on One Recording): PARTIALLY READY — Requires a Composite/Multi-Head Stage 10 Runner.**
  - Currently, each sample in `manifest.json` is bound to a single `track_id` (for instance, `phytmo_1c_EFE_C02_seg2` runs `Track 1C` 3-class classification, whereas `optotrak_3a_elbow_flex_s11_r1` runs `Track 3A` continuous angle regression).
  - However, because `PHYTMO`, `ULTra-MoCap`, and resampled `Optotrak vs DOT` all share the **exact same `100 Hz` 12-channel SI input representation `(B, 200, 12)`**, any `100 Hz` 12-channel dual-IMU recording can be fed simultaneously through:
    1. **Angle & ROM Regression Checkpoints** (`Track 3C` at `100 Hz` or `Track 3A` resampled to `120 Hz`),
    2. **Exercise Classification Checkpoints** (`Track 1C` / `Track 5A-1` at `100 Hz` or `Track 1A` at `120 Hz`), and
    3. **Execution Quality Checkpoints** (`Track 2A` / `Track 2B` at `100 Hz`),
    or through a newly trained **Stage 10 Unified `100 Hz` Multi-Task Checkpoint** (`TaskOutputHead(task_mode="multitask")`).

---

## 6. Identified Blockers & Exact Next Step for Stage 10.2

### 6.1 Summary of Blockers to Resolve in Stage 10

1. **Blocker 1 — Single-Track Sample Binding vs. Unified 5-Task Rehabilitation Output:**
   - Existing Stage 8 checkpoints are single-track (`classification` OR `regression`), and existing demo samples are mapped 1-to-1 to a single track. To display all 5 rehabilitation outputs (Elbow Flexion/Extension angle, Forearm Pronation/Supination angle, Full-Repetition ROM, Exercise Class, and Correct/Incorrect Execution Quality) for a selected sample across `TCN`, `1D_CNN`, `LSTM`, and `GRU`, Stage 10 must either provide a **Unified `100 Hz` Multi-Head / Multi-Checkpoint Inference Pipeline** or train **Stage 10 Unified `100 Hz` Checkpoints** standardized on `(W=200, C=12)`.
2. **Blocker 2 — Sampling Rate Heterogeneity Across Stage 8 Checkpoints (`120 Hz` vs `100 Hz` vs `80 Hz`):**
   - `Optotrak vs DOT` checkpoints (`1A, 3A`) were trained at `120 Hz` (`W=240`), `PHYTMO` (`1C, 2A, 2B`) and `ULTra-MoCap` (`3C`) at `100 Hz` (`W=200`), and `JU-IMU` (`1B, 4A`) at `80 Hz` (`W=160`). For Stage 10 and future ESP32 + dual-IMU hardware, standardizing the final Stage 10 models on **`100 Hz` (`W=200` samples = `2.0 s`, `C=12` SI channels)** eliminates cross-rate mismatches.
3. **Blocker 3 — Non-Causal Full-Trial Mean in `compute_orientation_invariant_12ch()`:**
   - Lines `455` and `459` of [`src/data/loaders.py`](file:///d:/me/project/src/data/loaders.py#L455-L459) subtract `ua_acc.mean(axis=0)` and `fa_acc.mean(axis=0)` computed over the full trial before windowing. If orientation-invariant features are used in Stage 10, this mean must be computed strictly within each 2.0-second window (or causally) so offline training matches real-time streaming window-for-window.
4. **Blocker 4 — `Optotrak vs DOT` Free Acceleration vs. Total Acceleration (Gravity Included):**
   - As documented in [`src/data/loaders.py` line 153](file:///d:/me/project/src/data/loaders.py#L153), `Optotrak vs DOT` provides **free acceleration** (gravity vector already subtracted by Xsens DOT firmware), whereas `PHYTMO`, `ULTra-MoCap`, `JU-IMU`, `ULF_in_ADL`, and standard raw 6-axis IMUs (MPU6050/BMI270/BNO085) output **proper acceleration** (gravity included, $\approx 9.81\text{ m/s}^2$ at rest). Therefore, raw-channel models cannot blindly pool `Optotrak vs DOT` and `ULTra-MoCap`/`PHYTMO` without either high-pass gravity removal (`>0.3 Hz` causal filter) or dataset-specific/orientation-invariant heads.
5. **Blocker 5 — `ULF_in_ADL` Unsegmented `P28` Files (`Tracks 3D & 4B`):**
   - If `ULF_in_ADL` is used for auxiliary angle regression testing, `load_ulf_in_adl_mvnx_12ch()` must reject `.mvnx` files exceeding `1,500` frames (`25 s` at `60 Hz`) so `P28` does not corrupt test metrics.

### 6.2 Exact Recommended Plan for Stage 10.2

In **Stage 10.2**, we should execute a clean, modular upgrade without breaking any existing Stage 8/9 checkpoints or demo routes:
1. **Standardize the Stage 10 Canonical `100 Hz` 12-Channel Pipeline:**
   - Ensure causal per-window gravity/mean handling and `100 Hz` (`W=200`, `C=12`) standardization across the primary rehabilitation datasets (`Optotrak vs DOT`, `ULTra-MoCap`, `PHYTMO`, and `JU-IMU`).
   - Add the `<= 1,500`-frame filter to `load_ulf_in_adl_mvnx_12ch()` for clean `ULF_in_ADL` evaluation.
2. **Implement Full-Repetition ROM Cycle Segmentation & Unified 4-Model Task Evaluation:**
   - Extend the inference and evaluation pipeline so that for any compatible dual-IMU sample, all 4 models (`TCN`, `1D_CNN`, `LSTM`, `GRU`) can be compared across continuous angle prediction (`Flexion/Extension` & `Pronation/Supination`), full-repetition and per-cycle ROM (`deg`), exercise classification, and execution quality (`Correct` vs `Incorrect`), clearly badging which ground-truth targets exist on the selected sample vs. which outputs are cross-task model predictions.
3. **Train/Validate Stage 10 Final `100 Hz` Models (Only Where Needed & Approved):**
   - Retain validated `100 Hz` Stage 8 checkpoints (`Track 1C`, `Track 2A`, `Track 2B`, `Track 3C`) and, if approved for Stage 10.2, train standardized `100 Hz` (`W=200, C=12`) angle/exercise checkpoints on `Optotrak vs DOT` (`120 Hz -> 100 Hz`) and cleaned `ULF_in_ADL` so every core task operates at a single unified `100 Hz` hardware sampling rate.
