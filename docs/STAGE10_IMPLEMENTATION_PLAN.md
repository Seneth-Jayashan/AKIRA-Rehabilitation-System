# Stage 10.2 — Implementation Plan & Preprocessing Standardization (`docs/STAGE10_IMPLEMENTATION_PLAN.md`)

**Project:** AI-Based Elbow Rehabilitation Assistance System After Olecranon Fracture ORIF  
**Workspace Root:** `D:\me\project`  
**Stage:** Stage 10.2 — Preprocessing Resolution, Input Pipeline Standardization, ROM Cycle Segmentation & 4-Model Sample Inference Preview  
**Date:** 2026-10-19

---

## 1. Objective & Scope

Following the **Stage 10.1 Final Model Readiness Audit** ([`docs/STAGE10_READINESS_AUDIT.md`](file:///d:/me/project/docs/STAGE10_READINESS_AUDIT.md)), **Stage 10.2** prepares all four non-negotiable deep-learning architectures (**`TCN`**, **`1D_CNN`**, **`LSTM`**, and **`GRU`**) for fair, side-by-side comparison on held-out sample data across all five core rehabilitation tasks, followed by future evaluation on real dual-IMU recordings—without modifying raw datasets, deleting/overwriting the 64 existing Stage 8 checkpoints, or running full training jobs prematurely.

---

## 2. Step-by-Step Implementation Summary

### Step 1 — Resolve Preprocessing Risks ([`src/data/loaders.py`](file:///d:/me/project/src/data/loaders.py))
1. **Causal & Per-Window Orientation-Invariant Features (`compute_orientation_invariant_12ch` & `compute_orientation_invariant_windows_12ch`):**
   - **Legacy Traceability Preserved (`mode="legacy_full_trial"`):** Default mode on 2D `(T, 12)` arrays preserves the exact full-trial mean (`ua_acc.mean(axis=0, keepdims=True)`) and central gradient (`np.gradient(..., axis=0)`) used when Stage 8 cross-dataset checkpoints (`track_5a1_cross_dataset_ju_to_optotrak_7class` and `track_5a2_cross_dataset_to_phytmo_2class`) were trained.
   - **Window-Isolated Mode (`mode="per_window"`):** Computes mean acceleration and backward first-difference jerk strictly within each individual 2.0-second window `(W, 12)` or `(N, W, 12)` with zero cross-window leakage.
   - **Strictly Causal Streaming Mode (`mode="causal_expanding"`):** Computes the sample-by-sample expanding mean $\mu_t = \frac{1}{t+1}\sum_{k=0}^t \mathbf{a}_k$ and backward difference $\Delta \boldsymbol{\omega}_t = \boldsymbol{\omega}_t - \boldsymbol{\omega}_{t-1}$ ($\Delta \boldsymbol{\omega}_0 = \mathbf{0}$), guaranteeing that output at frame $t$ is mathematically invariant to any future frames $k > t$.
   - **Training/Inference Parity:** `compute_orientation_invariant_windows_12ch(x_windows_nwc, mode="causal_expanding")` ensures training and inference execute the exact same transformation function.
2. **`ULF_in_ADL` Subject `P28` Unsegmented Recording Safeguard (`load_ulf_in_adl_trial_12ch` & `validate_ulf_in_adl_mvnx_segmentation`):**
   - Added `ULF_IN_ADL_MAX_SEGMENTED_FRAMES = 1500` (`25.0 s` at `60 Hz`), `ULFInADLUnsegmentedRecordingError(ValueError)`, and `validate_ulf_in_adl_mvnx_segmentation()`.
   - When `enforce_p28_safeguard=True` or `max_frames=1500` is active (the Stage 10 standard configured in `preproc_v10_ulf_60hz_raw12_p28_guarded`), any `.mvnx` file exceeding `1,500` frames (such as `P28`'s 13 unsegmented `~39,935`-frame recordings) raises `ULFInADLUnsegmentedRecordingError` unless `segmentation_verified=True`.

### Step 2 — Versioned Input Pipelines & Evaluation of the `100 Hz`, `200`-Frame, `12`-Channel Standard ([`configs/stage10/preprocessing_configs.json`](file:///d:/me/project/configs/stage10/preprocessing_configs.json))
- **Evaluation of the `100 Hz`, `200`-Frame (`2.0 s`), `12`-Channel Standard:**
  - **Directly Compatible at `100 Hz` (`W=200, C=12`):** `PHYTMO` (`Tracks 1C, 2A, 2B`, proper acceleration in `m/s²` + `rad/s`) and `ULTra-MoCap` (`Tracks 3C, 5B`, decimated from `2,000 Hz` to match `100 Hz` OpenSim IK frames, proper acceleration in `m/s²` + `rad/s`).
  - **Why Separate Dataset Adapters / Pipelines Are Retained Where Physically Necessary:**
    1. **Gravity Convention Difference:** `Optotrak vs DOT` (`Tracks 1A, 3A`) records Xsens DOT **free acceleration** (gravity vector already subtracted onboard, $\|\mathbf{a}_{\text{free}}\| \approx 0\text{ m/s}^2$ at rest) at `120 Hz` (`W=240`), whereas `PHYTMO`, `ULTra-MoCap`, and `JU-IMU` record **proper acceleration** ($\|\mathbf{a}\| \approx 9.81\text{ m/s}^2$ at rest).
    2. **Local Coordinate Frame & Calibration Differences:** Each dataset mounts Upper Arm and Forearm IMUs with dataset-specific local axis conventions without a shared N-pose rotation matrix.
    3. **Quaternion-Only Input:** `VIDIMU` (`Track 3E`) provides `50 Hz` (`W=100`) 8-channel unit quaternions without raw 12-channel accelerometer/gyroscope signals.
- All 8 versioned pipeline configurations (`preproc_v8_*` and `preproc_v10_*`) are saved in [`configs/stage10/preprocessing_configs.json`](file:///d:/me/project/configs/stage10/preprocessing_configs.json).

### Step 3 — Four Candidate Architectures Prepared Across Five Core Tasks ([`configs/stage10/task_candidates.json`](file:///d:/me/project/configs/stage10/task_candidates.json))
- Prepared `TCN`, `1D_CNN`, `LSTM`, and `GRU` candidates with task-specific output heads and explicit checkpoint mappings across all 5 tasks (detailed in [`docs/STAGE10_MODEL_DESIGN.md`](file:///d:/me/project/docs/STAGE10_MODEL_DESIGN.md)).

### Step 4 — Three-Tier Range of Motion (`ROM`) & Repetition-Cycle Segmentation ([`src/evaluation/rom_segmentation.py`](file:///d:/me/project/src/evaluation/rom_segmentation.py))
- Implemented `segment_repetition_cycles_rom()` and `compute_three_tier_rom_comparison()` to keep three ROM definitions strictly separate:
  1. **Window-Level ROM (`window_2s_mean_rom_deg`):** Mean $\max - \min$ within 2.0-second windows (matching Stage 8 `rom_mae_deg`).
  2. **Trial-Level ROM (`full_trial_peak_to_trough_rom_deg`):** Global $\max - \min$ across the entire reconstructed trial trajectory.
  3. **Repetition-Level ROM (`repetition_level_mean_rom_deg`):** Mean peak-to-trough excursion across individual segmented movement repetitions with validated `start_time_s -> peak_time_s -> end_time_s` cycle boundaries and `matched_cycle_rom_mae_deg`.

### Step 5 — Supervisor Web Demo Extended for Stage 10 Sample Inference Preview ([`src/inference/engine.py`](file:///d:/me/project/src/inference/engine.py), [`src/inference/app.py`](file:///d:/me/project/src/inference/app.py), [`src/inference/static/`](file:///d:/me/project/src/inference/static/index.html))
- Added `/api/stage10/configs`, `/api/stage10/tasks`, and `/api/stage10/preview` while preserving all existing Stage 9.5 and Stage 9.6 endpoints.
- Added Section 5 (`#stage10-sample-preview-section`) to the web dashboard, displaying full recording metadata, versioned preprocessing config IDs, side-by-side 4-model predictions, segmented repetition cycles, and explicit compatibility guard errors when an unsupported task is requested for a sample.

---

## 3. Roadmap to Stage 10.3 (Awaiting User Approval)

1. **Targeted Stage 10.3 Retraining (Only Where Needed):**
   - Train causal cross-dataset models using `preproc_v10_cross_100hz_inv12_causal_window` (`compute_orientation_invariant_windows_12ch(mode="causal_expanding")`).
   - Optionally train `100 Hz` (`W=200`) resampled `Optotrak vs DOT` checkpoints (`Track 1A-100Hz`, `Track 3A-100Hz`) and `P28`-filtered `ULF_in_ADL` auxiliary checkpoints (`preproc_v10_ulf_60hz_raw12_p28_guarded`).
2. **Real Dual-IMU Recording Evaluation Protocol:**
   - Record synchronized Upper-Arm + Forearm `100 Hz` SI (`m/s²`, `rad/s`) CSV trials with a 2-second static N-pose + $90^\circ$ elbow flexion calibration header and evaluate all 4 architectures side by side.
