# Stage 10.2 — Four-Model Candidate Design & Task Mapping (`docs/STAGE10_MODEL_DESIGN.md`)

**Project:** AI-Based Elbow Rehabilitation Assistance System After Olecranon Fracture ORIF  
**Workspace Root:** `D:\me\project`  
**Config Reference:** [`configs/stage10/task_candidates.json`](file:///d:/me/project/configs/stage10/task_candidates.json) & [`configs/stage10/preprocessing_configs.json`](file:///d:/me/project/configs/stage10/preprocessing_configs.json)  
**Date:** 2026-10-19

---

## 1. Core Architectural Design Principle

All four candidate deep-learning architectures (**`TCN`**, **`1D_CNN`**, **`LSTM`**, and **`GRU`**) in [`src/models/architectures.py`](file:///d:/me/project/src/models/architectures.py) share an identical input tensor interface `(batch_size, window_len, in_channels)` and a unified `TaskOutputHead(feat_dim, task_mode, num_classes, num_reg_targets)`:

1. **`TCN` (`TCNModel`):** 4 dilated residual `TemporalBlock` layers (`hidden_channels=(56, 56, 56, 56)`, `kernel_size=5`, dilations $d \in \{1, 2, 4, 8\}$, `BatchNorm1d + GELU + Dropout(0.2)`), `115.3k–116.8k` parameters (`114.0k` on 8-ch), **`2.22 ms`** single-core CPU latency.
2. **`1D_CNN` (`CNN1DModel`):** 4 non-dilated `Conv1d` blocks (`hidden_channels=(64, 64, 96, 96)`, `kernel_sizes=(7, 5, 5, 3)`, $d=1$, `BatchNorm1d + GELU + Dropout(0.2)`), `85.4k–87.9k` parameters (`83.6k` on 8-ch), **`0.78 ms`** single-core CPU latency.
3. **`LSTM` (`LSTMModel`):** `Linear(in_ch, 80) + GELU` $\rightarrow$ 2-layer unidirectional `nn.LSTM(80, 80, num_layers=2, dropout=0.2)` $\rightarrow$ `LayerNorm(80)`, `105.0k–107.1k` parameters (`104.7k` on 8-ch), **`1.43 ms`** single-core CPU latency.
4. **`GRU` (`GRUModel`):** `Linear(in_ch, 92) + GELU` $\rightarrow$ 2-layer unidirectional `nn.GRU(92, 92, num_layers=2, dropout=0.2)` $\rightarrow$ `LayerNorm(92)`, `104.2k–106.6k` parameters (`103.9k` on 8-ch), **`9.52 ms`** single-core CPU latency.

---

## 2. Candidate Models, Output Heads & Checkpoint Reusability Across the Five Core Tasks

Every task uses a task-specific output head and dataset-matched checkpoint. Single-track checkpoints are **never** forced onto unrelated tasks or incompatible channel/gravity schemas.

| Core Task | Supports All 4 Models (`TCN, 1D_CNN, LSTM, GRU`) on Same Input? | Task-Specific Output Head | Directly Reusable Checkpoints (`models/checkpoints/`) | Preprocessing Config ID | Requires New Training or New Data? |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **Task 1: Elbow Flexion/Extension Angle Prediction (`deg`)** | **YES** (`100%`) | `TaskOutputHead(task_mode="regression", num_reg_targets=2)` $\rightarrow$ target `0` (`elbow_flex_ext_deg`) | • **`track_3c_ultra_mocap_angle_regression_*`** (`100 Hz`, `W=200, C=12`, `TCN` `9.96°` Flex MAE)<br>• **`track_3a_optotrak_elbow_prosup_regression_*`** (`120 Hz`, `W=240, C=12`, `GRU` `12.24°` Flex MAE)<br>• Aux: `track_3e_*` (`50 Hz` 8-ch quat), `track_3b_*` (`100 Hz` StrokeRehab controls) | • `preproc_v8_ultramocap_100hz_raw12_properacc`<br>• `preproc_v8_optotrak_120hz_raw12_freeacc` | • **New Training (Optional):** `100 Hz` (`W=200`) resampled `Optotrak vs DOT` angle model; `P28`-filtered `ULF_in_ADL` `Track 3D` model.<br>• **New Data:** Post-olecranon ORIF patient optical/goniometer reference recordings. |
| **Task 2: Forearm Pronation/Supination Angle Prediction (`deg`)** | **YES** (`100%`) | `TaskOutputHead(task_mode="regression", num_reg_targets=2)` $\rightarrow$ target `1` (`forearm_pro_sup_deg`) | • **`track_3c_ultra_mocap_angle_regression_*`** (`100 Hz`, `W=200, C=12`, `TCN` `10.19°` Pro/Sup MAE)<br>• **`track_3a_optotrak_elbow_prosup_regression_*`** (`120 Hz`, `W=240, C=12`, `GRU` `8.48°` MAE on isolated `Task 12`; `20.14°` across 3D ADLs)<br>• Aux: `track_3e_*` (`50 Hz` 8-ch quat, `TCN` `14.29°` MAE) | • `preproc_v8_ultramocap_100hz_raw12_properacc`<br>• `preproc_v8_optotrak_120hz_raw12_freeacc` | • **Biomechanical Requirement:** 6-axis IMUs require elbow flexed $\sim 90^\circ$ or a session-start N-pose + $90^\circ$ flexion functional heading calibration for axial rotation observability. |
| **Task 3: Full-Repetition ROM Estimation (`deg`)** | **YES** (`100%`) | Continuous angle trajectory (`TaskOutputHead("regression")`) $\rightarrow$ [`compute_three_tier_rom_comparison()`](file:///d:/me/project/src/evaluation/rom_segmentation.py) | • **`track_3c_ultra_mocap_angle_regression_*`** (`100 Hz`, `W=200, C=12`)<br>• **`track_3a_optotrak_elbow_prosup_regression_*`** (`120 Hz`, `W=240, C=12`)<br>• Aux: `track_3e_*` (`50 Hz` 8-ch quat) | • `preproc_v8_ultramocap_100hz_raw12_properacc`<br>• `preproc_v8_optotrak_120hz_raw12_freeacc` | • **Ready Now:** All 4 models output 3-tier ROM (2.0s Window ROM, Full-Trial ROM, and Segmented Per-Repetition Cycle ROM) via `src/evaluation/rom_segmentation.py`. |
| **Task 4: Exercise Classification** | **YES** (`100%`) | `TaskOutputHead(task_mode="classification", num_classes=C)` ($C \in \{3, 7, 10, 14\}$) | • **`track_1c_phytmo_3class_exercise_*`** (`100 Hz`, `W=200, C=12`, `TCN` `0.9400` Macro-F1)<br>• **`track_1a_optotrak_10class_exercise_*`** (`120 Hz`, `W=240, C=12`, `1D_CNN` `0.9808` Macro-F1)<br>• **`track_1b_ju_imu_14class_exercise_*`** (`80 Hz`, `W=160, C=12`, `TCN` `0.8030` Macro-F1)<br>• Aux Offline: `track_5a1_*` (`7-class`), `track_5a2_*` (`2-class`), `track_5b_*` (`3-class`) | • `preproc_v8_phytmo_100hz_raw12_properacc`<br>• `preproc_v8_optotrak_120hz_raw12_freeacc`<br>• `preproc_v8_juimu_80hz_raw12_properacc` | • **New Training:** Retrain cross-dataset `Tracks 5A-1 & 5A-2` using causal window-local features (`preproc_v10_cross_100hz_inv12_causal_window`) for real-time streaming. |
| **Task 5: Correct vs. Incorrect Movement Classification** | **YES** (`100%` on `PHYTMO`) | `TaskOutputHead(task_mode="classification", num_classes=2 or 6)` | • **`track_2a_phytmo_efe_quality_binary_*`** (`100 Hz`, `W=200, C=12`, `TCN` `0.8925` Macro-F1, `92.2%` incorrect-execution recall)<br>• **`track_2b_phytmo_6class_exercise_quality_*`** (`100 Hz`, `W=200, C=12`, `LSTM` `0.8356` Macro-F1) | • `preproc_v8_phytmo_100hz_raw12_properacc` | • **New Data Required for Expansion:** `PHYTMO` is the only dataset with valid execution-quality labels (simulated healthy compensatory movements). Extending quality assessment to Forearm Pronation/Supination or post-ORIF surgical patients requires clinician-annotated data. |

---

## 3. Three-Tier Range of Motion (`ROM`) & Repetition-Cycle Segmentation Design

Implemented in [`src/evaluation/rom_segmentation.py`](file:///d:/me/project/src/evaluation/rom_segmentation.py), our ROM pipeline strictly separates three biomechanical quantities on every reconstructed trajectory $\hat{\theta}(t)$ (`deg`):

1. **Tier 1 — Window-Level ROM (`window_2s_mean_rom_deg`):**
   $$\text{ROM}_{\text{win}} = \frac{1}{N_w} \sum_{w=1}^{N_w} \left( \max_{t \in W_w} \hat{\theta}(t) - \min_{t \in W_w} \hat{\theta}(t) \right)$$
   Matches the Stage 8 benchmark `rom_mae_deg` definition over 2.0-second windows.
2. **Tier 2 — Trial-Level ROM (`full_trial_peak_to_trough_rom_deg`):**
   $$\text{ROM}_{\text{trial}} = \max_{t \in [0, T]} \hat{\theta}(t) - \min_{t \in [0, T]} \hat{\theta}(t)$$
   Measures total excursion across the complete recording.
3. **Tier 3 — Repetition-Level ROM (`repetition_level_mean_rom_deg`):**
   - Uses Savitzky-Golay prominence peak/trough localization (`segment_repetition_cycles_rom()`) with physiological minimum prominence ($\ge 10^\circ$ for elbow flexion, $\ge 8^\circ$ for forearm pronation/supination) and minimum inter-peak spacing ($\ge 0.6\text{ s}$) to segment individual trough-to-peak-to-trough movement cycles $k \in \{1 \dots K\}$.
   - Reports per-cycle boundaries (`start_time_s`, `peak_time_s`, `end_time_s`, `duration_sec`), per-cycle angular range `[min_angle_deg, max_angle_deg]`, per-cycle excursion `rom_deg`, mean repetition ROM $\frac{1}{K}\sum_{k=1}^K \text{ROM}_k$, and `matched_cycle_rom_mae_deg` against ground-truth repetition boundaries.
