# Final Model Selection & Deployment Strategy (`docs/FINAL_MODEL_SELECTION.md`)

**Project:** AI-Based Elbow Rehabilitation Assistance System (Post-Olecranon ORIF Recovery)  
**Stage:** Stage 9 — Task-Specific Architecture Selection & Trade-Off Analysis  
**Candidate Architectures Compared:** `TCN`, `1D_CNN` (`1D CNN`), `LSTM`, `GRU`  
**Audit Reference:** [`experiments/reports/STAGE9_AUDIT_REPORT.md`](file:///d:/me/project/experiments/reports/STAGE9_AUDIT_REPORT.md) & [`docs/FINAL_DATASET_SELECTION.md`](file:///d:/me/project/docs/FINAL_DATASET_SELECTION.md)

---

## 1. Core Selection Principles

1. **Task-Specific Selection Over Blind Uniformity:** Our 14-track benchmark proved that feed-forward dilated convolutional networks (`TCN`), standard local convolutional networks (`1D_CNN`), and gated recurrent networks (`LSTM`, `GRU`) have distinct inductive biases across continuous kinematic regression, fine execution-error detection, clinical motor-impairment grading, and cross-domain transfer. Therefore, we select the **optimal architecture separately for each intended task**.
2. **No 4-Model Ensembling:** Combining `TCN + 1D_CNN + LSTM + GRU` into a 4-model ensemble would quadruple flash/RAM footprint (`~420 KB` $\rightarrow$ `~1.7 MB`) and increase single-core CPU inference latency from `1.4–4.3 ms` to `~16 ms` without solving the primary physical error source (uncalibrated forearm axial heading offset during 3D multi-joint movements). Single task-specialized models are recommended.
3. **Audit-Corrected Evidence Only:** All recommendations incorporate the Stage 9 forensic audit corrections (including subject-equalized evaluation on `ULF_in_ADL` after identifying `P28`'s unsegmented files, single-core latency profiling, and realistic 7-class cross-dataset transfer on `Track 5A-1`).

---

## 2. Task-by-Task Architecture Evaluation & Selection

### Task 1: Continuous Elbow Flexion/Extension Angle Estimation ($\theta_{\text{flex}}(t)$ in `deg`)
- **Recommended Primary Architecture:** **`TCN`** (with **`GRU`** as close co-leader on `Optotrak vs DOT` `Track 3A`)
- **Supporting Evidence:**
  - **`ULTra-MoCap` (`Track 3C`, 100 Hz OpenSim IK across 4 speeds):** **`TCN` ranks #1** with **Elbow Flexion MAE = `9.96°`** (`RMSE = 12.85°`, **$R^2 = 0.8876, r = 0.9510$**), beating `GRU` (`10.06°`), `1D_CNN` (`10.88°`), and `LSTM` (`11.96°`).
  - **`StrokeRehab` (`Track 3B`, 100 Hz 9 Complex ADLs):** **`TCN` ranks #1** with **Elbow Flexion MAE = `13.52°`** (`r = 0.7424`), beating `LSTM` (`14.60°`), `1D_CNN` (`15.77°`), and `GRU` (`16.05°`).
  - **`ULF_in_ADL` (`Track 3D`, 60 Hz Stroke & Healthy ADLs):** Under Stage 9 audit-corrected **Subject-Equalized MAE** across all 4 test subjects (`H04, P08, P19, P28`), **`TCN` ranks #1** (`18.74°` Elbow Flexion MAE vs `19.45°` `GRU`, `19.93°` `LSTM`, `24.68°` `1D_CNN`).
  - **`Optotrak vs DOT` (`Track 3A`, 120 Hz Optical MoCap across 10 tasks):** **`GRU` ranks #1** (`12.24°` Elbow Flexion MAE, $R^2 = 0.8103, r = 0.9088$; `6.17°` MAE on isolated `forearm_pro_sup` and `12.32°` on isolated `elbow_flex_ext`), with `LSTM` (`12.58°`), `1D_CNN` (`12.91°`), and `TCN` (`13.28°`, $R^2 = 0.8014$) all within `1.04°` of each other.
- **Why `TCN` Wins Overall for Flexion-Extension:** Dilated residual 1D convolutions (`d = 1, 2, 4, 8`, receptive field covering the 2.0-second window) act as a learnable zero-phase complementary filter fusing low-frequency gravity inclination ($\text{Acc}$) with high-frequency angular rate integration ($\text{Gyr}$) without recurrent hidden-state drift.

---

### Task 2: Continuous Forearm Pronation/Supination Angle Estimation ($\theta_{\text{pro/sup}}(t)$ in `deg`)
- **Recommended Primary Architecture:** **`TCN`** (for multi-speed & quaternion/IK tracking) or **`GRU`** (for `Optotrak vs DOT` raw 6-DOF tracking)
- **Supporting Evidence:**
  - **`ULTra-MoCap` (`Track 3C`):** **`TCN` ranks #1** with **Pronation-Supination MAE = `10.19°`** ($R^2 = 0.8869, r = 0.9484$), outperforming `GRU` (`11.23°`), `LSTM` (`11.58°`), and `1D_CNN` (`12.55°`).
  - **`VIDIMU` (`Track 3E`, 8-Ch Quaternions):** **`TCN` ranks #1** with **Pronation-Supination MAE = `14.29°`** ($R^2 = 0.8106, r = 0.9014$), outperforming `1D_CNN` (`17.03°`), `LSTM` (`17.11°`), and `GRU` (`17.34°`).
  - **`Optotrak vs DOT` (`Track 3A`):** **`GRU` ranks #1** (`20.14°` across all 10 multi-joint tasks; **`8.48°` MAE during isolated Task 12 `forearm_pro_sup`**), followed by `TCN` (`20.70°`), `LSTM` (`20.96°`), and `1D_CNN` (`22.45°`).
- **Critical Biomechanical Error Characteristic:** During isolated forearm rotation with the elbow flexed at $90^\circ$ (`Task 12`), gravity is perpendicular to the forearm shaft and 6-DOF IMU models track pronation-supination within **`8.48°–10.19°` MAE**. However, during 3D shoulder-reaching tasks (`box_off_shelf`, `shoulder_flex`), axial rotation around the gravity vector is unobservable from accelerometers alone, raising MAE to `20°–30°` across all four neural architectures unless a session-start functional calibration pose is locked.

---

### Task 3: Range-of-Motion (`ROM`) Measurement ($\text{ROM} = \max \theta - \min \theta$ in `deg`)
- **Recommended Primary Architecture:** **`TCN`** (with **`GRU`** co-recommended on `Optotrak vs DOT`)
- **Supporting Evidence:**
  - **`TCN` achieves the lowest Window ROM MAE on 3 of 5 regression tracks** (`ULTra-MoCap` Track 3C: **`10.95°` Overall ROM MAE / `8.31°` Elbow Flexion ROM MAE**; `VIDIMU` Track 3E: **`8.73°` Overall ROM MAE / `8.06°` Elbow Flexion ROM MAE**; `StrokeRehab` Track 3B: **`17.89°` Overall ROM MAE / `11.50°` Elbow Flexion ROM MAE**).
  - **`GRU` achieves the lowest Window ROM MAE on `Optotrak vs DOT` Track 3A (`14.30°` Overall / `13.65°` Elbow Flexion ROM MAE)** and `ULF_in_ADL` Track 3D (`12.63°` Overall ROM MAE).
  - By contrast, **`1D_CNN` exhibits severe ROM inflation (`20°–33°` ROM MAE)** because its small non-dilated receptive field produces high-frequency spikes at window edges that inflate $\max \hat{\theta} - \min \hat{\theta}$.

---

### Task 4: Rehabilitation Exercise Classification
- **Recommended Primary Architecture:** **`TCN`** (for multi-exercise & mixed healthy/impaired cohorts) or **`1D_CNN`** (for ultra-low-latency healthy exercise classification)
- **Supporting Evidence:**
  - **In-Domain Rehabilitation Benchmarks:**
    - `PHYTMO` 3-Class (`Track 1C`, `EFE, EAH, SQZ`): **`TCN` ranks #1 (`0.9400` Macro-F1, `0.9429` BalAcc)**, followed by `1D_CNN` (`0.9333`), `LSTM` (`0.9289`), and `GRU` (`0.9258`).
    - `JU-IMU` 14-Class (`Track 1B`, `41` subjects): **`TCN` ranks #1 (`0.8030` Macro-F1 overall; `97.56%` accuracy / `0.9726` Macro-F1 on `ND` cohort)**, followed by `GRU` (`0.7973`), `LSTM` (`0.7935`), and `1D_CNN` (`0.7820`).
    - `Optotrak vs DOT` 10-Class (`Track 1A`, `15` subjects): **`1D_CNN` ranks #1 (`0.9808` Macro-F1, `0.9825` BalAcc, `0.78 ms` single-core latency)**, with `LSTM` (`0.9764`), `GRU` (`0.9753`), and `TCN` (`0.9682`) all exceeding `96.8%` Macro-F1.
  - **Cross-Dataset & Cross-Speed Exercise Generalization:**
    - When classifying across unseen sensor hardware (`Track 5A-1`: `JU-IMU -> Optotrak vs DOT` 7-Class) or unseen execution speeds (`Track 5B`: `Normal -> Slow, Fast, VeryFast`), **`LSTM` ranks #1 (`0.7392` and `0.9113` Macro-F1)** because recurrent gating adapts better to temporal time-warping across speeds (`Slow` vs `Fast`).

---

### Task 5: Correct vs. Incorrect Exercise Execution Classification
- **Recommended Primary Architecture:**
  - **Binary Elbow Flexion-Extension Execution Quality (`EFE0` vs `EFE1`, `Track 2A`):** **`TCN`**
  - **Joint 6-Class Multi-Exercise + Execution Quality (`EFE0..SQZ1`, `Track 2B`):** **`LSTM`**
- **Supporting Evidence:**
  - On **`Track 2A` (`PHYTMO` `EFE0` Correct vs `EFE1` Incorrect across 27 subjects)**—the exact target exercise for elbow rehabilitation—**`TCN` ranks #1 (`0.8925` Test Macro-F1, `0.8939` Balanced Accuracy, `0.8113` Val Macro-F1)**, achieving **`92.20%` sensitivity/recall on `EFE1 (Incorrect)`** and `92.42%` precision on `EFE0 (Correct)`, outperforming `GRU` (`0.8829`), `LSTM` (`0.8819`), and `1D_CNN` (`0.8135`).
  - On **`Track 2B` (`PHYTMO` 6-Class `EFE0..SQZ1` across 30 subjects)**, **`LSTM` ranks #1 on the held-out test set (`0.8356` Macro-F1, `0.8385` BalAcc)** (`TCN` had the highest validation Macro-F1 at `0.7094` vs `0.6768` for `LSTM`, indicating mild validation-to-test subject variance across the 5 validation vs 5 test subjects).

---

### Task 6: Stroke-Related / Impaired Upper-Limb Movement Analysis
- **Recommended Primary Architecture:** **`GRU`** (with **`1D_CNN`** as ultra-fast runner-up on `Track 4A`)
- **Supporting Evidence:**
  - **`Track 4A` (`JU-IMU` `Stroke` vs `Non-Disabled` across 41 subjects):** **`GRU` ranks #1 (`0.8733` Test Macro-F1, `0.8990` Balanced Accuracy, `95.05%` Stroke recall)**, closely followed by **`1D_CNN` (`0.8653` Macro-F1, `0.8732` BalAcc, `0.71 ms` latency)**, both substantially outperforming `TCN` (`0.7683`) and `LSTM` (`0.7278`).
  - **`Track 4C` (`REHAB-120` 3-Class Clinical `FMA-UE` Item Score `0, 1, 2` — Session-Split Caveat):** **`GRU` ranks #1 (`0.8051` Macro-F1, `0.8036` BalAcc, `82.50%` Overall Accuracy)**, ahead of `LSTM` (`0.7819`), `TCN` (`0.7720`), and `1D_CNN` (`0.6956`).

---

## 3. Deployment Trade-Off Analysis: Multi-Architecture Specialist Strategy vs. Single Shared Backbone (`TCN`)

For the future wearable elbow rehabilitation system, two deployment strategies are viable:

### Option A (Recommended for Maximum Task Accuracy): Task-Specialized Architectures
Deploy the top-performing validated architecture per module (`~840 KB` total flash for 2 checkpoints, `< 4 ms` combined latency):
1. **`TCN` (`478 KB`, `2.9 ms` multi-core / `4.3 ms` single-core latency)** for **Continuous Elbow Flexion-Extension & Pronation-Supination Angle + ROM Estimation** (`Track 3C/3A`) **AND** **Binary Elbow Flexion-Extension Execution Quality (`EFE0 vs EFE1`, `Track 2A`)**.
2. **`LSTM` (`418 KB`, `1.4 ms` single-core latency)** for **Multi-Speed / Cross-Dataset Exercise & 6-Class Quality Classification** (`Tracks 2B, 5A-1, 5B`), **OR `GRU` (`414 KB`)** when running offline clinical **Stroke / `FMA-UE` Impairment Screening** (`Tracks 4A, 4C`).

### Option B (Single Shared Architecture for Firmware Simplicity): All-`TCN` Backbone
If embedded firmware maintenance requires compiling and maintaining **only a single neural architecture family** on the microcontroller/edge device, **`TCN` is the best single shared architecture**:
- **Advantages of All-`TCN`:**
  - Pure feed-forward `Conv1d + BatchNorm1d + GELU` graph (100% compatible with CMSIS-NN / TFLite Micro / ONNX Runtime without recurrent state management).
  - Ranks **#1** on `PHYTMO` 3-class exercise (`0.9400` F1), **#1** on `JU-IMU` 14-class exercise (`0.8030` F1), **#1** on `PHYTMO` binary `EFE` execution quality (`0.8925` F1), and **#1** on 4 of 5 continuous angle & ROM regression tracks (`Tracks 3B, 3C, 3D [subject-equalized], 3E`).
- **Quantified Trade-Offs of Choosing All-`TCN` Over Task Specialists:**
  - On `Optotrak vs DOT` (`Track 3A`), `TCN` trades **`0.80°` higher overall MAE** (`16.99°` vs `16.19°` for `GRU`; `13.28°` vs `12.24°` on Elbow Flexion).
  - On `PHYTMO` 6-class joint quality (`Track 2B`), `TCN` trades **`0.070` lower Test Macro-F1** (`0.7655` vs `0.8356` for `LSTM`, though `TCN` had higher validation Macro-F1 [`0.7094` vs `0.6768`]).
  - On 7-class cross-dataset transfer (`Track 5A-1`), `TCN` trades **`0.034` lower Macro-F1** (`0.7056` vs `0.7392` for `LSTM`).
  - On clinical stroke vs. ND screening (`Track 4A`), `TCN` trades **`0.105` lower Macro-F1** (`0.7683` vs `0.8733` for `GRU`).

---

## 4. Master Final Summary Table (Tasks, Selected Datasets, Recommended Architectures, Evidence & Limitations)

| Intended Rehabilitation Task | Selected Primary Dataset(s) | Selected Secondary Dataset(s) | Excluded Datasets (For This Task) | Recommended Architecture (Runner-Up) | Validated Supporting Evidence (Held-Out Test Set) | Remaining Scientific & Clinical Limitations |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Continuous Elbow Flexion/Extension Angle Estimation ($\theta_{\text{flex}}$)** | **`Optotrak vs DOT`** (`Track 3A`, 14 subj, 120 Hz Optical GT) | **`ULTra-MoCap`** (`Track 3C`, 13 subj, 100 Hz OpenSim IK); **`ULF_in_ADL`** (`Track 3D`, filtered $\le 1500$ frames) | `PHYTMO` (optical occlusion NaNs), `JU-IMU` (no angle GT), `StrokeRehab` (no raw gyro), `VIDIMU` (quats only) | **`TCN`** (`115k` params, `2.9–4.3 ms`)<br>*Co-leader on Track 3A:* **`GRU`** | • **`Track 3C` (`ULTra-MoCap`):** **`TCN` = `9.96°` MAE** ($R^2=0.888, r=0.951$)<br>• **`Track 3A` (`Optotrak`):** **`GRU` = `12.24°` MAE** ($R^2=0.810, r=0.909$), **`TCN` = `13.28°` MAE** ($R^2=0.801$)<br>• **`Track 3D` (`ULF_in_ADL` Subj-Eq):** **`TCN` = `18.74°` MAE** (#1) | Evaluated on healthy volunteers (`Optotrak`, `ULTra-MoCap`) and chronic stroke (`ULF_in_ADL`); requires prospective validation on post-olecranon ORIF surgical patients. |
| **2. Continuous Forearm Pronation/Supination Angle Estimation ($\theta_{\text{pro/sup}}$)** | **`Optotrak vs DOT`** (`Track 3A`, 14 subj, 120 Hz Optical GT) | **`ULTra-MoCap`** (`Track 3C`, 13 subj, 100 Hz OpenSim IK) | `PHYTMO`, `JU-IMU`, `StrokeRehab` (wrist supination without raw gyro), `REHAB-120` | **`TCN`** (Multi-speed IK) / **`GRU`** (`Optotrak` raw 6-DOF) | • **`Track 3C` (`ULTra-MoCap`):** **`TCN` = `10.19°` MAE** ($R^2=0.887, r=0.948$)<br>• **`Track 3A` (`Optotrak` Task 12 isolated pro/sup):** **`GRU` = `8.48°` MAE**<br>• **`Track 3A` (All 10 3D tasks):** **`GRU` = `20.14°` MAE**, **`TCN` = `20.70°` MAE** | 6-axis IMUs cannot observe axial rotation around the gravity vector during complex 3D shoulder reaching (`20°–30°` MAE without session-start N-pose calibration). |
| **3. Range-of-Motion (`ROM`) Measurement (`deg`)** | **`Optotrak vs DOT`** (`Track 3A`) | **`ULTra-MoCap`** (`Track 3C`); **`VIDIMU`** (`Track 3E` if quat input used) | `PHYTMO`, `JU-IMU`, `REHAB-120`, `1D_CNN` model (high edge-spike ROM error) | **`TCN`**<br>*Co-leader on Track 3A:* **`GRU`** | • **`Track 3C` (`ULTra-MoCap` Flexion ROM):** **`TCN` = `8.31°` MAE** (`10.95°` overall)<br>• **`Track 3E` (`VIDIMU` Flexion ROM):** **`TCN` = `8.06°` MAE** (`8.73°` overall)<br>• **`Track 3A` (`Optotrak` Flexion ROM):** **`GRU` = `13.65°` MAE** (`14.30°` overall) | Stage 8 measured **2.0-second window ROM** ($\max_{t \in W} \theta - \min_{t \in W} \theta$); final clinical software must also compute full-repetition peak-to-trough ROM. |
| **4. Rehabilitation Exercise Classification** | **`Optotrak vs DOT`** (`Track 1A`, 10-cls); **`JU-IMU`** (`Track 1B`, 14-cls); **`PHYTMO`** (`Track 1C`, 3-cls) | **`ULTra-MoCap`** (`Track 5B` speed variation) | `StrokeRehab` (unscripted ADL primitives), `VIDIMU` (no raw Acc/Gyr), `REHAB-120` | **`TCN`** (In-domain multi-exercise)<br>**`1D_CNN`** (Ultra-fast `0.78 ms`)<br>**`LSTM`** (Cross-dataset / cross-speed) | • **`Track 1C` (`PHYTMO` 3-cls):** **`TCN` = `0.9400` Macro-F1**<br>• **`Track 1B` (`JU-IMU` 14-cls):** **`TCN` = `0.8030` Macro-F1** (`97.56%` Acc on `ND`)<br>• **`Track 1A` (`Optotrak` 10-cls):** **`1D_CNN` = `0.9808` Macro-F1** (`TCN` = `0.9682`)<br>• **`Track 5A-1` (7-cls Cross-Dataset):** **`LSTM` = `0.7392` Macro-F1** | Severe motor impairment (`Stroke6` in `JU-IMU`) reduces 14-class exercise classification accuracy to `35.0%` due to pathological movement synergies. |
| **5. Correct vs. Incorrect Exercise Execution Classification** | **`PHYTMO`** (`Track 2A` Binary `EFE0 vs EFE1`, 27 subj; `Track 2B` 6-cls `EFE0..SQZ1`, 30 subj) | *None* (No other dataset has valid subject-independent form-error labels) | `UCI Biceps Curl` (fatal subject-class confounding in 4,024-row excerpt), all other datasets | **`TCN`** (Binary `EFE0 vs EFE1` `Track 2A`)<br>**`LSTM`** (6-Class `EFE0..SQZ1` `Track 2B`) | • **`Track 2A` (`EFE0 vs EFE1`):** **`TCN` = `0.8925` Macro-F1** (`0.8939` BalAcc, **`92.20%` recall on incorrect execution `EFE1`**)<br>• **`Track 2B` (`6-Class Joint`):** **`LSTM` = `0.8356` Macro-F1** (`1D_CNN` = `0.8253`, `TCN` = `0.7094` Val / `0.7655` Test) | `PHYTMO` errors are simulated compensatory movements by healthy volunteers, **not** post-surgical ORIF structural instability or pain-avoidance compensations. |
| **6. Stroke-Related / Impaired Movement Analysis** | **`JU-IMU`** (`Track 4A`, `27 ND` vs `14 Stroke`, 41 subj) | **`REHAB-120`** (`Track 4C`, 3-class `FMA-UE` `0, 1, 2` — *Session-Split Caveat*) | `StrokeRehab` (stroke branch missing on disk), `ULF_in_ADL` Track 4B (`P28` unsegmented files + healthy/non-paretic overlap) | **`GRU`**<br>*(Fast alternative on Track 4A: **`1D_CNN`**)* | • **`Track 4A` (`JU-IMU` Stroke vs ND):** **`GRU` = `0.8733` Macro-F1** (`0.8990` BalAcc, **`95.05%` Stroke recall**); **`1D_CNN` = `0.8653` Macro-F1** (`0.71 ms` latency)<br>• **`Track 4C` (`REHAB-120` FMA-UE):** **`GRU` = `0.8051` Macro-F1** (`82.50%` Acc) | `REHAB-120` (`Track 4C`) is session-disjoint rather than guaranteed patient-disjoint because local patient-to-session IDs are absent; `ULF_in_ADL` Track 4B (`0.3620` F1) is excluded. |
