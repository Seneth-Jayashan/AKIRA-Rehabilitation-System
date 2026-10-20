# Stage 2: Four-Architecture Benchmark & Evaluation Report

**Project:** AI-Based Elbow Rehabilitation Assistance System (Post-Olecranon ORIF Recovery)  
**Workspace:** `D:\me\project`  
**Stage:** **STAGE 2 — TRAIN AND COMPARE FOUR AI MODELS**  
**Evaluation Standard:** Subject-Independent Holdout (~80% Train+Val / ~20% Test, Val strictly from Train)  
**Architectures Benchmarked:** Temporal Convolutional Network (**TCN**), 1D Convolutional Neural Network (**1D-CNN**), Long Short-Term Memory (**LSTM**), Gated Recurrent Unit (**GRU**)  
**Preservation Status:** All raw datasets, 64 original Stage 8 checkpoints, reports, and demo backend remain strictly preserved. New Stage 2 assets are isolated in `experiments/stage2/` and `models/stage2/`.

---

## 1. Executive Summary & Overall Architecture Leaderboard

In Stage 2, we conducted a rigorous, fair, side-by-side benchmark of the four non-negotiable deep-learning architectures across all approved elbow rehabilitation tasks using identical subject-independent splits, train-only normalization, and identical input window dimensions $(B, W, C)$.

### Master Stage 2 Task-Architecture Winner Matrix:

| Core Clinical Task | Benchmark Track & Dataset | Input Shape | TCN | 1D-CNN | LSTM | GRU | Winning Model | Primary Metric |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Task 1: Elbow Flex/Ext Angle Regression** | `Track 3C` ([ULTra-MoCap](file:///d:/me/project/datasets/raw/ULTra-MoCap)) | $(200, 12)$ | **10.07°** | 11.72° | 11.77° | 10.65° | **TCN** | **MAE = 10.07° ($R^2=0.8873$)** |
| **Task 2: Forearm Pro/Sup Angle Regression** | `Track 3A` ([Optotrak vs DOT](file:///d:/me/project/datasets/raw/Optotrak%20vs%20DOT)) | $(240, 12)$ | 20.70° | 22.45° | 20.96° | **20.14°** | **GRU** | **MAE = 20.14° ($R^2=0.6120$)** |
| **Task 3: Range of Motion (ROM) Estimation** | `Track 3C` ([ULTra-MoCap](file:///d:/me/project/datasets/raw/ULTra-MoCap)) | $(200, 12)$ | **8.31°** | 9.92° | 10.05° | 9.14° | **TCN** | **Flexion ROM MAE = 8.31°** |
| **Task 4: Rehab Exercise Classification** | `Track 1C` ([PHYTMO](file:///d:/me/project/datasets/raw/PHYTMO)) | $(200, 12)$ | **0.9400** | 0.9333 | 0.9289 | 0.9258 | **TCN** | **Macro-F1 = 0.9400 (94.02% Acc)** |
| **Task 4: Functional 10-Class Movement** | `Track 1A` ([Optotrak vs DOT](file:///d:/me/project/datasets/raw/Optotrak%20vs%20DOT)) | $(240, 12)$ | 0.9682 | **0.9808** | 0.9764 | 0.9753 | **1D-CNN** | **Macro-F1 = 0.9808 (98.11% Acc)** |
| **Task 4: ROM 14-Class Movement** | `Track 1B` ([JU-IMU](file:///d:/me/project/datasets/raw/JU-IMU)) | $(160, 12)$ | **0.8030** | 0.7820 | 0.7935 | 0.7973 | **TCN** | **Macro-F1 = 0.8030 (80.12% Acc)** |
| **Task 5: Binary Form Quality (EFE0/1)** | `Track 2A` ([PHYTMO](file:///d:/me/project/datasets/raw/PHYTMO)) | $(200, 12)$ | **0.8925** | 0.8135 | 0.8819 | 0.8829 | **TCN** | **Macro-F1 = 0.8925 (92.2% Rec)** |
| **Task 5: Joint 6-Class Exercise $\times$ Quality** | `Track 2B` ([PHYTMO](file:///d:/me/project/datasets/raw/PHYTMO)) | $(200, 12)$ | 0.7655 | 0.8253 | **0.8356** | 0.8081 | **LSTM** | **Macro-F1 = 0.8356 (83.82% Acc)** |

---

## 2. Preflight Audit Findings

Before executing evaluation and benchmarking, [`generate_stage2_preflight_report()`](file:///d:/me/project/src/training/stage2_runner.py#L125-L215) verified all dataset partitions and model shapes:
- **Status:** **`PREFLIGHT_PASSED`**
- **Data Leakage Detected:** **`False`** (Pairwise disjoint partitions across all tasks).
- **Partition Ratio:** Train + Validation accounts for $78.6\%\text{ to }85.4\%$ of subjects; Test accounts for $14.6\%\text{ to }21.4\%$.
- **Model Parameter Budgets:**
  - TCN: 98,382 parameters (4 dilated residual blocks, channels: 56, dilations: 1, 2, 4, 8)
  - 1D-CNN: 105,482 parameters (4 conv blocks, channels: 64, 64, 96, 96, kernels: 7, 5, 5, 3)
  - LSTM: 109,722 parameters (2 layers, hidden dimension: 80, dropout: 0.2)
  - GRU: 109,474 parameters (2 layers, hidden dimension: 92, dropout: 0.2)

---

## 3. Detailed Task-by-Task Benchmark Results

### 3.1 Task 1: Continuous Elbow Flexion/Extension Angle Regression
- **Dataset:** `ULTra-MoCap` (100 Hz, Delsys Trigno IMU 4 Upper Arm + IMU 1 Forearm; Vicon OpenSim Inverse Kinematics Ground Truth).
- **Test Subjects (Held-Out):** `P06`, `P12` (2 subjects, 32 trials, 1,380 evaluation windows).
- **Target:** `elbow_flex_ext_deg` ($0^\circ\text{ to }140^\circ$).

| Architecture | Elbow Flex MAE | Overall MAE | Overall RMSE | Overall $R^2$ | ROM MAE | Latency (CPU) | Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **TCN (Winner)** | **9.96°** | **10.07°** | **13.08°** | **0.8873** | **8.31°** | 2.14 ms | [`track_3c_TCN_best.pt`](file:///d:/me/project/models/stage2/checkpoints/track_3c_ultra_mocap_angle_regression_TCN_best.pt) |
| **GRU** | 10.58° | 10.65° | 13.92° | 0.8724 | 9.14° | 1.86 ms | [`track_3c_GRU_best.pt`](file:///d:/me/project/models/stage2/checkpoints/track_3c_ultra_mocap_angle_regression_GRU_best.pt) |
| **1D-CNN** | 11.64° | 11.72° | 15.11° | 0.8496 | 9.92° | 1.62 ms | [`track_3c_1D_CNN_best.pt`](file:///d:/me/project/models/stage2/checkpoints/track_3c_ultra_mocap_angle_regression_1D_CNN_best.pt) |
| **LSTM** | 11.82° | 11.77° | 15.24° | 0.8471 | 10.05° | 2.38 ms | [`track_3c_LSTM_best.pt`](file:///d:/me/project/models/stage2/checkpoints/track_3c_ultra_mocap_angle_regression_LSTM_best.pt) |

*Clinical Insight:* **TCN is the clear winner for continuous elbow flexion/extension**, achieving $<10^\circ$ MAE and $R^2 = 0.8873$. Its exponential receptive field enables it to capture smooth continuous trajectory reversals without drift.

---

### 3.2 Task 2: Continuous Forearm Pronation/Supination Angle Regression
- **Dataset:** `Optotrak vs DOT` (120 Hz, Xsens DOT Upper Arm + Forearm; Optotrak Certus 3D Optical Ground Truth).
- **Test Subjects (Held-Out):** `subject 5`, `subject 11`, `subject 15` (3 subjects, 30 trials, 638 evaluation windows).
- **Target:** `forearm_pro_sup_deg` (ISB physiological rotation: $-62^\circ\text{ to }+69^\circ$).

| Architecture | Forearm Pro/Sup MAE | Overall MAE | Overall RMSE | Overall $R^2$ | ROM MAE | Latency (CPU) | Checkpoint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **GRU (Winner)** | **18.42°** | **20.14°** | **25.86°** | **0.6120** | **17.21°** | 1.95 ms | [`track_3a_GRU_best.pt`](file:///d:/me/project/models/stage2/checkpoints/track_3a_optotrak_elbow_prosup_regression_GRU_best.pt) |
| **TCN** | 19.12° | 20.70° | 26.44° | 0.5944 | 17.85° | 2.21 ms | [`track_3a_TCN_best.pt`](file:///d:/me/project/models/stage2/checkpoints/track_3a_optotrak_elbow_prosup_regression_TCN_best.pt) |
| **LSTM** | 19.35° | 20.96° | 26.81° | 0.5830 | 18.04° | 2.45 ms | [`track_3a_LSTM_best.pt`](file:///d:/me/project/models/stage2/checkpoints/track_3a_optotrak_elbow_prosup_regression_LSTM_best.pt) |
| **1D-CNN** | 21.05° | 22.45° | 28.52° | 0.5283 | 19.42° | 1.70 ms | [`track_3a_1D_CNN_best.pt`](file:///d:/me/project/models/stage2/checkpoints/track_3a_optotrak_elbow_prosup_regression_1D_CNN_best.pt) |

*Clinical Insight:* **GRU achieves the highest accuracy on forearm pronation/supination** ($18.42^\circ$ MAE across all 10 multi-joint ADL tasks; $<9.0^\circ$ MAE on isolated pronation/supination Task 12). Pronation/supination error is higher than flexion/extension across all architectures due to the well-known biomechanical heading unobservability when the forearm aligns with gravity.

---

### 3.3 Task 3: Range of Motion (ROM) Estimation
- **Metrics Evaluated:** Tier 1 (Window ROM), Tier 2 (Trial Peak-to-Trough ROM), Tier 3 (Repetition-Cycle Segmented ROM).
- On ULTra-MoCap elbow flexion, **TCN achieved an overall ROM MAE of $8.31^\circ$**, outperforming GRU ($9.14^\circ$), 1D-CNN ($9.92^\circ$), and LSTM ($10.05^\circ$).
- On isolated elbow flexion cycles, TCN repetition-cycle ROM error is **$6.24^\circ$**, well within the clinical threshold of $\le 10^\circ$ for physical therapy goniometry.

---

### 3.4 Task 4: Rehabilitation Movement Classification
#### A. PHYTMO 3-Class Rehabilitation Exercises (`EFE, EAH, SQZ` at 100 Hz):
- **Test Subjects (Held-Out):** `A03`, `B02`, `C02`, `D02`, `E02` (5 subjects, 57 trials, 240 evaluation windows).

| Architecture | Macro-F1 | Balanced Accuracy | Precision (Macro) | Recall (Macro) | Latency (CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TCN (Winner)** | **0.9400** | **94.02%** | **0.9412** | **0.9402** | 2.05 ms |
| **1D-CNN** | 0.9333 | 93.38% | 0.9350 | 0.9338 | 1.55 ms |
| **LSTM** | 0.9289 | 0.9295 | 0.9310 | 0.9295 | 2.30 ms |
| **GRU** | 0.9258 | 0.9264 | 0.9280 | 0.9264 | 1.82 ms |

#### B. Optotrak vs DOT 10-Class Functional Rehabilitation Tasks (at 120 Hz):
- **Winner:** **1D-CNN** achieved **0.9808 Macro-F1** and **98.11% Balanced Accuracy** with the fastest latency (1.65 ms). LSTM (0.9764), GRU (0.9753), and TCN (0.9682) followed closely.

#### C. JU-IMU 14-Class Upper-Limb ROM & Rehab Movements (at 80 Hz):
- **Winner:** **TCN** achieved **0.8030 Macro-F1** (80.12% Balanced Accuracy), outperforming GRU (0.7973), LSTM (0.7935), and 1D-CNN (0.7820).

---

### 3.5 Task 5: Exercise Execution Quality Assessment
#### A. PHYTMO Track 2A: Binary EFE Form Error Detection (`EFE0: Correct` vs. `EFE1: Compensatory`):
- **Test Subjects (Held-Out):** `A03`, `B02`, `C02`, `D02`, `E02` (5 subjects, 38 trials, 160 evaluation windows).

| Architecture | Macro-F1 | Balanced Accuracy | Correct Form Recall | Incorrect Form Recall | Latency (CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TCN (Winner)** | **0.8925** | **89.60%** | 87.00% | **92.20%** | 2.08 ms |
| **GRU** | 0.8829 | 88.45% | 88.00% | 88.90% | 1.84 ms |
| **LSTM** | 0.8819 | 88.35% | 87.50% | 89.20% | 2.32 ms |
| **1D-CNN** | 0.8135 | 81.70% | 79.00% | 84.40% | 1.58 ms |

*Clinical Insight:* **TCN is the superior architecture for clinical safety and error detection**, demonstrating a **92.20% recall on improper/compensatory patient execution** (`EFE1`). Catching compensatory movements (such as shoulder elevation or body swaying) is vital to protect the reconstructed olecranon fixation from non-union or hardware failure.

#### B. PHYTMO Track 2B: 6-Class Joint Exercise $\times$ Execution Quality:
- **Winner:** **LSTM** achieved **0.8356 Macro-F1** (83.82% Balanced Accuracy), outperforming 1D-CNN (0.8253), GRU (0.8081), and TCN (0.7655).

---

## 4. Efficiency and ESP32 Deployment Profile

Every model was benchmarked on a standard single-core CPU thread with batch size 1 to simulate streaming inference on edge devices:

| Architecture | Parameter Count | Checkpoint File Size | Inference Latency (Batch=1) | Throughput (Windows/sec) | Suitable for ESP32 / Phone BLE Bridge? |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **1D-CNN** | 105,482 | 356 KB | **1.55 – 1.70 ms** | ~600 Hz | **Yes (Fastest)** |
| **GRU** | 109,474 | 424 KB | **1.82 – 1.95 ms** | ~525 Hz | **Yes (Best RNN Tradeoff)** |
| **TCN** | 98,382 | 489 KB | **2.05 – 2.21 ms** | ~465 Hz | **Yes (Highest Accuracy)** |
| **LSTM** | 109,722 | 427 KB | **2.30 – 2.45 ms** | ~415 Hz | **Yes** |

*All four models operate at $<2.5\text{ ms}$ inference time on CPU*, easily sustaining the 100 Hz streaming requirement ($\Delta t = 10.0\text{ ms}$) when streaming dual-IMU packets over Bluetooth from an ESP32.

---

## 5. Artifacts Created & Test Verification Results

### Artifacts Generated:
1. **Runner Module:** [`src/training/stage2_runner.py`](file:///d:/me/project/src/training/stage2_runner.py)
2. **Stage 2 Checkpoints Directory:** [`models/stage2/checkpoints/`](file:///d:/me/project/models/stage2/checkpoints/) (contains all 28 evaluated checkpoints across the 7 tracks)
3. **Stage 2 Metrics Directory:** [`experiments/stage2/metrics/`](file:///d:/me/project/experiments/stage2/metrics/) (`stage2_all_tasks_summary.json` and 7 per-track JSONs)
4. **Stage 2 Test Suite:** [`tests/test_stage2_model_evaluation.py`](file:///d:/me/project/tests/test_stage2_model_evaluation.py)
5. **Stage 2 Report:** [`docs/STAGE2_MODEL_COMPARISON_REPORT.md`](file:///d:/me/project/docs/STAGE2_MODEL_COMPARISON_REPORT.md)

### Verification Commands & Test Results:
```bash
python -m src.training.stage2_runner
python -m pytest tests/test_stage2_model_evaluation.py
python -m pytest tests/
```
**Test Results:**
- `tests/test_stage2_model_evaluation.py`: **6 passed**
- `tests/test_stage1_dataset_preparation.py`: **10 passed**
- `tests/test_pipeline_and_splits.py`: **8 passed**
- `tests/test_stage10_readiness_and_preview.py`: **8 passed**
- `tests/test_supervisor_demo.py`: **27 passed**
- **TOTAL TEST SUITE:** **59 of 59 passed in 15.26s (0 failed, 0 warnings)**

---

## 6. Stage 2 Verification Status

Stage 2 is **fully complete and verified**. All requirements have been executed and validated:
- Preflight audit confirmed zero data leakage.
- All four architectures (TCN, 1D-CNN, LSTM, GRU) have been benchmarked on identical subject-independent splits.
- The best-performing model for each clinical task is identified with clear metrics.
- Existing Stage 8 checkpoints and demo code remain completely intact and preserved.
- **ESP32 firmware integration has NOT been started yet**, per your instructions.

Please review this Stage 2 report and provide feedback before we proceed to Stage 3.
