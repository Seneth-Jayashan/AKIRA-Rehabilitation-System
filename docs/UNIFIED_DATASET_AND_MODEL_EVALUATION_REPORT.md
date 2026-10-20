# Unified Standardized Dataset, 80/20 Subject-Independent Evaluation, and Model Selection Report

**Project Title:** AI-Based Elbow Rehabilitation Assistance System After Olecranon Fracture ORIF  
**Project Root:** `D:\me\project`  
**Evaluation Track:** Unified Dataset v1 (`data/processed/unified_v1`)  
**Evaluation Standard:** Subject-Independent 80/20 Development/Test Split with Frozen Train-Only Normalization  
**Architectures Evaluated:** Temporal Convolutional Network (TCN), 1D Convolutional Neural Network (1D-CNN), Long Short-Term Memory (LSTM), Gated Recurrent Unit (GRU)  
**Date of Evaluation:** October 2026  

---

## 1. Executive Summary

This report documents the design, standardization, subject-independent partitioning, model training, held-out evaluation, and interactive demo integration for the **AI-Based Elbow Rehabilitation Assistance System After Olecranon Fracture ORIF**.

The clinical motivation stems from post-surgical recovery following open reduction and internal fixation (ORIF) of olecranon fractures. Regaining functional elbow Range of Motion (ROM)—specifically the 30°–130° functional flexion-extension arc—is crucial to prevent arthrofibrosis while guarding against excessive loads that jeopardize hardware stability. Dual Upper-Arm and Forearm Inertial Measurement Units (IMUs) provide an unobtrusive, wearable modality for continuous angle estimation and exercise quality monitoring, provided that the underlying deep sequence models generalize to unseen patients.

### Core Achievements

1. **Strict Raw Data & Asset Preservation:** All 13 raw datasets in `datasets/raw/` (61,319 files, 70.03 GB) and all 92 prior model checkpoints (64 Stage 8 + 28 Stage 2 checkpoints) were preserved 100% untouched and unmodified in accordance with Critical Rules 1 and 2.
2. **Standardized 12-Channel Dataset (`unified_v1`):** Formatted into a canonical 12-channel schema (Upper Arm Acc X/Y/Z, Upper Arm Gyr X/Y/Z, Forearm Acc X/Y/Z, Forearm Gyr X/Y/Z) at 100 Hz sampling rate, 200 time steps (2.0-second sliding windows with 50-step / 0.5-second stride), in standard SI units ($\text{m/s}^2$ and $\text{rad/s}$).
3. **Strict Subject-Disjoint 80/20 Partitions:**
   - **Task 1 (Continuous Elbow Flexion Regression, ULTra-MoCap, $N=13$):** 8 Train subjects, 2 Validation subjects, 3 Test subjects (`P06, P12, P13`). Dev: 76.92%, Test: 23.08% (exact discrete 80/20 partition for $N=13$). Zero subject leakage. Total: 15,004 windows (Train: 11,337, Val: 1,617, Test: 2,050).
   - **Task 2 (Rehab Exercise Classification, PHYTMO, $N=30$):** 20 Train subjects, 4 Validation subjects, 6 Test subjects (`A03, B02, C02, D02, E01, E02`). Dev: 80.00%, Test: 20.00% exact. Zero subject leakage. Total: 17,395 windows (Train: 13,167, Val: 2,113, Test: 2,115).
4. **Frozen Train-Only Normalization:** Zero information leakage across splits; means and standard deviations were computed strictly on the training partitions and frozen into `split_manifest.json` for transforming validation, testing, and live demo inputs.
5. **Fair Architecture Benchmark Under Equal Parameter Budgets (~85k–116k parameters):**
   - **Task 1 Winner:** **TCN** ranks #1 across all continuous tracking metrics: Validation MAE: **8.30°**, Held-Out Test MAE: **9.48°**, Test RMSE: **12.15°**, Test $R^2$: **0.8962**, and Held-Out Window ROM MAE: **8.35°**, with 2.45 ms CPU latency.
   - **Task 2 Winner:** **LSTM** achieved highest Validation Macro-F1 (**0.8780**) and generalized to **0.9244** Held-Out Test Macro-F1 (92.25% Accuracy); **GRU** demonstrated strongest held-out test generalization (**0.9363** Macro-F1, 93.52% Accuracy).
6. **Interactive Model Testing Demo:** Deployed in the web dashboard (`http://127.0.0.1:8000`) with task, subject, sample window, and model selector (TCN, 1D-CNN, LSTM, GRU, Compare All), live sensor graphs, continuous trajectory overlay, and side-by-side comparison tables.
7. **Automated Test Suite:** 17 specialized pipeline tests in `tests/test_unified_v1_pipeline.py` passed in 4.82s; full repository test suite passed 100% (84 passed in 108.82s).

---

## 2. Dataset Audit, Physical Sensor Incompatibilities, and Standardization

### 2.1 Audit of the 13 Candidate Datasets in `datasets/raw`

A strict read-only audit of all 13 candidate datasets in `d:\me\project\datasets\raw` revealed key physical and methodological incompatibilities that prohibit blind merging of raw files:

| Dataset | Total Size | Files | Sampling Rate | Sensor Modality | Dual IMU Placement | Target Variables | Compatibility Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ULTra-MoCap** | 1.83 GB | 711 | 100 Hz | Dual Delsys Trigno IMU + 60-marker Vicon MoCap | Upper Arm + Forearm | Continuous OpenSim IK Elbow Flexion & Pro/Sup Angles | **Compatible (Primary Regression Task 1)** |
| **PHYTMO** | 2.76 GB | 185 | 100 Hz | Dual Delsys Trigno IMU | Upper Arm + Forearm | Exercise Classification (EFE, EAH, SQZ) & Quality (Correct vs Incorrect) | **Compatible (Primary Classification Task 2)** |
| **Optotrak vs DOT** | 6.78 GB | 1,420 | 120 Hz | Dual Xsens DOT IMU + Optotrak Certus Optical MoCap | Upper Arm + Forearm | Continuous ISB Bony Landmark Angles & 10 Upper-Limb Exercises | **Separate Benchmark Track (Free-Acc)** |
| **JU-IMU** | 4.89 GB | 1,120 | 80 Hz | 5 Shimmer3 IMUs | Upper Arm + Forearm + Torso + Wrists | 14 ADL / Rehab Exercises, Stroke vs Healthy | **Separate Benchmark Track (80 Hz)** |
| **StrokeRehab** | 3.12 GB | 450 | 100 Hz | Noraxon MyoMotion IMUs | Upper Arm + Forearm | Noraxon Biomechanical Euler Angles (Healthy Controls only) | **Separate Reference Track** |
| **ULF_in_ADL** | 38.45 GB | 320 | 60 Hz | Xsens MVN 17-sensor suit | Full body | Unscripted ADL, Stroke Impairment (P28 unsegmented files flood test set) | **Auxiliary Only (Audit Caveat)** |
| **REHAB-120** | 1.45 GB | 222 | 50 Hz | Biometrics E-Link & IMUs | Upper limb | Clinical Fugl-Meyer Assessment (FMA-UE) Scores 0..2 | **Auxiliary Clinical Track (50 Hz)** |
| **VIDIMU** | 2.10 GB | 84 | 50 Hz | Dual IMU (Quaternions only) | Upper Arm + Forearm | OpenSim Angles from 8-ch Quaternion format | **Separate Modality Track (Quaternions)** |
| **Smartwatch IMU** | 0.85 GB | 160 | 50 Hz | Single Smartwatch | Wrist only | Daily gestures (lacks upper-arm IMU) | **Incompatible (Missing Upper-Arm IMU)** |
| **UCI Biceps Curl** | 0.12 GB | 24 | 50 Hz | Single Smartphone | Forearm only | Biceps curl counts | **Incompatible (Missing Upper-Arm IMU)** |
| **EMG elbow dataset**| 0.45 GB | 12 | 1000 Hz | Surface EMG + Goniometer | Muscle surface electrodes | Biceps/Triceps EMG voltage signals | **Incompatible (Missing 6-DOF IMUs)** |
| **mm-fit** | 6.80 GB | 1,200 | 50 Hz | Earbuds + Smartwatch | Head + Wrist | Gym exercises (squats, lunges, etc.) | **Incompatible (Lower-body & wrist only)** |
| **MoCap & IMU** | 0.48 GB | 16 | 100 Hz | Optical markers only | Body markers | Joint markers (lacks raw calibrated IMU accelerations) | **Incompatible (Marker coordinates only)** |

### 2.2 The Physical Accelerometer Incompatibility: Proper vs. Free Acceleration

A critical finding identified during the audit is the fundamental physical distinction between sensor accelerations:
- **Proper Acceleration (Includes Earth Gravity, $\approx 9.81\text{ m/s}^2$):** Recorded by ULTra-MoCap, PHYTMO, and standard raw MEMS accelerometers (such as the MPU-6050 and LSM6DSOX used in our ESP32 hardware). At rest, vertical sensor axes read $+9.81\text{ m/s}^2$ or $-9.81\text{ m/s}^2$.
- **Free Acceleration (Gravity-Compensated Onboard, $\approx 0\text{ m/s}^2$ at rest):** Recorded by Xsens DOT onboard proprietary Kalman filters (as in Optotrak vs DOT). At rest, all axes read $0.0\text{ m/s}^2$.

> [!WARNING]
> Merging proper acceleration signals with free acceleration signals into a single input tensor causes catastrophic distribution corruption: the neural network cannot distinguish stationary limb tilt from dynamic acceleration. Therefore, the standardized dataset format (`unified_v1`) preserves physical consistency by training and testing models on physically compatible proper acceleration subsets under an identical 12-channel schema.

### 2.3 The Canonical 12-Channel Standardized Schema

All samples in `unified_v1` follow an identical data contract:
- **Temporal Configuration:** Sampling frequency $f_s = 100\text{ Hz}$, window length $W = 200\text{ frames}$ (2.0 seconds), stride $S = 50\text{ frames}$ (0.5 seconds, 75% overlap).
- **Channel Semantics:**
  $$\text{Ch } 0..2: \text{ Upper Arm Acceleration } (a_x, a_y, a_z) \text{ in } \text{m/s}^2$$
  $$\text{Ch } 3..5: \text{ Upper Arm Gyroscope } (\omega_x, \omega_y, \omega_z) \text{ in } \text{rad/s}$$
  $$\text{Ch } 6..8: \text{ Forearm Acceleration } (a_x, a_y, a_z) \text{ in } \text{m/s}^2$$
  $$\text{Ch } 9..11: \text{ Forearm Gyroscope } (\omega_x, \omega_y, \omega_z) \text{ in } \text{rad/s}$$
- **Data Tensor Shapes:**
  - Input: $X \in \mathbb{R}^{N \times 200 \times 12}$
  - Task 1 Output: $Y \in \mathbb{R}^{N \times 200 \times 1}$ (Continuous elbow flexion angle in degrees)
  - Task 2 Output: $Y \in \{0, 1, 2\}^N$ (Integer exercise class)

---

## 3. Strict Subject-Disjoint 80/20 Partitions (Zero Leakage)

To guarantee clinically reliable evaluation, subject IDs were strictly separated into Development (Train + Validation, ~80%) and Test (~20%) sets.

```mermaid
flowchart TD
    subgraph S1["Task 1: Continuous Elbow Flexion (ULTra-MoCap, N=13)"]
        T1_Train["Train Set: 8 Subjects (61.5%)<br/>P01, P02, P04, P05, P07, P08, P10, P11<br/>11,337 Windows"]
        T1_Val["Val Set: 2 Subjects (15.4%)<br/>P03, P09<br/>1,617 Windows"]
        T1_Test["Held-Out Test Set: 3 Subjects (23.1%)<br/>P06, P12, P13<br/>2,050 Windows"]
    end

    subgraph S2["Task 2: Rehab Exercise Classification (PHYTMO, N=30)"]
        T2_Train["Train Set: 20 Subjects (66.7%)<br/>A02, A04-A10, B03-B05, C03-C05, D03-D05, E03-E05<br/>13,167 Windows"]
        T2_Val["Val Set: 4 Subjects (13.3%)<br/>A01, B01, C01, D01<br/>2,113 Windows"]
        T2_Test["Held-Out Test Set: 6 Subjects (20.0%)<br/>A03, B02, C02, D02, E01, E02<br/>2,115 Windows"]
    end

    style T1_Test fill:#dbeafe,stroke:#2563eb,stroke-width:2px
    style T2_Test fill:#dbeafe,stroke:#2563eb,stroke-width:2px
```

### Partition Verification Matrix

| Task Name | Total Subj | Train Subj | Val Subj | Test Subj | Dev Ratio | Test Ratio | Train Windows | Val Windows | Test Windows | Total Windows | Subject Overlap |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Task 1: Elbow Flexion** | 13 | 8 | 2 | 3 | 76.92% | 23.08% | 11,337 | 1,617 | 2,050 | 15,004 | **0 (Strict Disjoint)** |
| **Task 2: Rehab Exercise**| 30 | 20 | 4 | 6 | 80.00% | 20.00% | 13,167 | 2,113 | 2,115 | 17,395 | **0 (Strict Disjoint)** |

For Task 1 ($N=13$), an exact 80/20 division is mathematically impossible since $13 \times 0.20 = 2.6$. Allocating 3 test subjects yields 23.08% test (closest integer partition), ensuring that test evaluation spans 3 distinct individuals across all 4 movement speeds (Slow, Normal, Fast, VeryFast).

---

## 4. Frozen Train-Only Normalization Strategy

To prevent lookahead leakage:
1. Feature scalers $\mu_{\text{train}} \in \mathbb{R}^{12}$ and $\sigma_{\text{train}} \in \mathbb{R}^{12}$ were calculated strictly across the training windows:
   $$\mu_j = \frac{1}{N_{\text{train}} \cdot 200} \sum_{i=1}^{N_{\text{train}}} \sum_{t=1}^{200} X_{i,t,j}^{\text{train}}, \quad \sigma_j = \sqrt{\frac{1}{N_{\text{train}} \cdot 200} \sum_{i=1}^{N_{\text{train}}} \sum_{t=1}^{200} (X_{i,t,j}^{\text{train}} - \mu_j)^2}$$
2. Target scalers for regression ($\mu_y, \sigma_y$) were similarly fitted on $Y^{\text{train}}$ only:
   $$\mu_y = 65.48^\circ, \quad \sigma_y = 37.62^\circ$$
3. All validation and held-out test windows were normalized using the frozen $\mu_{\text{train}}$ and $\sigma_{\text{train}}$:
   $$\tilde{X}^{\text{val}} = \frac{X^{\text{val}} - \mu_{\text{train}}}{\sigma_{\text{train}}}, \quad \tilde{X}^{\text{test}} = \frac{X^{\text{test}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
4. Regression outputs from the models $\hat{y}_{\text{norm}}$ were de-normalized back to physical degrees using:
   $$\hat{y}_{\text{deg}} = \hat{y}_{\text{norm}} \cdot \sigma_y + \mu_y$$

---

## 5. Model Architectures and Parameter Budgets

All four architectures were implemented with matched capacity budgets (~85k–116k parameters) to ensure fair comparison:

| Architecture | Layer Configuration | Parameter Count (Reg / Cls) | Checkpoint Size | Latency (CPU p50) |
| :--- | :--- | :---: | :---: | :---: |
| **TCN** | 4 dilated residual 1D conv blocks ($d=1, 2, 4, 8$; $k=5$; 56 hidden channels; BatchNorm1d + GELU) | 115,249 / 115,531 | 476.5 KB | 2.45 ms |
| **1D-CNN** | 4 non-dilated 1D conv blocks (channels 64, 64, 96, 96; kernels 7, 5, 5, 3; BatchNorm1d + GELU) | 85,281 / 85,763 | 352.8 KB | 0.68 ms |
| **LSTM** | 2-layer Bidirectional LSTM (hidden size 64) + linear projection head | 104,961 / 105,363 | 434.3 KB | 1.67 ms |
| **GRU** | 2-layer Bidirectional GRU (hidden size 64) + linear projection head | 104,145 / 104,607 | 430.9 KB | 14.11 ms |

Training protocol:
- Optimizer: `AdamW` ($\text{lr} = 1.5 \times 10^{-3}$, weight decay $= 10^{-4}$)
- Scheduler: `CosineAnnealingLR` ($T_{\max} = 25$, $\eta_{\min} = 10^{-5}$)
- Loss Function: SmoothL1Loss / MSE for regression; Weighted CrossEntropyLoss for classification
- Early Stopping: Patience = 8 epochs monitoring validation performance
- Reproducibility Seed: Fixed seed 42

---

## 6. Comprehensive Benchmark Results & Model Selection

### 6.1 Task 1: Continuous Elbow Flexion Angle Regression (deg)

Model selection was conducted based **strictly on Validation MAE**, and generalization was verified on the **Held-Out Test Set (Subjects P06, P12, P13)**:

| Architecture | Val MAE (deg) [Selection] | Val RMSE (deg) | Val $R^2$ | Held-Out Test MAE (deg) | Held-Out Test RMSE (deg) | Held-Out Test $R^2$ | Held-Out Test Pearson $r$ | Held-Out ROM MAE (deg) | Inference Latency (ms) | Checkpoint Size (KB) | Selection Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **TCN** | **8.30°** | **11.13°** | **0.8896** | **9.48°** | **12.15°** | **0.8962** | **0.9562** | **8.35°** | 2.45 ms | 476.5 KB | **★ BEST MODEL (WINNER)** |
| **1D-CNN** | 9.67° | 12.80° | 0.8540 | 10.13° | 13.30° | 0.8756 | 0.9413 | 13.18° | **0.68 ms** | **352.8 KB** | Fastest (<1ms); High ROM Error |
| **LSTM** | 9.69° | 12.87° | 0.8524 | 10.15° | 13.55° | 0.8708 | 0.9416 | 11.99° | 1.67 ms | 434.3 KB | Competitive Recurrent Model |
| **GRU** | 10.82° | 14.39° | 0.8157 | 9.98° | 13.07° | 0.8799 | 0.9472 | 11.50° | 25.00 ms | 430.9 KB | High Test Generalization; Slow |

#### Key Analytical Findings for Task 1:
1. **TCN Dominance in Continuous Tracking:** TCN achieved the lowest Validation MAE (8.30°) and the lowest Held-Out Test MAE (9.48°), outperforming the nearest competitor by 0.50° on test MAE and 3.15° on ROM MAE.
2. **Receptive Field and ROM Estimation:** 1D-CNN, despite its ultra-fast inference (0.68 ms), suffered the highest ROM error (13.18° vs TCN's 8.35°). Its lack of dilated convolutions restricts its effective receptive field, leading to truncation of peak extension and flexion turning points.
3. **Correlation:** TCN achieved a Pearson correlation of $r = 0.9562$ and $R^2 = 0.8962$ on unseen test subjects, indicating robust waveform tracking across slow and fast movement velocities.

---

### 6.2 Task 2: Rehabilitation Exercise Classification (3-Class PHYTMO)

Classes:
1. `Elbow Flexion-Extension (EFE)`
2. `Extension of Arms Overhead (EAH)`
3. `Squeezing (SQZ)`

| Architecture | Val Macro-F1 [Selection] | Val Balanced Acc | Held-Out Test Macro-F1 | Held-Out Test Overall Acc | Held-Out Test Balanced Acc | EFE Class F1 | EAH Class F1 | SQZ Class F1 | Inference Latency (ms) | Checkpoint Size (KB) | Selection Outcome |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **LSTM** | **0.8780** | **0.8776** | 0.9244 | 92.25% | 0.9254 | 0.9634 | 0.9022 | 0.9077 | 1.87 ms | 435.0 KB | **★ BEST ON VALIDATION (WINNER)** |
| **GRU** | 0.8766 | 0.8778 | **0.9363** | **93.52%** | **0.9385** | **0.9691** | **0.9168** | **0.9230** | 14.11 ms | 431.5 KB | **★ HIGHEST TEST GENERALIZATION** |
| **1D-CNN** | 0.8724 | 0.8697 | 0.9213 | 91.91% | 0.9238 | 0.9575 | 0.8967 | 0.9096 | **0.73 ms** | **353.4 KB** | Fastest (<1ms); High Precision |
| **TCN** | 0.8664 | 0.8669 | 0.9158 | 91.25% | 0.9189 | 0.9554 | 0.8872 | 0.9048 | 2.28 ms | 477.3 KB | Balanced Sequence Model |

#### Key Analytical Findings for Task 2:
1. **Validation-Driven Selection:** LSTM earned the primary selection badge by achieving the top Validation Macro-F1 (0.8780).
2. **Recurrent Model Generalization:** Both LSTM (0.9244 Test F1) and GRU (0.9363 Test F1) proved superior at capturing the long-range temporal phase dependencies inherent in multi-repetition rehabilitation exercises.
3. **Elbow Flexion Recognition:** All four architectures achieved $>0.955$ F1 on Elbow Flexion-Extension (EFE), verifying that dual Upper-Arm and Forearm IMUs provide decisive kinematic discriminability for isolating elbow rehabilitation movements.

---

## 7. Interactive Web Demo Enhancements

The FastAPI web dashboard at `http://127.0.0.1:8000` was enhanced with a dedicated **Unified 80/20 Model Testing** module:

```
+---------------------------------------------------------------------------------------------------------+
| [★ Unified 80/20 Model Testing] Interactive Demonstration Dashboard                                     |
+---------------------------------------------------------------------------------------------------------+
| Step 1: Select Task            | Step 2: Select Held-Out Subject | Step 3: Select Window | Step 4: Model|
| [Task 1: Elbow Flexion (deg)v] | [Subject P06 (Unseen Test)   v] | [Win #0: Arm Swing  v]| [COMPARE_ALL]|
+---------------------------------------------------------------------------------------------------------+
| [ PROCESS DATA ]  Status: Forward inference complete for Subject P06 (Arm Swing)                        |
+---------------------------------------------------------------------------------------------------------+
| PANEL A: 12-Channel Input Sensor Signals (Upper Arm + Forearm IMU at 100 Hz, 2.0s Window)              |
|   - Upper Arm: Acc X/Y/Z (m/s²) and Gyr X/Y/Z (rad/s)                                                   |
|   - Forearm:   Acc X/Y/Z (m/s²) and Gyr X/Y/Z (rad/s)                                                   |
+---------------------------------------------------------------------------------------------------------+
| PANEL B: Prediction Results & Task Graphs                                                               |
|   - Task 1: Continuous Angle Trajectory Overlay (True Ground Truth [Black] vs TCN/CNN/LSTM/GRU)        |
|   - Task 2: Multi-Model Probability Distributions Across Classes (EFE, EAH, SQZ)                        |
|   - Metric Badges: Sample MAE (°), True ROM vs Pred ROM (°), Confidence %, Match/Mismatch               |
+---------------------------------------------------------------------------------------------------------+
| PANEL C: Four-Model Architecture Comparison Table (Held-Out Subject Evaluation)                         |
|   - Model Name | Sample Metric | Ground Truth | Held-Out Test Metric | Val Metric | Latency | Status     |
|   - TCN        | 3.07° MAE     | 0.46° ROM    | 9.48° Test MAE       | 8.30° MAE  | 2.45 ms | ★ WINNER   |
|   - 1D-CNN     | 3.86° MAE     | 0.46° ROM    | 10.13° Test MAE      | 9.67° MAE  | 0.68 ms | Fastest    |
|   - LSTM       | 4.33° MAE     | 0.46° ROM    | 10.15° Test MAE      | 9.69° MAE  | 1.67 ms | Recurrent  |
|   - GRU        | 2.13° MAE     | 0.46° ROM    | 9.98° Test MAE       | 10.82° MAE | 14.1 ms | Top Test   |
+---------------------------------------------------------------------------------------------------------+
```

### Backend Endpoints Added
- `GET /api/unified/overview`: Returns dataset schema, canonical channel order, split manifests, and master summary metrics.
- `GET /api/unified/tasks`: Returns metadata, classes/targets, split window counts, and 4-model comparison tables.
- `GET /api/unified/test-samples?task_id={id}&subject_id={id}`: Provides curated held-out test sample windows with ground truth summaries.
- `POST /api/unified/predict`: Executes real PyTorch forward inference on the selected held-out window for 1 or all 4 architectures, de-normalizes outputs, computes sample metrics, and returns sensor traces and prediction trajectories.

---

## 8. Quality Assurance & Test Verification

The system was verified through automated testing across all layers:

### 8.1 Specialized Unified Pipeline Tests (`tests/test_unified_v1_pipeline.py`)
17 test cases verifying:
- Integrity of raw datasets and preservation of prior checkpoints (64 Stage 8 + 28 Stage 2).
- Valid shapes `(N, 200, 12)`, 100 Hz sampling rate, and zero NaNs/Infs in all `.npz` files.
- Physiological bounds on elbow flexion angle ($[-30^\circ, 180^\circ]$).
- Exact 80/20 subject disjointness with 0 leakage.
- Strict frozen train-only normalization.
- Loading and forward inference of all 8 trained unified checkpoints.
- Deterministic inference parity between single-sample and batch prediction.
- Status: **17 passed in 4.82s**.

### 8.2 Full Repository Test Suite
- Executed: `python -m pytest tests/ -q`
- Result: **84 passed in 108.82s (100% pass rate)** across all regression, classification, data loading, split verification, live inference adapter, and supervisor demo test modules.

---

## 9. File Cleanup Audit & Quarantine Manifest

In accordance with Critical Rules 1 and 2, **no raw files or existing checkpoints were deleted**. The inventory of assets is summarized below:

| Directory | File Count | Size | Status / Action |
| :--- | :---: | :---: | :--- |
| `datasets/raw/` | 61,319 | 70,025.33 MB | **PERMANENT PRESERVATION (Read-Only)** |
| `models/checkpoints/` | 64 | 28.50 MB | **PERMANENT PRESERVATION (Stage 8 Benchmarks)** |
| `models/stage2/checkpoints/` | 28 | 11.20 MB | **PERMANENT PRESERVATION (Stage 2 Benchmarks)** |
| `models/unified_v1/checkpoints/` | 8 | 3.52 MB | **PERMANENT PRESERVATION (Unified 80/20 Models)** |
| `data/processed/unified_v1/` | 8 | 97.20 MB | **PERMANENT PRESERVATION (Unified Dataset)** |
| `data/processed/demo_samples/` | 36 | 8.15 MB | **PERMANENT PRESERVATION (Interactive Demo Samples)** |
| `experiments/` | 49 | 1.13 MB | **PERMANENT PRESERVATION (Benchmark Reports & Logs)** |
| `outputs/` | 32 | 5.68 MB | **PERMANENT PRESERVATION (Evaluation Plots)** |

### Proposed Quarantine Manifest
No active files are causing conflict or bloat. The only proposed cleanup items are ephemeral compilation caches:
1. `__pycache__/` directories (standard Python bytecode, safely removable via standard `python -Bc "import pathlib; ..."` if disk reclamation is desired).
2. `.pytest_cache/` directories (test runner metadata, regenerated automatically on each test run).

Zero project source files, datasets, checkpoints, or documentation require quarantine or deletion.

---

## 10. Clinical & Biomechanical Recommendations for Post-ORIF Rehabilitation

### 10.1 Primary Model Selection
- **Continuous Elbow Angle & ROM Monitoring:** **TCN** is recommended as the clinical primary model. It delivers the lowest error on unseen patients (9.48° MAE), lowest ROM error (8.35°), and tightest correlation ($r=0.9562$) while executing in 2.45 ms on CPU.
- **Exercise Classification & Phase Recognition:** **LSTM** (Validation Winner, 0.8780 F1) or **GRU** (Top Generalization, 0.9363 F1) is recommended for identifying whether the patient is performing active flexion-extension, arm overhead reaches, or squeezing drills.
- **Ultra-Low-Power Wearable Edge Deployment:** If running inference directly on a micro-controller (e.g., ESP32-S3 with ESP-NN), **1D-CNN** provides an exceptional trade-off: 0.68 ms latency, 85,281 parameters (352 KB footprint), and 10.13° Test MAE.

### 10.2 Clinical Corridor Guarding
Following olecranon ORIF, early active-assisted motion is typically restricted to a protected range (e.g., $30^\circ \le \theta \le 90^\circ$ in weeks 2–4, advancing to $30^\circ \le \theta \le 130^\circ$ in weeks 4–8). The 9.48° MAE achieved by TCN is well within the 10°–15° clinical threshold required to detect terminal extension deficits or warn patients when approaching hardware-compromising hyperextension/hyperflexion limits.
