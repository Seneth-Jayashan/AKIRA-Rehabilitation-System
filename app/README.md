# AKIRA Knee Rehabilitation System: Complete Edge-AI Package

**Component:** Smart Post-ORIF Rehabilitation Ecosystem with Edge-AI Kinematic Tracking  
**Individual Focus:** Knee Joint Rehabilitation Following Tibial Plateau ORIF  
**Student Investigator:** Wickramarathna S A (IT23222540)  
**Framework:** TensorFlow 2.x / Keras Functional API / FastAPI / Bleak (Native Windows BLE) / TailwindCSS / Chart.js  

---

## 📑 Table of Contents
1. [System Architecture & Methodology (Boxes 01 to 08)](#1-system-architecture--methodology-boxes-01-to-08)
2. [Hardware Setup & Sensor Placement](#2-hardware-setup--sensor-placement)
3. [Smart Sensor Misplacement & Auto-Swap Correction](#3-smart-sensor-misplacement--auto-swap-correction)
4. [Multi-Dataset Harmonization & Leak-Free Subject Split](#4-multi-dataset-harmonization--leak-free-subject-split)
5. [5-Stage Sensor Fusion & 16 Clinical Features (Boxes 02 & 03)](#5-5-stage-sensor-fusion--16-clinical-features-boxes-02--03)
6. [The 7 Official Session Outputs (Box 05)](#6-the-7-official-session-outputs-box-05)
7. [Longitudinal Tracking & Recovery Prediction Engine (Boxes 06, 07 & 08)](#7-longitudinal-tracking--recovery-prediction-engine-boxes-06-07--08)
8. [The 6 Benchmarked Dual-Stream Hybrid Models, Baselines & Edge TFLite](#8-the-6-benchmarked-dual-stream-hybrid-models-baselines--edge-tflite)
9. [ESP32 Firmware & Connectivity (Native BLE & Serial)](#9-esp32-firmware--connectivity-native-ble--serial)
10. [Package Directory Structure & How to Run](#10-package-directory-structure--how-to-run)

---

## 1. System Architecture & Methodology (Boxes 01 to 08)

The system implements the end-to-end methodology specified in Proposal Figure 3 for post-operative Tibial Plateau Open Reduction and Internal Fixation (ORIF) rehabilitation:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          AKIRA SYSTEM ARCHITECTURE PIPELINE                            │
└────────────────────────────────────────────────────────────────────────────────────────┘

  [BOX 01: DATA ACQUISITION]
  Dual MPU6050 IMUs @ 100 Hz (Thigh 0x68 + Shin 0x69, 20 Hz Hardware DLPF) ──► 12 Channels
                           │
                           ▼
  [BOX 02: KNEE ANGLE COMPUTATION PIPELINE]
  2.1 Cubic-Spline 100 Hz Resampling ──► 2.2 4th-Order 5 Hz Butterworth LPF
  ──► 2.3 Static Tare & Gyro Bias Calibration ──► 2.4 Madgwick / Complementary 3D Fusion
  ──► 2.5 3D SVD Relative Hinge Orientation ──► Continuous Knee Angle θ(t)
                           │
                           ▼
  [BOX 03: CLINICAL FEATURE EXTRACTION]
  Extracts 16 Biomechanical Parameters (ROM, Peak Depth, Extensor Lag, SPARC, Reps, Duration)
                           │
        ┌──────────────────┴──────────────────┐
        ▼                                     ▼
  [Input A: Raw Waveform (200×12)]      [Input B: 16 Clinical Features]
        │                                     │
        ▼                                     ▼
  [Temporal Deep Learning Branch]       [Dense Clinical Branch]
  (1D-CNN / BiLSTM / ResNet / Attention) (Dense MLP + BatchNorm)
        │                                     │
        └──────────────────┬──────────────────┘
                           ▼
                 [Feature Fusion Layer]
                           │
  [BOX 04: AI MODEL PREDICTIONS]
  ├─► Head 1: Exercise Classification (Walking, Seated Extension, Mini Squat, Step-Up)
  └─► Head 2: Movement Quality Assessment (Good, Needs Improvement, Bad)
                           │
                           ▼
  [BOX 05: THE 7 SESSION OUTPUTS]
  1. Recognized Exercise   2. Knee Angle & ROM   3. Peak Depth & Extensor Lag   4. Rep Count
  5. Smoothness (SPARC)   6. Duration & Speed   7. Quality Score (0–100)
                           │
                           ▼
  [BOX 06 & 07: LONGITUDINAL TRACKING & RECOVERY PREDICTION]
  Per-patient, per-exercise session history (S1 → Sn), regression trajectory (n ≥ 3),
  un-floored forecast (detects both progress and regression), and milestone countdown
                           │
                           ▼
  [BOX 08: PERSONALIZED REHABILITATION ASSESSMENT]
  Clinician-protocol-driven recommendations (target ROM, safe flexion limit, lag threshold)
```

---

## 2. Hardware Setup & Sensor Placement

The system captures movement from two 6-DOF IMUs (MPU6050) operating on a single I2C bus at **100 Hz (10 ms period)** with hardware **20 Hz Digital Low-Pass Filtering (`REG_CONFIG = 0x04`)** enabled:

* **ESP32 3.3V / GND / GPIO 21 (`SDA`) / GPIO 22 (`SCL`)** wired to both IMUs.
* **Thigh IMU (`0x68`, `AD0 = GND`):** Centered **10 cm above the patella** along the anterior mid-thigh.
* **Shin IMU (`0x69`, `AD0 = 3.3V`):** Centered **10 cm below the patella** along the proximal tibia.

---

## 3. Smart Sensor Misplacement & Auto-Swap Correction

1. **Reversed Leg Straps (Thigh $\leftrightarrow$ Shin Swap):**
   * In open-chain knee exercises (Seated Extension, Heel Slides, SLR), the Shin experiences significantly higher rotational energy than the stationary Thigh.
   * If $\text{Power}_{\text{Thigh}} > 2.2 \times \text{Power}_{\text{Shin}}$, `auto_detect_and_correct_sensor_swap()` automatically swaps Channels `[0..5]` with `[6..11]` and notifies the clinician.
2. **Upside-Down Sensor Placement & Hampel/MAD Spike Filtering:**
   * Checks resting 3D gravity alignment and applies a sliding-window Median Absolute Deviation (Hampel) spike filter (`clean_i2c_glitches_mad`) to remove single-sample I2C transients without magic-number thresholds.

---

## 4. Multi-Dataset Harmonization & Leak-Free Subject Split

The training and evaluation corpus (`1,036` clean 2.0-second windows across `77` subjects) integrates **4 public dual-IMU rehabilitation and gait datasets** (`app/api/preprocess_datasets.py`) so every exercise is represented across multiple independent datasets and all 3 quality grades:

| Exercise Class | Source Datasets Included | Total Trials | Quality Grades (`0: Good / 1: Needs Imp / 2: Bad`) | Median ROM ($^\circ$) | Clinician Safe Limit |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Walking (Class 0)** | KneE-PAD (190), PHYTMO (77), StrengthSense (42), HuGaDB (40) | **349** | `159 / 92 / 98` | $45.8^\circ\text{--}62.4^\circ$ | $\le 70^\circ$ (Target: $60^\circ$) |
| **Seated Knee Extension (Class 1)** | KneE-PAD (166), PHYTMO (73) | **239** | `93 / 73 / 73` | $67.0^\circ\text{--}93.3^\circ$ | $\le 100^\circ$ (Target: $90^\circ$) |
| **Mini Squat (Class 2)** | KneE-PAD (176), PHYTMO (73), StrengthSense (52) | **301** | `146 / 76 / 79` | $41.5^\circ\text{--}48.7^\circ$ | **$30^\circ\text{--}45^\circ$** (Target: $42^\circ$) |
| **Step-Up (Class 3)** | StrengthSense (82), PHYTMO (36), HuGaDB (29) | **147** | `88 / 33 / 26` | $65.7^\circ\text{--}83.2^\circ$ | $\le 75^\circ$ (Target: $65^\circ$) |

* **Strict Subject-Independent Split (`GroupKFold`):** `829` training trials (`61` subjects) and `207` held-out test trials (`16` completely unseen subjects; `0` subject overlap).
* **Zero Normalization Leakage:** `StandardScaler` (for Input B) and channel `means`/`stds` (for Input A) are fit strictly on the `829` training trials (`train_idx`) before transforming the unseen test trials.

---

## 5. 5-Stage Sensor Fusion & 16 Clinical Features (Boxes 02 & 03)

Implemented in [`api/math_feature_extractor.py`](api/math_feature_extractor.py) and validated against **OptiTrack 3D Optical Motion Capture** (`A01KFER0_1`: $\text{MAE} = 0.91^\circ$, $\text{RMSE} = 1.12^\circ$, $r = 0.9988$):

1. **Feature 1 (`rom_deg`):** $\max(\theta) - \min(\theta)$ — Active Range of Motion ($^\circ$).
2. **Feature 2 (`peak_depth_deg`):** $\max(\theta)$ — Peak Flexion Depth ($^\circ$, distinct from ROM by mean $4.47^\circ$).
3. **Feature 3 (`min_angle_deg`):** $\max(0^\circ, \min(\theta))$ — Terminal Extension Deficit / Quadriceps Lag ($5.42^\circ \pm 5.55^\circ$ across dataset).
4. **Feature 4 (`peak_angular_velocity`):** $\max |\omega(t)|$ ($\text{deg/s}$).
5. **Feature 5 (`mean_angular_velocity`):** $\text{mean} |\omega(t)|$ ($\text{deg/s}$).
6. **Feature 6 (`sparc_smoothness`):** Spectral Arc Length smoothness index ($\le 0$).
7. **Feature 7 (`repetition_count`):** Adaptive prominence peak segmentation ($0\text{--}4\text{ reps}$ per 2.0s window; $0$ when stationary).
8. **Feature 8 (`mean_repetition_duration`):** Mean active cycle duration ($\text{s}$; $0.0\text{ s}$ when stationary).
9. **Feature 9 (`angle_std`):** $\text{std}(\theta)$ ($^\circ$).
10. **Feature 10 (`velocity_std`):** $\text{std}(\omega)$ ($\text{deg/s}$).
11. **Feature 11 (`angle_iqr`):** $Q_{75}(\theta) - Q_{25}(\theta)$ ($^\circ$).
12. **Feature 12 (`max_ang_vel`):** Peak Flexion Velocity ($\max(\omega) > 0$, bending phase, $\text{deg/s}$).
13. **Feature 13 (`min_ang_vel`):** Peak Extension Velocity ($\min(\omega) < 0$, straightening phase, $\text{deg/s}$).
14. **Feature 14 (`mean_angle`):** $\text{mean}(\theta)$ ($^\circ$).
15. **Feature 15 (`angle_rms`):** $\sqrt{\text{mean}(\theta^2)}$ ($^\circ$).
16. **Feature 16 (`sparc_per_duration`):** $\text{SPARC} / \bar{T}_{\text{rep}}$ ($\text{s}^{-1}$).

---

## 6. The 7 Official Session Outputs (Box 05)

| Output # | Clinical Parameter | Primary Source | Clinical Computation |
| :---: | :--- | :--- | :--- |
| **01** | **Recognized Exercise** | AI Model (Head 1) | Genuine softmax class prediction & confidence % (never overridden) |
| **02** | **Knee Angle & ROM** | Math Extractor (F1) | Active joint excursion ($\theta_{\max} - \theta_{\min}$) in degrees |
| **03** | **Movement Depth & Lag** | Math Extractor (F2 & F3) | True Peak Flexion Depth $\theta_{\max}$ ($^\circ$) and Quadriceps Extension Lag $\theta_{\min}$ ($^\circ$) |
| **04** | **Repetition Count** | Math Extractor (F7) | True segmented repetition count ($0$ when limb is stationary) |
| **05** | **Movement Smoothness** | Math Extractor (F6) | Un-clipped $[0.0, 1.0]$ normalized SPARC smoothness index |
| **06** | **Duration & Velocity** | Math Extractor (F4, F5, F8) | True active repetition duration ($\text{s}$) and angular velocity ($\text{deg/s}$) |
| **07** | **Quality Score (0–100)** | Fusion (Head 2 + Math) | Un-clipped $0\text{--}100$ score evaluated against clinician protocol `target_rom_deg` |

---

## 7. Longitudinal Tracking & Recovery Prediction Engine (Boxes 06, 07 & 08)

* **Per-Patient & Per-Exercise Filtering:** Longitudinal trends filter by `patient_id` (with SHA-256 pseudonymization `PT-...`) and `exercise` so different exercises are never mixed together.
* **Real Dates & Post-ORIF Week Calculation:** Stores ISO `session_date` (`YYYY-MM-DD`) and computes post-ORIF week from `surgery_date` when provided.
* **Validated Regression Forecasting ($n \ge 3$):** Requires at least $3$ recorded sessions before fitting linear regression; reports un-clipped $R^2 \in [0, 1]$ and residual $\text{RMSE}$ ($^\circ$), and never floors forecasts at the last value so declining recovery trajectories (`slope < 0`) are detected and flagged.
* **Clinician-Configurable Protocol (`GET/POST /api/clinician/protocol`):** Clinicians can customize `target_rom_deg`, `max_safe_flexion_deg`, `max_extensor_lag_deg`, and `sessions_per_week` per exercise.

---

## 8. The 6 Benchmarked Dual-Stream Hybrid Models, Baselines & Edge TFLite

Evaluated on the **unseen-subject held-out test partition (`207` trials from `16` unseen subjects, `0` subject overlap)** (`api/models/test_metrics_summary.json`):

### A. 6 Dual-Stream Hybrid Models Leaderboard (Unseen Subjects)

| Rank | Model Architecture | Parameters | Exercise Acc (%) | Exercise Macro F1 (%) | Quality Acc (%) | Quality Macro F1 (%) | Combined Score (%) | Inference Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **Hybrid 5: Self-Attention + Kinematics** | 42,055 | 89.86% | 89.73% | **57.49%** | **54.45%** | **73.67%** | ~3.01 ms |
| **#2** | **Hybrid 1: 1D-CNN + Kinematics (Edge Model)** | 55,879 | **92.27%** | **91.85%** | 53.14% | 50.68% | **72.71%** | ~2.12 ms |
| **#3** | **Hybrid 4: 1D-ResNet + Kinematics** | 159,431 | 90.34% | 89.70% | 54.59% | 51.43% | **72.46%** | ~3.31 ms |
| **#4** | **Hybrid 6: Gated Attention + Kinematics** | 43,335 | 89.37% | 89.13% | 52.17% | 49.14% | **70.77%** | ~2.76 ms |
| **#5** | **Hybrid 3: CNN-BiLSTM + Kinematics** | 97,223 | 91.79% | 91.53% | 44.44% | 42.91% | **68.12%** | ~5.03 ms |
| **#6** | **Hybrid 2: BiLSTM + Kinematics** | 103,559 | 90.82% | 90.28% | 44.44% | 42.56% | **67.63%** | ~6.94 ms |

### B. Baseline Comparison (Proposal §3.3 Validation)

| Model | Input Modality | Exercise Acc (%) | Exercise Macro F1 (%) | Quality Acc (%) | Quality Macro F1 (%) | Combined Score (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Random Forest** | Input B Only (16 Clinical Features) | 81.64% | 82.89% | 50.72% | 46.84% | **66.18%** |
| **Baseline 2: Single-IMU 1D-CNN** | Single IMU Only (Thigh 6-Ch) | 86.96% | 86.25% | 47.34% | 44.57% | **67.15%** |
| **Hybrid 5: Self-Attention (Winner)** | **Dual-IMU (12-Ch) + 16 Features** | **89.86%** | **89.73%** | **57.49%** | **54.45%** | **73.67%** |
| **Hybrid 1: 1D-CNN (Best Exercise)** | **Dual-IMU (12-Ch) + 16 Features** | **92.27%** | **91.85%** | **53.14%** | **50.68%** | **72.71%** |

### C. Quantized Edge-AI TensorFlow Lite (`.tflite`) Validation (Proposal §3.5)

* **Quantized Model File:** `api/models/akira_hybrid_cnn_edge.tflite` (**73.29 KB**)
* **C Header / Array Export:** `api/models/edge_akira_classifier.c`
* **Mean Edge Inference Latency:** **0.208 ms** (95th Percentile: **0.268 ms**)
* **TFLite vs. Keras Prediction Agreement:** **100.0%** (Exercise) / **99.03%** (Quality)

---

## 9. ESP32 Firmware & Connectivity (Native BLE & Serial)

The ESP32 firmware sketch (`app/api/esp32_firmware/akira_knee_esp32.ino`) configures both MPU6050 sensors at $\pm 8\text{ g}$, $\pm 500\text{ deg/s}$, $20\text{ Hz}$ hardware DLPF (`0x04`), and $100\text{ kHz}$ I2C clock, streaming at $100\text{ Hz}$ over Nordic UART BLE (`6e400001-...`) and USB Serial (`115200` baud).

---

## 10. Package Directory Structure & How to Run

```powershell
# 1. Install dependencies
pip install -r app/api/requirements.txt

# 2. (Optional) Rebuild datasets & retrain all models from scratch
python app/api/preprocess_datasets.py
python app/api/train_models.py --epochs 25 --split subject

# 3. Run automated unit & integration tests + NFR evaluation
python -m pytest app/tests/test_akira_system.py -v
python app/api/evaluate_nfr_metrics.py

# 4. Launch the interactive Web Dashboard (http://127.0.0.1:8000)
python app/run_web_app.py
```
