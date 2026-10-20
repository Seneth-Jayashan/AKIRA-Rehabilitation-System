# AKIRA Knee Rehabilitation System: Complete Edge-AI Package

**Component:** Smart Post-ORIF Rehabilitation Ecosystem with Edge-AI Kinematic Tracking  
**Individual Focus:** Knee Joint Rehabilitation Following Tibial Plateau ORIF  
**Student Investigator:** Wickramarathna S A (IT23222540)  
**Framework:** TensorFlow 2.22.0-rc0 / Keras Functional API / FastAPI / Bleak (Native Windows BLE) / TailwindCSS / Chart.js  

---

## 📑 Table of Contents
1. [System Architecture & Methodology (Boxes 01 to 08)](#1-system-architecture--methodology-boxes-01-to-08)
2. [Hardware Setup & Sensor Placement](#2-hardware-setup--sensor-placement)
3. [Smart Sensor Misplacement Diagnostics & Hampel Filtering](#3-smart-sensor-misplacement-diagnostics--hampel-filtering)
4. [Multi-Dataset Harmonization & Leak-Free 3-Way Subject Split](#4-multi-dataset-harmonization--leak-free-3-way-subject-split)
5. [5-Stage Sensor Fusion & 16 Clinical Features (Boxes 02 & 03)](#5-5-stage-sensor-fusion--16-clinical-features-boxes-02--03)
6. [The 7 Official Session Outputs (Box 05)](#6-the-7-official-session-outputs-box-05)
7. [Longitudinal Tracking & Recovery Prediction Engine (Boxes 06, 07 & 08)](#7-longitudinal-tracking--recovery-prediction-engine-boxes-06-07--08)
8. [The 6 Benchmarked Dual-Stream Hybrid Models, 5-Fold CV, Baselines & Edge TFLite](#8-the-6-benchmarked-dual-stream-hybrid-models-5-fold-cv-baselines--edge-tflite)
9. [ESP32 Firmware & Connectivity (Native BLE & Serial)](#9-esp32-firmware--connectivity-native-ble--serial)
10. [Clinical Justification, Validation Honesty & Research Limitations](#10-clinical-justification-validation-honesty--research-limitations)
11. [Package Directory Structure & How to Run](#11-package-directory-structure--how-to-run)

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
  ──► 2.3 Gyro Bias Calibration ──► 2.4 3D Madgwick Fusion (per sensor)
  ──► 2.5 Relative Quaternion (thigh frame) on the Hinge Axis
  ──► 2.6 Zero at Most-Extended Posture ──► Continuous Knee Angle θ(t) ≥ 0
                           │
                           ▼
  [BOX 03: CLINICAL FEATURE EXTRACTION]
  Extracts 16 Biomechanical Parameters (ROM, Peak Depth, Return Deficit, SPARC, Reps, Duration)
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
      (trained only on windows with dataset quality labels; not validated for Step-Up)
                           │
                           ▼
  [BOX 05: THE 7 SESSION OUTPUTS (FULL-SESSION AGGREGATED)]
  1. Recognized Exercise   2. Knee Angle & ROM   3. Peak Depth & Return Deficit   4. Rep Count
  5. Smoothness (SPARC)   6. Duration & Speed   7. Expected AI Quality Score (0–100)
                           │
                           ▼
  [BOX 06 & 07: LONGITUDINAL TRACKING & RECOVERY PREDICTION]
  Per-patient (SHA-256 pseudonymized), per-exercise session history (S1 → Sn),
  regression trajectory (n ≥ 3), un-floored forecast, and milestone countdown
                           │
                           ▼
  [BOX 08: PERSONALIZED REHABILITATION ASSESSMENT]
  Clinician-protocol-driven recommendations (target ROM, safe flexion limit, return-deficit threshold)
```

---

## 2. Hardware Setup & Sensor Placement

The system captures movement from two 6-DOF IMUs (MPU6050) operating on a single I2C bus at **100 Hz (10 ms period)** with hardware **20 Hz Digital Low-Pass Filtering (`REG_CONFIG = 0x04`)** enabled:

* **ESP32 3.3V / GND / GPIO 21 (`SDA`) / GPIO 22 (`SCL`)** wired to both IMUs.
* **Thigh IMU (`0x68`, `AD0 = GND`):** Centered **10 cm above the patella** along the anterior mid-thigh.
* **Shin IMU (`0x69`, `AD0 = 3.3V`):** Centered **10 cm below the patella** along the proximal tibia.

---

## 3. Smart Sensor Misplacement Diagnostics & Hampel Filtering

1. **Non-Destructive Open-Chain Strap Swap Detection (`auto_detect_and_correct_sensor_swap`):**
   * In open-chain Seated Knee Extension (`04_Seated_Knee_Extension`), the thigh rests stationary on the chair (`< 8 deg/s` RMS) while the shin swings vigorously (`> 35 deg/s` RMS).
   * By default (`auto_swap=False`), `auto_detect_and_correct_sensor_swap()` warns the user without mutating the sensor channels, and never modifies closed-chain exercises (`Walking`, `Mini Squat`, `Step-Up`) where both segments rotate simultaneously. Explicit channel swapping is available via `auto_swap=True`.
2. **Deterministic Hardware Axis Mapping & Hampel/MAD Spike Filtering:**
   * Applies a deterministic, configurable axis permutation (`ESP32_HARDWARE_AXIS_PERMUTATION` in `main.py`) and a sliding-window Median Absolute Deviation (Hampel) spike filter (`clean_i2c_glitches_mad`) to remove single-sample I2C transients.

---

## 4. Multi-Dataset Harmonization & Leak-Free 3-Way Subject Split

The corpus (`997` clean 2.0-second windows, `87` subject IDs) is built by `app/api/preprocess_datasets.py`
from **4 public dual-IMU datasets**, with no signal rescaling. Every skipped file or window is logged with its
reason in `app/api/datasets/preprocessing_audit_log.json`.

| Exercise Class | Source Datasets (windows) | Total | Quality `-1 / 0 / 1 / 2` | Median ROM ($^\circ$) |
| :--- | :--- | :---: | :---: | :---: |
| **Walking (0)** | KneE-PAD (190), PHYTMO (76), StrengthSense (41), HuGaDB (45) | **352** | `86 / 114 / 92 / 60` | $69.1^\circ$ |
| **Seated Knee Extension (1)** | KneE-PAD (172), PHYTMO (74) | **246** | `0 / 94 / 56 / 96` | $85.8^\circ$ |
| **Mini Squat (2)** | KneE-PAD (176), PHYTMO (63), StrengthSense (51) | **290** | `83 / 93 / 58 / 56` | $100.0^\circ$ |
| **Step-Up (3)** | StrengthSense stair ascent (80), HuGaDB `going_up` (29) | **109** | `109 / 0 / 0 / 0` | $76.3^\circ$ |

* **Quality grades come only from published dataset labels:** KneE-PAD 0/3/6 → 0 (Good), 1/4/7 → 1
  (incomplete ROM / unloading), 2/5/8 → 2 (postural compensation). PHYTMO correct → 0, KFE wrong → 2
  (plane deviation / thigh-driven / both legs), GAT wrong → 1 (dragging the feet), SQT wrong → unlabelled
  (mixed fault types). StrengthSense and HuGaDB have no correctness annotation → `-1` (unlabelled): these
  windows train exercise recognition only and are excluded from the quality loss and quality metrics.
* **Step-Up proxy:** stair ascent. StrengthSense labels ascent and descent both `a10`; the odd takes are used
  as ascent (across all subjects their shank gyro-Y skewness is +0.84 vs −0.03 for even takes).
* **Rejected windows:** saturated gyroscope (±2000°/s full scale, 320 HuGaDB windows) and ROM outside
  4–155° (221: HuGaDB 206, PHYTMO 11, StrengthSense 4).
* **Strict 3-Way Subject-Independent Split (`GroupKFold`, fixed fold 0, not chosen by its composition):**
  * **Inner Train:** `663` windows (`57` subjects) — fits normalizers and model weights.
  * **Validation:** `134` windows (`12` subjects) — early stopping and model selection only.
  * **Held-Out Test:** `200` windows (`18` unseen subjects, `129` with quality labels) — evaluated once.

---

## 5. 5-Stage Sensor Fusion & 16 Clinical Features (Boxes 02 & 03)

Implemented in [`api/math_feature_extractor.py`](api/math_feature_extractor.py) (details in [`docs/HOW_16_FEATURES_ARE_CALCULATED.md`](docs/HOW_16_FEATURES_ARE_CALCULATED.md)). The knee angle θ(t) is measured from the most-extended posture of each recording (θ ≥ 0). OptiTrack validation on 97 PHYTMO trials: angle-trace error ignoring polarity median **3.43°** (mean 5.84°); polarity agreement 79.4% (see §10B).

1. **Feature 1 (`rom_deg`):** $\max(\theta) - \min(\theta)$ — Active Range of Motion ($^\circ$).
2. **Feature 2 (`peak_depth_deg`):** mean repetition peak when $\ge 2$ reps, otherwise $\max(\theta)$ — Peak Depth ($^\circ$).
3. **Feature 3 (`extension_return_deficit_deg`):** mean repetition trough above the most-extended posture — how completely the knee returns to extension between repetitions ($5.5^\circ \pm 7.6^\circ$ across the dataset). Not an absolute extensor lag (needs a straight-leg calibration pose).
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
| **01** | **Recognized Exercise** | AI Model (Head 1) | Genuine softmax class prediction & confidence % (or Kinematic Protocol Rule Engine for OOD exercises) |
| **02** | **Knee Angle & ROM** | Math Extractor (F1) | Active joint excursion ($\theta_{\max} - \theta_{\min}$) across the full session in degrees |
| **03** | **Movement Depth & Return Deficit** | Math Extractor (F2 & F3) | Mean repetition excursion ($^\circ$) and extension return deficit ($^\circ$) |
| **04** | **Repetition Count** | Math Extractor (F7) | Total segmented repetitions across the full session ($0$ when limb is stationary) |
| **05** | **Movement Smoothness** | Math Extractor (F6) | Un-clipped $[0.0, 1.0]$ normalized SPARC smoothness index |
| **06** | **Duration & Velocity** | Math Extractor (F4, F5, F8) | True active repetition duration ($\text{s}$) and angular velocity ($\text{deg/s}$) |
| **07** | **Quality Score (0–100)** | AI Head 2 Probabilities | $100 \cdot P(\text{Good}) + 50 \cdot P(\text{Needs Imp})$; coarse (test MAE 29.5 points vs grade targets) and not validated for Step-Up |

---

## 7. Longitudinal Tracking & Recovery Prediction Engine (Boxes 06, 07 & 08)

* **Pseudonym-Only Storage & Strict Per-Patient / Per-Exercise Filtering:** Stores only SHA-256 salted pseudonyms (`PT-...`, never raw `patient_id` on disk) using a persistent 256-bit secret salt (`.akira_secret_salt`), and returns an empty session list (`total_sessions: 0`) when no matching sessions exist for a patient/exercise instead of falling back to another patient's data.
* **Real Dates & Post-ORIF Week Calculation:** Stores ISO `session_date` (`YYYY-MM-DD`) and computes post-ORIF week from `surgery_date` when provided.
* **Validated Regression Forecasting ($n \ge 3$):** Requires at least $3$ recorded sessions before fitting linear regression; reports un-clipped $R^2 \in [0, 1]$ and residual $\text{RMSE}$ ($^\circ$), and never floors forecasts at the last value so declining recovery trajectories (`slope < 0`) are detected and flagged.
* **Clinician-Configurable Protocol (`GET/POST /api/clinician/protocol`):** Clinicians can customize `target_rom_deg`, `max_safe_flexion_deg`, `max_extensor_lag_deg` (applied to the extension return deficit), and `sessions_per_week` per exercise.
* **Reference sessions:** the five built-in example sessions are stored under `DEMO_REF`, flagged `is_reference_baseline`, and never mixed into a real patient's trend.

---

## 8. The 6 Benchmarked Dual-Stream Hybrid Models, 5-Fold CV, Baselines & Edge TFLite

Evaluated on the **unseen-subject held-out test partition (`200` windows from `18` unseen subjects; quality on the
`129` labelled windows)** (`api/models/test_metrics_summary.json`, full report in
[`docs/HYBRID_MODELS_BENCHMARK.md`](docs/HYBRID_MODELS_BENCHMARK.md)):

### A. Leaderboard (Unseen Subjects)

| Rank | Model Architecture | Parameters | Val Score (%) | Exercise Acc (%) | Exercise Macro F1 (%) | Quality Acc (%) | Quality Macro F1 (%) | Combined Score (%) | Host PC Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **Hybrid 5: Self-Attention + Kinematics** | 42,055 | 82.80% | **97.00%** | **97.07%** | **58.91%** | **57.63%** | **77.96%** | ~1.12 ms |
| **#2** | **Hybrid 1: 1D-CNN + Kinematics (Edge Model)** | 55,879 | **86.08%** | **96.50%** | **96.67%** | **58.91%** | 56.97% | **77.71%** | **0.162 ms** |
| **#3** | **Hybrid 4: 1D-ResNet + Kinematics** | 159,431 | 81.87% | 96.00% | 96.03% | 57.36% | 55.33% | **76.68%** | ~0.99 ms |
| **#4** | **Hybrid 2: BiLSTM + Kinematics** | 103,559 | 78.51% | 96.50% | 96.57% | 54.26% | 52.62% | **75.38%** | ~2.40 ms |
| **#5** | **Hybrid 3: CNN-BiLSTM + Kinematics** | 97,223 | 76.36% | 94.00% | 93.40% | 55.81% | 54.38% | **74.91%** | ~1.27 ms |
| **#6** | **Hybrid 6: Gated Attention + Kinematics** | 43,335 | 75.52% | 93.50% | 93.00% | 48.06% | 44.77% | **70.78%** | ~0.84 ms |

### B. 5-Fold Subject-Independent Cross-Validation (Hybrid 1, all 5 folds)

* **Exercise Accuracy:** **96.29% ± 1.44%** (Macro-F1 95.94%)
* **Quality Accuracy (labelled windows):** **57.64% ± 3.97%** (Macro-F1 56.30%)

### C. Baselines (Proposal §3.3)

| Model | Input | Exercise Acc (%) | Quality Acc (%) | Combined (%) |
| :--- | :--- | :---: | :---: | :---: |
| Baseline 1: Random Forest | 16 clinical features | 74.00 | 46.51 | 60.26 |
| Baseline 2: Single-IMU 1D-CNN | Thigh IMU only (6-ch) | 97.00 | 46.51 | 71.76 |
| **Hybrid 1: 1D-CNN + Kinematics** | **Dual IMU (12-ch) + 16 features** | **96.50** | **58.91** | **77.71** |

### D. Edge-AI TensorFlow Lite (Proposal §3.5)

* **Model:** `api/models/akira_hybrid_cnn_edge.tflite` (**76.91 KB**), **full int8** quantization calibrated on
  training windows (verified: 0 float internal tensors; float I/O through QUANTIZE/DEQUANTIZE builtins).
* **C array:** `api/models/edge_akira_classifier.c` (**78,760 bytes**, complete model).
* **Host PC interpreter latency:** **0.096 ms**; agreement with Keras 100.0% (exercise) / 99.5% (quality).
* **On-device ESP32 latency and RAM:** not measured — the firmware does not yet link TFLite Micro.

---

## 9. ESP32 Firmware & Connectivity (Native BLE & Serial)

The ESP32 firmware sketch (`app/api/esp32_firmware/akira_knee_esp32.ino`) configures both MPU6050 sensors at $\pm 4\text{ g}$ (8192 LSB/g), $\pm 500\text{ deg/s}$, $20\text{ Hz}$ hardware DLPF (`0x04`), and $100\text{ kHz}$ I2C clock, streaming at $100\text{ Hz}$ over Nordic UART BLE (`6e400001-...`) and USB Serial (`115200` baud).

---

## 10. Clinical Justification, Validation Honesty & Research Limitations

### A. What the second IMU adds
For **exercise recognition** a thigh-only IMU matches the dual-IMU models (97.5% vs 96.5%). The shin IMU is
required for **joint kinematics** (knee angle, ROM, depth, extension return deficit come from the relative
thigh–shin orientation) and improves quality grading (56.6% vs 48.1%).

### B. Optical Motion Capture (OptiTrack) Validation
`app/api/models/optitrack_validation_metrics.json`: 97 PHYTMO trials evaluated (34 optical drop-outs, 7 failed
synchronisations and 5 files without rigid bodies excluded and logged; no trial excluded for disagreeing with
the IMU). On the 93 windows the system accepts:
* **Angle-trace error ignoring polarity:** mean **5.84°**, median **3.43°**.
* **As output:** mean **12.45°**, median **4.17°**. Polarity agreement **79.4%**.
* IMU data alone cannot determine which direction is flexion without a known sensor orientation; the angle
  direction is a convention (usually extension excursion for seated knee extension).

### C. Hardware Latency & Dual-IMU Synchronization
* **Latency:** ~12.8 ms per window on the host PC; **ESP32 on-device latency not measured**.
* **Synchronization:** **not measured** on AKIRA hardware (recordings do not keep per-sample device
  timestamps); ~1.3 ms expected from the shared I2C bus is a design figure only.

### D. Public Benchmark Dataset Limitations (Threats to Validity)
* **Mini Squat** data are squats of 75–115° median ROM (no public mini-squat recordings); **Step-Up** data are
  stair ascent. Both are proxies for the clinical exercises.
* **Quality** labels are dataset fault codes, available for Walking, Seated Knee Extension and Mini Squat
  only; quality accuracy is 52–57% and Step-Up quality is not validated.
* Volunteers are healthy or general knee-pathology participants, not tibial plateau ORIF patients; collecting
  target-population data with the AKIRA device is still required.

### E. Security
Patient IDs are stored only as salted SHA-256 pseudonyms. Set `AKIRA_API_KEY` to require the
`X-AKIRA-API-Key` header on patient-data endpoints; no key is stored in the code.

---

## 11. Package Directory Structure & How to Run

```powershell
# 1. Rebuild Harmonized Datasets & Multi-Trial OptiTrack Validation
python app/api/preprocess_datasets.py

# 2. Train All 6 Hybrid Models, Baselines, 5-Fold CV & Export Full-Int8 TFLite Edge Model
python app/api/train_models.py

# 3. Run Non-Functional Requirements (NFR) Evaluation
python app/api/evaluate_nfr_metrics.py

# 4. Run Self-Contained Automated Pytest Suite
python -m pytest app/tests/test_akira_system.py -v

# 5. Launch Clinical Web Application Dashboard
python app/run_web_app.py
```
