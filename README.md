# AKIRA Knee Rehabilitation System: Smart Post-ORIF Edge-AI Kinematic Tracking

**Project:** Smart Post-ORIF Rehabilitation Ecosystem with Edge-AI Kinematic Tracking (AKIRA)  
**Component Focus:** Knee Joint Rehabilitation Following Tibial Plateau Open Reduction and Internal Fixation (ORIF)  
**Investigator:** Wickramarathna S A (IT23222540)  

---

## 1. Executive Summary

This repository contains the complete hardware firmware, multi-dataset signal processing pipeline, 3D sensor fusion & 16-feature biomechanical extractor, leak-free subject-independent Dual-Stream Hybrid Deep Learning models, quantized TensorFlow Lite (`.tflite`) edge deployment artifacts, and FastAPI/Chart.js clinical web application for **Post-ORIF Knee Rehabilitation**.

### Key Engineering & Clinical Highlights
* **Multi-Dataset Harmonized Corpus (`1,036` Trials, `77` Subjects):** Integrates **KneE-PAD** (general knee rehabilitation), **PHYTMO** (optical mocap & IMU knee kinematics), **StrengthSense** (lower-limb strength & stair negotiation), and **HuGaDB** (human gait & stair ascent) so all 4 target exercises (`Walking`, `Seated Knee Extension`, `Mini Squat`, `Step-Up`) span multiple datasets and all 3 quality grades (`Good`, `Needs Improvement`, `Bad`).
* **5-Stage Kinematic Sensor Fusion (`math_feature_extractor.py`):** Implements Cubic-Spline $100\text{ Hz}$ resampling, 4th-order $5\text{ Hz}$ Butterworth low-pass filtering, static sensor-to-segment tare & gyro bias calibration, 3D Madgwick/Complementary sensor fusion, and 3D SVD principal hinge-axis projection.
* **Optical Motion Capture Validation:** Validated against synchronized **OptiTrack 3D Optical Motion Capture** (`A01KFER0_1`): $\text{MAE} = 0.91^\circ$, $\text{RMSE} = 1.12^\circ$, Pearson $r = 0.9988$.
* **Distinct 16 Biomechanical Features:** Independently quantifies Active ROM (Feature 1), Peak Flexion Depth (Feature 2, mean $|\text{F1}-\text{F2}| = 4.47^\circ$), Quadriceps Terminal Extension Lag (Feature 3, $5.42^\circ \pm 5.55^\circ$), SPARC Smoothness (Feature 6), Adaptive Repetition Count (Feature 7), and Standardized Flexion/Extension Angular Velocities (Features 12 & 13).
* **Leak-Free Subject-Independent Evaluation (`train_models.py`):** Uses `GroupKFold` by subject (`61` train subjects / `16` unseen test subjects, `0` subject overlap) with `StandardScaler` and channel normalizers fit strictly on the training split.
* **Quantized Edge-AI Deployment (`.tflite` & C Array):** Exports `akira_hybrid_cnn_edge.tflite` (**73.29 KB**, **0.208 ms** mean inference latency, **100.0%** Exercise / **99.03%** Quality agreement with Keras) and `edge_akira_classifier.c`.

---

## 2. Unseen-Subject Benchmark Leaderboard (`207` Test Trials, `16` Unseen Subjects)

| Rank | Model Architecture | Parameters | Exercise Acc (%) | Exercise Macro F1 (%) | Quality Acc (%) | Quality Macro F1 (%) | Combined Score (%) | Inference Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **Hybrid 5: Self-Attention + Kinematics** | 42,055 | 89.86% | 89.73% | **57.49%** | **54.45%** | **73.67%** | ~3.01 ms |
| **#2** | **Hybrid 1: 1D-CNN + Kinematics (Edge Model)** | 55,879 | **92.27%** | **91.85%** | 53.14% | 50.68% | **72.71%** | ~2.12 ms |
| **#3** | **Hybrid 4: 1D-ResNet + Kinematics** | 159,431 | 90.34% | 89.70% | 54.59% | 51.43% | **72.46%** | ~3.31 ms |
| **#4** | **Hybrid 6: Gated Attention + Kinematics** | 43,335 | 89.37% | 89.13% | 52.17% | 49.14% | **70.77%** | ~2.76 ms |
| **#5** | **Hybrid 3: CNN-BiLSTM + Kinematics** | 97,223 | 91.79% | 91.53% | 44.44% | 42.91% | **68.12%** | ~5.03 ms |
| **#6** | **Hybrid 2: BiLSTM + Kinematics** | 103,559 | 90.82% | 90.28% | 44.44% | 42.56% | **67.63%** | ~6.94 ms |
| — | *Baseline 2: Single-IMU 1D-CNN (Thigh 6-Ch)* | 27,271 | 86.96% | 86.25% | 47.34% | 44.57% | *67.15%* | ~1.40 ms |
| — | *Baseline 1: Random Forest (16 Features)* | — | 81.64% | 82.89% | 50.72% | 46.84% | *66.18%* | ~0.85 ms |

---

## 3. Clinician Rehabilitation Protocols & Safe Flexion Limits

Consistent across backend code (`main.py`), documentation, and clinician configuration endpoints (`/api/clinician/protocol`):

| Exercise | Target ROM ($^\circ$) | Max Safe Flexion Ceiling ($^\circ$) | Max Extensor Lag Threshold ($^\circ$) | Recommended Frequency |
| :--- | :---: | :---: | :---: | :---: |
| **Mini Squat** | $42.0^\circ$ | **$45.0^\circ$ ($30^\circ\text{--}45^\circ$ Safe Zone)** | $\le 5.0^\circ$ | 3 sessions / week |
| **Seated Knee Extension** | $90.0^\circ$ | $100.0^\circ$ | $\le 5.0^\circ$ | 3 sessions / week |
| **Step-Up** | $65.0^\circ$ | $75.0^\circ$ | $\le 5.0^\circ$ | 2 sessions / week |
| **Walking** | $60.0^\circ$ | $70.0^\circ$ | $\le 5.0^\circ$ | 3 sessions / week |

---

## 4. Repository Structure & Quick Start

* [`app/README.md`](app/README.md) — Detailed architecture, hardware pinout, and operational guide.
* [`app/docs/HOW_16_FEATURES_ARE_CALCULATED.md`](app/docs/HOW_16_FEATURES_ARE_CALCULATED.md) — Mathematical equations for Steps 2.1–2.5 and all 16 clinical features.
* [`app/docs/HYBRID_MODELS_BENCHMARK.md`](app/docs/HYBRID_MODELS_BENCHMARK.md) — Synchronized unseen-subject benchmark, baselines, and TFLite report.
* [`app/api/preprocess_datasets.py`](app/api/preprocess_datasets.py) — Reproducible multi-dataset builder (`KneE-PAD`, `PHYTMO`, `StrengthSense`, `HuGaDB`) & OptiTrack validator.
* [`app/api/math_feature_extractor.py`](app/api/math_feature_extractor.py) — 3D Sensor Fusion, 16-Feature Extractor, and Auto-Swap Detection Engine.
* [`app/api/train_models.py`](app/api/train_models.py) — Leak-free `GroupKFold` trainer, baseline evaluator, and `.tflite` exporter.
* [`app/api/main.py`](app/api/main.py) — FastAPI server, live ESP32 BLE/Serial/Wi-Fi recorder, and longitudinal recovery analytics.
* [`app/tests/test_akira_system.py`](app/tests/test_akira_system.py) — Automated `pytest` verification suite.

```powershell
# Run the Web Application Dashboard
python app/run_web_app.py
```
