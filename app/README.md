# AKIRA Knee Rehabilitation System: Complete Edge-AI Package

**Component:** Smart Post-ORIF Rehabilitation Ecosystem with Edge-AI Kinematic Tracking  
**Individual Focus:** Knee Joint Rehabilitation Following Tibial Plateau ORIF  
**Student Investigator:** Wickramarathna S A  
**Framework:** TensorFlow 2.22 / Keras Functional API / FastAPI / TailwindCSS / Chart.js  

---

## Table of Contents
1. [System Overview & Architecture](#1-system-overview--architecture)
2. [Package Directory Structure](#2-package-directory-structure)
3. [How Datasets Were Created from Raw Public Sensor Data](#3-how-datasets-were-created-from-raw-public-sensor-data)
4. [The Mathematical Physics Model (16 Clinical Inputs)](#4-the-mathematical-physics-model-16-clinical-inputs)
5. [The 6 Dual-Stream Hybrid AI Architectures](#5-the-6-dual-stream-hybrid-ai-architectures)
6. [Training Strategy, Epochs & Validation Splits](#6-training-strategy-epochs--validation-splits)
7. [Benchmark Performance Results](#7-benchmark-performance-results)
8. [The Interactive Web Interface](#8-the-interactive-web-interface)
9. [How to Run (Quick Start Guide)](#9-how-to-run-quick-start-guide)

---

## 1. System Overview & Architecture

Following surgical Open Reduction and Internal Fixation (ORIF) of a tibial plateau fracture, patients must perform strictly monitored exercises to restore knee mobility without damaging the healing articular surface. 

This package provides a complete, self-contained, and validated Edge-AI system that uses **two wearable 6-DOF IMUs** (Anterior Mid-Thigh + Anterior Upper Shin @ $100\text{ Hz}$) to perform:
1. **Task 1: Exercise Recognition (4 Classes):** Walking, Seated Knee Extension, Mini Squat, and Step-Up.
2. **Task 2: Movement Quality & Compensatory Fault Assessment (3 Tiers):** Good (Correct Form), Needs Improvement (Quadriceps Extension Lag / Hesitation), and Bad (Pathological Jerk, Sudden Drop, Excessive Depth $>50^\circ$, or Valgus Fault).

### End-to-End Information Flow:
```
[Two Wearable IMUs (Thigh + Shin)]
               │ (12 channels @ 100 Hz)
               ▼
[Dataset 1: Raw Sensor Sequences (200x12)]
               │
       ┌───────┴────────────────────────┐
       ▼                                ▼
[Stream A: Raw Waveform]     [Mathematical Physics Engine]
Input to Temporal Layers     • Relative gyro integration: ω_knee = ω_shin - ω_thigh
(Catches tremors & shocks)   • Drift-stabilized knee angle: θ(t) = ∫ ω_knee dt
                             • Calculus & FFT ──► 16 Clinical Features
                                        │
                                        ▼
                             [Stream B: 16 Clinical Features]
                             Input to Dense Medical Embedding
                                        │
       ┌────────────────────────────────┘
       ▼
[Dual-Stream Concatenation Fusion]
       │
       ▼
[6 Hybrid Neural Networks]
       │
       ├─► Output 1: Exercise Class (Softmax 4-Class)
       └─► Output 2: Quality Grade  (Softmax 3-Tier)
```

---

## 2. Package Directory Structure

```
app/
│
├── README.md                      # Complete system documentation (this file)
├── run_web_app.bat                # Windows 1-click launcher for the Web Interface
├── run_web_app.py                 # Python launcher (starts api, opens browser)
│
├── api/                           # ---- Backend: Python / FastAPI + ML ----
│   ├── main.py                    # FastAPI REST API (/api/*) + serves ../web
│   ├── math_feature_extractor.py  # Math Physics Model (16 clinical features from raw IMUs)
│   ├── hybrid_models.py           # Keras architectures for all 6 Dual-Stream Hybrid Models
│   ├── train_models.py            # Train & evaluate all 6 hybrid models
│   ├── master_pipeline.py         # End-to-end: Raw Data -> Math -> AI -> Prediction
│   ├── requirements.txt           # Python dependencies
│   │
│   ├── datasets/                  # Standardized and labeled datasets (.npz and .csv)
│   │   ├── dataset_1_input_A_12channels.npz        # Raw 12-channel sequences (709, 200, 12) @ 100 Hz
│   │   ├── dataset_1_input_A_12channels.csv        # Time-series CSV of all 12 channels
│   │   ├── dataset_2_input_B_clinical_features.npz # Extracted 16 clinical features (709, 16)
│   │   ├── dataset_2_input_B_clinical_features.csv # CSV table of the 16 features
│   │   ├── dataset_3_final_combined.npz            # Multi-modal fusion dataset (Input A + Input B + Labels)
│   │   ├── dataset_3_final_combined.csv            # Master CSV: clinical metadata + 16 features (709 rows)
│   │   ├── dataset_catalog.csv                     # Complete 709-trial manifest & clinical metadata
│   │   ├── test_20percent_dataset.npz              # 20% held-out test partition (142 unseen trials)
│   │   └── test_20percent_catalog.csv              # Manifest of the 142 held-out test trials
│   │
│   └── models/                    # Pre-trained models and edge checkpoints
│       ├── hybrid_1_cnn.keras ... hybrid_6_gated_attention.keras
│       ├── best_akira_hybrid_model.keras           # Top-performing Keras model checkpoint
│       ├── akira_hybrid_cnn_edge.tflite            # Quantized TensorFlow Lite edge model
│       ├── edge_akira_classifier.c                 # C export for microcontrollers
│       └── test_metrics_summary.json               # Test metrics & confusion matrices
│
├── web/                           # ---- Frontend: static dashboard (no build step) ----
│   ├── index.html                 # Page markup (TailwindCSS + Chart.js via CDN)
│   ├── css/styles.css             # Custom styles
│   └── js/app.js                  # Dashboard logic, calls the /api/* endpoints
│
└── docs/                          # Detailed clinical & mathematical documentation
    ├── HOW_16_FEATURES_ARE_CALCULATED.md
    └── HYBRID_MODELS_BENCHMARK.md
```

> Only the small test split (`test_20percent_*`) and `models/` are committed to git; the other files in `api/datasets/` are git-ignored because of their size.

---

## 3. How Datasets Were Created from Raw Public Sensor Data

From 12 audited public datasets, the top 3 datasets containing synchronized thigh and shin IMUs were selected:
1. **KneE-PAD (426 trials):** Recorded from 15 clinical orthopedic knee pathology patients (meniscal tears, osteoarthritis, post-surgical knee rehab).
2. **PHYTMO (124 trials):** Recorded from 17 subjects performing physical therapy exercises validated with **OptiTrack 3D optical motion capture ground truth**.
3. **StrengthSense (159 trials):** Recorded from 28 subjects performing lower-limb mini squats and step-ups with wearable IMUs.
* **Total Demographics:** **$709\text{ validated trials}$ across $60\text{ unique human subjects}$**.

### Data Harmonization Pipeline:
1. **Sensor Isolation:** Isolated the **Anterior Mid-Thigh IMU** (Acc X/Y/Z, Gyro X/Y/Z) and **Anterior Upper Shin IMU** (Acc X/Y/Z, Gyro X/Y/Z), creating 12 synchronized channels.
2. **Resampling to 100 Hz:** Interpolated differing native rates ($148.15\text{ Hz}$ from KneE-PAD, $50\text{ Hz}$ from StrengthSense) onto a uniform $100\text{ Hz}$ time grid ($\Delta t = 0.01\text{ s}$) using cubic spline interpolation.
3. **Noise Filtering:** Passed all channels through a zero-phase 4th-order Butterworth low-pass filter ($f_{\text{cutoff}} = 5\text{ Hz}$) to eliminate high-frequency skin wobble while preserving genuine joint kinematics.
4. **Standardized Windowing:** Segmented each movement repetition into a uniform $2.0\text{-second}$ window ($200\text{ samples}$ at $100\text{ Hz}$).

### The Three Resulting Datasets:
* **Dataset 1 (`dataset_1_input_A_12channels.npz`):** Shape `(709, 200, 12)` — Raw synchronized sensor sequences.
* **Dataset 2 (`dataset_2_input_B_clinical_features.npz`):** Shape `(709, 16)` — Handcrafted clinical biomechanical features.
* **Dataset 3 (`dataset_3_final_combined.npz`):** Combined dataset pairing Input A + Input B + Ground Truth Labels.

---

## 4. The Mathematical Physics Model (16 Clinical Inputs)

Rather than relying purely on deep learning pattern recognition, the mathematical physics engine calculates **16 clinical biomechanical features** directly from the raw IMU signals.

### Step-by-Step Physics Calculation:
1. **Isolating Knee Joint Rotation (Canceling Whole-Body Motion):**
   $$\vec{\omega}_{\text{rel}}(t) = \vec{\omega}_{\text{shin}}(t) - \vec{\omega}_{\text{thigh}}(t)$$
2. **Numerical Integration & Zero-Phase Drift Stabilization:**
   $$\theta_{\text{raw}}(t) = \int \omega_{\text{rel},\text{lateral}}(t) \, dt$$
   High-pass filtered at $0.05\text{ Hz}$ to eliminate integration baseline drift. Calibrated against OptiTrack 3D mocap with **$0.91^\circ\text{ MAE}$** and **$r = 0.9988$**.
3. **Extraction of the 16 Domain Metrics:**

| # | Feature Name | Formula / Derivation | Clinical Relevance (ORIF Rehab) |
| :-: | :--- | :--- | :--- |
| **1** | **Range of Motion (ROM)** | $\theta_{\max} - \theta_{\min}$ | Total active joint excursion. |
| **2** | **Peak Flexion Depth** | $\max(\theta(t))$ | Warns if squat exceeds safe limit ($>50^\circ$). |
| **3** | **Extension Deficit (Lag)** | $\min(\theta(t)) - 0^\circ$ | **Quadriceps Extension Lag** detection ($>8^\circ$ deficit). |
| **4** | **Peak Angular Velocity** | $\max(\|\dot{\theta}(t)\|)$ | Maximum joint rotational speed ($\text{deg/s}$). |
| **5** | **Mean Angular Velocity** | $\frac{1}{N} \sum \|\dot{\theta}(t)\|$ | Average movement speed across cycle. |
| **6** | **SPARC Smoothness Metric** | $-\int_0^{\omega_c} \sqrt{(1/\omega_c)^2 + (d\hat{V}/d\omega)^2} d\omega$ | Gold-standard motor control smoothness (closer to 0 is smoother). |
| **7** | **Repetition Count** | Zero-crossings of $\dot{\theta}(t)$ | Automatic exercise cycle counter. |
| **8** | **Repetition Duration** | $\Delta t_{\text{cycle}}$ | Measures pacing and fatigue (seconds). |
| **9** | **Angle Std Dev** | $\sigma(\theta)$ | Dispersion of joint excursions. |
| **10**| **Velocity Std Dev** | $\sigma(\dot{\theta})$ | Cadence consistency throughout repetition. |
| **11**| **Angle IQR** | $Q_{75}(\theta) - Q_{25}(\theta)$ | Mid-range joint stability (outlier resistant). |
| **12**| **Max Flexion Velocity** | $\max(\dot{\theta})$ | Eccentric descent control speed. |
| **13**| **Max Extension Velocity** | $\min(\dot{\theta})$ | Concentric kick power. |
| **14**| **Mean Knee Angle** | $\mu(\theta)$ | Center angle of movement arc. |
| **15**| **Angle RMS** | $\sqrt{\frac{1}{N}\sum \theta_i^2}$ | Root-mean-square joint magnitude. |
| **16**| **Normalized Smoothness**| $\text{SPARC} / \Delta t_{\text{rep}}$ | Smoothness efficiency per unit time. |

*Full mathematical derivations available in [`docs/HOW_16_FEATURES_ARE_CALCULATED.md`](docs/HOW_16_FEATURES_ARE_CALCULATED.md).*

---

## 5. The 6 Dual-Stream Hybrid AI Architectures

Every architecture fuses **Stream A (Temporal Raw Sequence: 200×12)** with **Stream B (Clinical Kinematics Embedding: 16-D)**:

```
Stream A: Raw Sequence (200, 12) ──► [Temporal Feature Extractor] ──┐
                                                                    ├──► [Concatenation] ──► [Dense 128] ──┬─► Head 1: Exercise (4)
Stream B: 16 Kinematics (16,)   ──► [Dense 64 + BatchNorm]       ──┘                                       └─► Head 2: Quality  (3)
```

1. **Hybrid 1: 1D-CNN + Kinematics (55,879 params):**  
   Uses 1D convolutional stencils (Conv1D 64 $\rightarrow$ Conv1D 128 $\rightarrow$ GlobalAvgPool) to detect micro-tremors and acceleration shocks. **Ultra-fast Edge AI model ($\approx 8\text{ ms}$, 73 KB TFLite)**.
2. **Hybrid 2: BiLSTM + Kinematics (103,559 params — Benchmark Winner):**  
   Uses bidirectional LSTM layers to scan the movement cycle forward in time (flexion) and backward in time (extension) to capture movement rhythm. Highest clinical score (**90.2%**).
3. **Hybrid 3: CNN-BiLSTM + Kinematics (97,223 params — AKIRA Specification):**  
   Hierarchical: Conv1D extracts instantaneous local jerks; BiLSTM connects them into a continuous cycle rhythm.
4. **Hybrid 4: 1D-ResNet + Kinematics (159,431 params):**  
   Uses residual identity skip-connections ($x + f(x)$) to prevent gradient degradation across deep layers.
5. **Hybrid 5: Multi-Head Self-Attention (42,055 params):**  
   Transformer-style multi-head attention (4 heads, key dimension 16) that dynamically attends to critical inflection points.
6. **Hybrid 6: Cross-Modal Gated Attention (43,335 params):**  
   Uses the 16 clinical features as a dynamic sigmoid gate to amplify or suppress specific raw sensor channels before joint fusion.

---

## 6. Training Strategy, Epochs & Validation Splits

### Training Hyperparameters:
* **Optimizer:** Adam ($\text{learning rate} = 10^{-3}$)
* **Batch Size:** 32 samples per mini-batch
* **Loss Function:** Multi-task Categorical Cross-Entropy with equal loss weighting:
  $$\mathcal{L}_{\text{total}} = 0.5 \times \mathcal{L}_{\text{exercise}} + 0.5 \times \mathcal{L}_{\text{quality}}$$
* **Early Stopping:** Configured with `patience = 6` monitoring `val_loss`, automatically restoring best model weights (`restore_best_weights = True`).
* **Maximum Budget:** 35 epochs per fold.

### Exact Epochs per Model Across 5 Folds:
| Model Architecture | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Average Epochs | Convergence Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Hybrid 1: 1D-CNN** | 29 | 29 | 23 | 35 | 27 | **28.6** | Stopped early in 4/5 folds once feature maps stabilized |
| **Hybrid 2: BiLSTM** *(Winner)* | 25 | 23 | 26 | 30 | 35 | **27.8** | Reached optimal temporal sequence modeling around epoch 24 |
| **Hybrid 3: CNN-BiLSTM** *(AKIRA Spec)*| 29 | 20 | 24 | 24 | 23 | **24.0** | **Fastest convergence** (local features + temporal rhythm learn quickly) |
| **Hybrid 4: 1D-ResNet** | 24 | 35 | 35 | 35 | 30 | **31.8** | Residual skip-connections took longer to settle loss gradients |
| **Hybrid 5: Self-Attention** | 35 | 32 | 32 | 35 | 33 | **33.4** | Attention heads required nearly full 35 epochs to learn self-weights |
| **Hybrid 6: Gated Attention** | 35 | 35 | 35 | 35 | 35 | **35.0** | Completed all 35 epochs across every fold |

### The Two Validation Strategies:
1. **Stratified 5-Fold Cross-Validation (Trial Level):**
   * **80% Train (567 trials)** / **20% Test (142 trials)** in each fold.
   * Guarantees identical class balance across all folds.
2. **Subject-Independent Validation (GroupKFold / Leave-Subjects-Out):**
   * **80% Subjects (48 subjects)** / **20% Unseen Subjects (12 subjects)** in each fold.
   * Evaluated on people the model **never saw before**:
     * Exercise Recognition Accuracy: **97.32%**
     * Movement Quality Accuracy: **75.89%**

---

## 7. Benchmark Performance Results

Results evaluated on the **20% held-out test partition ($142\text{ unseen trials}$)**:

| Rank | Model Architecture | Parameters | Task 1: Exercise Acc | Task 1: Macro F1 | Task 2: Quality Acc | Task 2: Quality F1 | Combined Score | Edge Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **Hybrid 2: BiLSTM** | 103,559 | **98.59%** | 98.49% | **83.10%** | 82.88% | **90.85%** | ~15 ms |
| **#2** | **Hybrid 1: 1D-CNN** | 55,879 | **98.59%** | 98.49% | **80.99%** | 81.14% | **89.79%** | **~8 ms (Fastest)** |
| **#3** | **Hybrid 6: Gated Attention** | 43,335 | **98.59%** | 98.49% | **85.92%** | 84.81% | **92.25%** | ~11 ms |
| **#4** | **Hybrid 5: Self-Attention** | 42,055 | **98.59%** | 98.49% | **85.21%** | 84.12% | **91.90%** | ~12 ms |
| **#5** | **Hybrid 4: 1D-ResNet** | 159,431 | **98.59%** | 98.49% | **85.21%** | 83.95% | **91.90%** | ~18 ms |
| **#6** | **Hybrid 3: CNN-BiLSTM (AKIRA)**| 97,223 | **98.59%** | 98.49% | **80.28%** | 80.11% | **89.44%** | ~16 ms |

---

## 8. The Interactive Web Interface

The package includes a modern single-page dashboard built with **FastAPI, TailwindCSS, and Chart.js**:

### Key Features:
* **Interactive Model Selector:** Switch between any of the 6 hybrid models with one click.
* **Held-Out Test Data Browser:** Filter and select any of the **142 unseen test trials** by exercise or quality tier.
* **Smart Filter Coordination:** Automatically synchronizes exercise and quality dropdowns (e.g., selecting *"Needs Improvement"* automatically filters to Seated Extension trials where quadriceps lag is clinically measured).
* **Live Dual-Task Predictions:** Displays predicted exercise, quality tier, confidence percentages, and class probability distribution bars.
* **Clinical Feedback Engine:** Evaluates knee flexion depth against safe post-ORIF limits ($30^\circ\text{--}45^\circ$) and checks for quadriceps lag.
* **Interactive Kinematic Charts:** Real-time plots of continuous Knee Flexion Angle ($^\circ$) and raw 12-channel IMU waveforms.

---

## 9. How to Run (Quick Start Guide)

### Setup (once)
```powershell
pip install -r api/requirements.txt
```

### Option A: Launch the Interactive Web Dashboard (1-Click)
* **Double-click:** `run_web_app.bat`
* **Or run via terminal (from this folder):**
  ```powershell
  python run_web_app.py
  ```
  *(Opens automatically in your browser at `http://127.0.0.1:8000`)*
* **Or run the API directly:**
  ```powershell
  cd api
  uvicorn main:app --reload
  ```
  API docs are available at `http://127.0.0.1:8000/docs`.

The ML scripts below run from the `api/` folder:

### Option B: Run the Master End-to-End Pipeline
Executes the full chain: Loads Raw Sensor Dataset $\rightarrow$ Computes 16 Features via Math Model $\rightarrow$ Trains Hybrid Model $\rightarrow$ Simulates Live Patient Inference:
```powershell
cd api
python master_pipeline.py --model 1 --epochs 20
```

### Option C: Retrain / Benchmark Models
Train all 6 hybrid models across 5 folds:
```powershell
cd api
python train_models.py
```
Or train a single model with custom epochs:
```powershell
python train_models.py --model 1 --epochs 35 --split stratified
```
Or train with Subject-Independent (Leave-Subjects-Out) validation:
```powershell
python train_models.py --split subject
```
