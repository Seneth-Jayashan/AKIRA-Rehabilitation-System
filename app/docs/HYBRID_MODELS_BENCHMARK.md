# AKIRA 6 All-Hybrid Multi-Modal Models Benchmark Report

**Component:** Knee Rehabilitation Assessment Following Tibial Plateau ORIF  
**Framework:** TensorFlow 2.22.0-rc0 / Keras Functional API  
**Validation Protocol:** 3-Way Subject-Independent Split (663 Inner Train [57 Subj] / 134 Validation [12 Subj] / 200 Unseen Test [18 Subj]) + Full 5-Fold Cross-Validation  

---

## 1. 6-Model Benchmark Leaderboard (Unseen Subjects Held-Out Test Set)

| Rank | Model Architecture | Parameters | Val Score (%) | Exercise Acc (%) | Exercise Macro F1 (%) | Quality Acc (%) | Quality Macro F1 (%) | Combined Score (%) | Host PC Inference Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **Hybrid 1: 1D-CNN + Kinematics** | 55,879 | 80.94% | **96.50%** | 96.95% | **60.47%** | 58.76% | **78.48%** | 0.45 ms |
| **#2** | **Hybrid 4: 1D-ResNet + Kinematics** | 159,431 | 80.75% | **96.00%** | 96.28% | **58.91%** | 57.77% | **77.46%** | 0.56 ms |
| **#3** | **Hybrid 2: BiLSTM + Kinematics** | 103,559 | 78.79% | **95.00%** | 94.87% | **55.04%** | 52.69% | **75.02%** | 2.61 ms |
| **#4** | **Hybrid 5: Self-Attention + Kinematics** | 42,055 | 82.53% | **95.00%** | 95.29% | **54.26%** | 51.89% | **74.63%** | 0.63 ms |
| **#5** | **Hybrid 6: Gated Attention + Kinematics** | 43,335 | 78.69% | **96.00%** | 95.80% | **51.94%** | 50.67% | **73.97%** | 0.47 ms |
| **#6** | **Hybrid 3: CNN-BiLSTM + Kinematics** | 97,223 | 83.18% | **95.00%** | 95.05% | **49.61%** | 47.42% | **72.31%** | 0.75 ms |

---

## 2. 5-Fold Subject-Independent Cross-Validation Summary (All 5 Folds Executed)

- **Exercise Accuracy (5-Fold Mean ± SD):** **96.19% ± 1.76%** (Macro-F1: **96.01%**)
- **Quality Accuracy (5-Fold Mean ± SD):** **58.52% ± 2.93%** (Macro-F1: **57.40%**)
- **Per-Fold Exercise Accuracies:** `[96.5, 98.0, 97.49, 95.98, 92.96]`
- **Per-Fold Quality Accuracies:** `[59.69, 62.07, 53.24, 58.11, 59.49]`

---

## 3. Baseline Comparison (Proposal §3.3 Validation)

| Baseline Model | Modality | Exercise Acc (%) | Exercise Macro F1 (%) | Quality Acc (%) | Quality Macro F1 (%) | Combined Score (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Random Forest (16 Clinical Features Only)** | Input B Only (16 Features) | 74.00% | 72.07% | 46.51% | 44.11% | **60.26%** |
| **Baseline 2: Single-IMU 1D-CNN (Thigh 6-Ch Only)** | Single IMU (Thigh 6-Ch Only) | 97.50% | 97.47% | 46.51% | 45.38% | **72.01%** |
| **Hybrid 1: 1D-CNN + Kinematics (Winning Dual-IMU Hybrid)** | Dual-IMU (12-Ch) + 16 Features | **96.50%** | **96.95%** | **60.47%** | **58.76%** | **78.48%** |

---

## 4. Edge-AI TensorFlow Lite (`.tflite`) Deployment Validation (Proposal §3.5)

- **Quantized Edge Model File:** `api/models/akira_hybrid_cnn_edge.tflite` (**76.91 KB**)
- **Quantization:** `full_int8` (full int8 verified: **True**, float internal tensors: 0)
- **Complete Embedded C Byte Array:** `api/models/edge_akira_classifier.c` (**78,760 bytes** full model array)
- **Host PC TFLite Interpreter Latency:** **0.055 ms** (95th Percentile: **0.066 ms**)
- **On-Device (ESP32) Latency:** Not measured: the ESP32 firmware does not yet run TFLite Micro.
- **TFLite vs. Full Keras Agreement:** **100.0%** (Exercise) / **99.5%** (Quality)
