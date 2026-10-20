# AKIRA 6 All-Hybrid Multi-Modal Models Benchmark Report

**Component:** Knee Rehabilitation Assessment Following Tibial Plateau ORIF  
**Framework:** TensorFlow 2.22.0-rc0 / Keras Functional API  
**Validation Protocol:** 3-Way Subject-Independent Split (663 Inner Train [57 Subj] / 134 Validation [12 Subj] / 200 Unseen Test [18 Subj]) + Full 5-Fold Cross-Validation  

---

## 1. 6-Model Benchmark Leaderboard (Unseen Subjects Held-Out Test Set)

| Rank | Model Architecture | Parameters | Val Score (%) | Exercise Acc (%) | Exercise Macro F1 (%) | Quality Acc (%) | Quality Macro F1 (%) | Combined Score (%) | Host PC Inference Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **Hybrid 5: Self-Attention + Kinematics** | 42,055 | 82.80% | **97.00%** | 97.07% | **58.91%** | 57.63% | **77.96%** | 1.12 ms |
| **#2** | **Hybrid 1: 1D-CNN + Kinematics** | 55,879 | 86.08% | **96.50%** | 96.67% | **58.91%** | 56.97% | **77.71%** | 0.78 ms |
| **#3** | **Hybrid 4: 1D-ResNet + Kinematics** | 159,431 | 81.87% | **96.00%** | 96.03% | **57.36%** | 55.33% | **76.68%** | 0.99 ms |
| **#4** | **Hybrid 2: BiLSTM + Kinematics** | 103,559 | 78.51% | **96.50%** | 96.57% | **54.26%** | 52.62% | **75.38%** | 2.4 ms |
| **#5** | **Hybrid 3: CNN-BiLSTM + Kinematics** | 97,223 | 76.36% | **94.00%** | 93.40% | **55.81%** | 54.38% | **74.91%** | 1.27 ms |
| **#6** | **Hybrid 6: Gated Attention + Kinematics** | 43,335 | 75.52% | **93.50%** | 93.00% | **48.06%** | 44.77% | **70.78%** | 0.84 ms |

---

## 2. 5-Fold Subject-Independent Cross-Validation Summary (All 5 Folds Executed)

- **Exercise Accuracy (5-Fold Mean ± SD):** **96.29% ± 1.44%** (Macro-F1: **95.94%**)
- **Quality Accuracy (5-Fold Mean ± SD):** **57.64% ± 3.97%** (Macro-F1: **56.30%**)
- **Per-Fold Exercise Accuracies:** `[96.0, 98.5, 96.48, 96.48, 93.97]`
- **Per-Fold Quality Accuracies:** `[54.26, 60.69, 52.52, 57.43, 63.29]`

---

## 3. Baseline Comparison (Proposal §3.3 Validation)

| Baseline Model | Modality | Exercise Acc (%) | Exercise Macro F1 (%) | Quality Acc (%) | Quality Macro F1 (%) | Combined Score (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Random Forest (16 Clinical Features Only)** | Input B Only (16 Features) | 74.00% | 72.07% | 46.51% | 44.11% | **60.26%** |
| **Baseline 2: Single-IMU 1D-CNN (Thigh 6-Ch Only)** | Single IMU (Thigh 6-Ch Only) | 97.00% | 97.10% | 46.51% | 43.44% | **71.76%** |
| **Hybrid 5: Self-Attention + Kinematics (Winning Dual-IMU Hybrid)** | Dual-IMU (12-Ch) + 16 Features | **97.00%** | **97.07%** | **58.91%** | **57.63%** | **77.96%** |

---

## 4. Edge-AI TensorFlow Lite (`.tflite`) Deployment Validation (Proposal §3.5)

- **Quantized Edge Model File:** `api/models/akira_hybrid_cnn_edge.tflite` (**76.91 KB**)
- **Quantization:** `full_int8` (full int8 verified: **True**, float internal tensors: 0)
- **Complete Embedded C Byte Array:** `api/models/edge_akira_classifier.c` (**78,760 bytes** full model array)
- **Host PC TFLite Interpreter Latency:** **0.162 ms** (95th Percentile: **0.261 ms**)
- **On-Device (ESP32) Latency:** Not measured: the ESP32 firmware does not yet run TFLite Micro.
- **TFLite vs. Full Keras Agreement:** **100.0%** (Exercise) / **98.5%** (Quality)
