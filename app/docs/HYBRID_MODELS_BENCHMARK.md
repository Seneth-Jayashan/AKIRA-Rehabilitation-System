# AKIRA 6 All-Hybrid Multi-Modal Models Benchmark Report

**Component:** Knee Rehabilitation Assessment Following Tibial Plateau ORIF  
**Framework:** TensorFlow 2.22.0-rc0 / Keras Functional API  
**Validation Protocol:** Strict Subject-Independent Held-Out Test Split (Zero Subject Overlap, Leak-Free Train-Only Scalers)  
**Dataset Partition:** 829 Training Trials (61 Subjects) / 207 Unseen Test Trials (16 Subjects)  

---

## 1. 6-Model Benchmark Leaderboard (Unseen Subjects Test Partition)

| Rank | Model Architecture | Parameters | Exercise Acc (%) | Exercise Macro F1 (%) | Quality Acc (%) | Quality Macro F1 (%) | Combined Score (%) | Inference Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **Hybrid 5: Self-Attention + Kinematics** | 42,055 | **89.86%** | 89.73% | **57.49%** | 54.45% | **73.67%** | ~3.99 ms |
| **#2** | **Hybrid 1: 1D-CNN + Kinematics** | 55,879 | **92.27%** | 91.85% | **53.14%** | 50.68% | **72.71%** | ~2.78 ms |
| **#3** | **Hybrid 4: 1D-ResNet + Kinematics** | 159,431 | **90.34%** | 89.70% | **54.59%** | 51.43% | **72.46%** | ~3.79 ms |
| **#4** | **Hybrid 6: Gated Attention + Kinematics** | 43,335 | **89.37%** | 89.13% | **52.17%** | 49.14% | **70.77%** | ~2.65 ms |
| **#5** | **Hybrid 3: CNN-BiLSTM + Kinematics** | 97,223 | **91.79%** | 91.53% | **44.44%** | 42.91% | **68.12%** | ~6.94 ms |
| **#6** | **Hybrid 2: BiLSTM + Kinematics** | 103,559 | **90.82%** | 90.28% | **44.44%** | 42.56% | **67.63%** | ~11.39 ms |

---

## 2. Baseline Comparison (Proposal §3.3 Validation)

| Baseline Model | Modality | Exercise Acc (%) | Exercise Macro F1 (%) | Quality Acc (%) | Quality Macro F1 (%) | Combined Score (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 1: Random Forest (16 Clinical Features Only)** | Input B Only (16 Features) | 81.64% | 80.66% | 50.72% | 47.40% | **66.18%** |
| **Baseline 2: Single-IMU 1D-CNN (Thigh 6-Ch Only)** | Single IMU (Thigh 6-Ch Only) | 86.96% | 86.52% | 47.34% | 44.87% | **67.15%** |
| **Hybrid 5: Self-Attention + Kinematics (Winning Dual-IMU Hybrid)** | Dual-IMU (12-Ch) + 16 Features | **89.86%** | **89.73%** | **57.49%** | **54.45%** | **73.67%** |

---

## 3. Edge-AI TensorFlow Lite (`.tflite`) Deployment Validation (Proposal §3.5)

- **Quantized Edge Model File:** `api/models/akira_hybrid_cnn_edge.tflite` (**73.29 KB**)
- **Embedded C Array Export:** `api/models/edge_akira_classifier.c`
- **Mean Edge Inference Latency:** **0.208 ms** (95th Percentile: **0.27 ms**)
- **TFLite vs. Full Keras Agreement:** **100.0%** (Exercise) / **99.03%** (Quality)
