# AKIRA 6 All-Hybrid Multi-Modal Models Benchmark Report

**Component:** Knee Rehabilitation Assessment Following Tibial Plateau ORIF  
**Framework:** TensorFlow 2.22.0-rc0 / Keras Functional API  
**Inputs:** Dual-Input Fusion:
- **Input A:** Raw 12-Channel IMU Sequence ($200$ samples @ $100\text{ Hz}$, Thigh + Shin)
- **Input B:** 16-D Clinical Kinematic Feature Vector (ROM, Depth, Quad Lag, SPARC Smoothness, Cycle Durations)  
**Total Benchmark Dataset:** $709$ Clinical & Reference Trials (KneE-PAD + StrengthSense + PHYTMO)  
**Validation Strategy:** 5-Fold Stratified Cross-Validation  

---

## 1. 6-Model Benchmark Comparison Table

| Rank | Model Architecture | Parameters | Task 1: Exercise Acc | Task 1: Macro F1 | Task 2: Quality Acc | Task 2: Quality F1 | Combined Score |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | **Hybrid 2: BiLSTM** | 103,559 | **98.17%** | 98.81% | **82.22%** | 81.55% | **90.20%** |
| **#2** | **Hybrid 1: 1D-CNN** | 55,879 | **97.60%** | 98.49% | **82.51%** | 82.23% | **90.05%** |
| **#3** | **Hybrid 5: Self-Attention** | 42,055 | **97.18%** | 98.11% | **82.79%** | 82.63% | **89.99%** |
| **#4** | **Hybrid 3: CNN-BiLSTM (AKIRA)** | 97,223 | **97.60%** | 97.69% | **80.39%** | 80.40% | **89.00%** |
| **#5** | **Hybrid 6: Gated Attention** | 43,335 | **98.17%** | 98.81% | **78.14%** | 77.60% | **88.15%** |
| **#6** | **Hybrid 4: 1D-ResNet** | 159,431 | **96.90%** | 98.08% | **78.98%** | 78.38% | **87.94%** |

---

## 2. Winning Architecture: Hybrid 2: BiLSTM

### Task 1: Exercise Classification Report (4 Classes)
```
                  precision    recall  f1-score   support

            Walk     1.0000    1.0000    1.0000        17
Seated Extension     0.9642    0.9926    0.9782       271
      Mini Squat     0.9912    0.9712    0.9811       347
         Step-Up     1.0000    0.9865    0.9932        74

        accuracy                         0.9817       709
       macro avg     0.9888    0.9876    0.9881       709
    weighted avg     0.9820    0.9817    0.9817       709

```

### Task 2: Movement Quality Grading Report (3 Grades)
```
                   precision    recall  f1-score   support

   Good (Correct)     0.8505    0.8616    0.8560       383
Needs Improvement     0.7763    0.8676    0.8194        68
      Bad (Fault)     0.7918    0.7519    0.7714       258

         accuracy                         0.8223       709
        macro avg     0.8062    0.8271    0.8156       709
     weighted avg     0.8220    0.8223    0.8217       709

```

---

## 3. Edge-AI & Mobile Deployment
- **Saved Keras Model:** [`api/models/best_akira_hybrid_model.keras`](../api/models/best_akira_hybrid_model.keras)
- **TensorFlow Lite Model:** [`api/models/akira_hybrid_cnn_edge.tflite`](../api/models/akira_hybrid_cnn_edge.tflite) (0.0 KB)
- **Visual Visualizations:**
  - Comparison Bar Chart: [`research/plots/hybrid_models_comparison.png`](../../research/plots/hybrid_models_comparison.png)
  - Dual-Task Confusion Matrices: [`research/plots/hybrid_confusion_matrices.png`](../../research/plots/hybrid_confusion_matrices.png)
