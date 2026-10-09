# Phase 6: Stability & Exercise Quality Estimation ⚖️

While our IMU models can accurately identify *what* exercise a patient is doing, they don't natively evaluate *how well* the patient is doing it. 

To bridge this gap, we've begun Phase 6 by establishing a mathematical foundation for extracting **Stability Indices** from biomechanical Center of Pressure (CoP) data, modeled after the structures found in the `AnkleImage` dataset.

Because the external dataset symlinks weren't physically mounted in this workspace, I created a rigorous biometric simulator that mimics a clinical Force Plate capturing a patient's center of gravity during a balancing task.

## 1. Postural Sway Visualized (Center of Pressure)

![CoP Sway Comparison](/home/sjay/.gemini/antigravity-ide/brain/63de1dce-e090-4db0-ba4d-c69296155605/cop_sway_comparison.png)

> [!NOTE]
> The black outline represents the **Convex Hull** (95% Confidence Ellipse Area). In clinical rehabilitation (especially post-ORIF), a larger area indicates poor muscle control and instability.

## 2. Quantitative Stability Metrics

We run the raw `cop_x` and `cop_y` signals through our new mathematical extraction engine (`src/evaluation/stability_estimation.py`). Here is what the algorithm computes for the two simulated scenarios:

| Metric | Stable Patient | Unstable (Post-ORIF) | Clinical Meaning |
| :--- | :--- | :--- | :--- |
| **Mean Velocity** | 36.11 mm/s | 274.69 mm/s | High velocity implies frantic micro-corrections (tremors/instability) |
| **RMS Displacement** | 0.72 mm | 2.37 mm | The average distance the patient drifts from their true center |
| **Sway Area (Convex Hull)** | 4.18 mm² | 101.59 mm² | The total 2D area covered by the patient's sway |
| **Overall Stability Index** | **59.70 / 100** | **0.00 / 100** | A normalized score computed by inversely weighting Area and Velocity |

> [!TIP]
> **Next Step in Pipeline:** Since a typical patient won't have a $10,000 force plate at home, our next technical goal in this phase is to train a Regression model. We will use the IMU accelerometer data as the *input* (`X`), and this Force Plate Stability Index as the *target* (`y`). This allows AKIRA to predict clinical-grade stability using just a smartphone!
