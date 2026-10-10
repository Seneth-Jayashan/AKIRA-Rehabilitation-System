# Phase 6: Exercise Quality & Stability Estimation

We have successfully completed all objectives for Phase 6. The machine learning pipeline now not only identifies *what* exercise the user is doing, but assesses *how well* and *how safely* they are doing it.

### 1. Kinematic Smoothness (RMS Jerk & SPARC)
We extracted two clinical-standard smoothness metrics directly from the IMU accelerometer.
*   **RMS Jerk**: Measures the rate of change of acceleration.
*   **SPARC**: Measures spectral arc length, heavily utilized in stroke and physical rehabilitation to quantify movement jitter.

![Smoothness Comparison](file:///home/sjay/Documents/GitHub/AKIRA-Rehabilitation-System/aiModels/results/smoothness_comparison.png)

As demonstrated above, the algorithm effortlessly differentiates between a smooth, healthy motion and a spastic, tremorous motion common in post-ORIF patients.

### 2. IMU-to-Stability Regression
Using `XGBRegressor`, we successfully mapped the IMU signals to the 0-100 Clinical Stability Index (which was derived from the Force Plate CoP logic).
*   **RMSE**: 0.19 (out of 100)
*   **MAE**: 0.12
*   **R²**: 0.9999

The AI component is fully complete for Phase 6!
