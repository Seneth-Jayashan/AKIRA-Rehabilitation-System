# Rehabilitation Metric Validation

## 1. ROM Validation
- **Elbow ROM MAE:** 14.56°
- **Elbow ROM Bias:** -5.52° (Positive means over-estimation)
- **Forearm ROM MAE:** 24.44°

## 2. Derivative Validation (Velocity)
- **Elbow Angular Velocity MAE:** 3.40 °/s
- **Forearm Angular Velocity MAE:** 6.02 °/s

## 3. Repetition Counting
- **Mean Absolute Repetition Counting Error:** 0.40 reps/window
- **% Windows within ±1 repetition:** 97.7%

## 4. Research Interpretation
**What the experiment demonstrated:** Subject-independent kinematic estimation and exercise-recognition validation using a healthy public dataset.
**What it suggests:** Edge-AI can viably replace optical trackers for raw joint-angle stream regression.
**What it did NOT demonstrate:** Clinical rehabilitation effectiveness, ORIF recovery prediction, or therapeutic benefit. Clinical cohorts are required for validation of pathological movement signatures.
