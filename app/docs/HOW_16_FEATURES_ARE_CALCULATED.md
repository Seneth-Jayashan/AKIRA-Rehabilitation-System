# Mathematical & Clinical Specification: How the 16 Kinematic Features Are Calculated

**Project:** Smart Post-ORIF Rehabilitation Ecosystem with Edge-AI Kinematic Tracking (AKIRA)  
**Target Joint:** Knee Joint (Tibial Plateau ORIF Rehabilitation)  
**Sensor Configuration:** 2 Synchronized 6-Axis IMUs (Anterior Mid-Thigh + Anterior Upper Shin) @ $100\text{ Hz}$ ($\Delta t = 0.01\text{ s}$)  
**Implementation Source:** [`api/math_feature_extractor.py`](../api/math_feature_extractor.py)  

---

## 1. Overview of the 5-Step Sensor Fusion & Feature Extraction Flow

Before the 16 clinical features are extracted, raw 12-channel dual-IMU streams (`[Thigh Acc X/Y/Z, Thigh Gyro X/Y/Z, Shin Acc X/Y/Z, Shin Gyro X/Y/Z]`) pass through the 5-stage kinematic signal processing pipeline in `Clinical16FeatureExtractor`:

```
[ Raw 12-Channel Dual-IMU Stream ]
                │
                ▼
  Step 2.1: Cubic-Spline Resampling (`resample_to_100hz`)
            (Normalizes 50 Hz / 148 Hz / variable BLE rates → 100.0 Hz)
                │
                ▼
  Step 2.2: 4th-Order Zero-Phase Butterworth Low-Pass Filter (`butter_lowpass_filter`, fc = 5.0 Hz)
            (Suppresses high-frequency soft-tissue artifact & vibration noise)
                │
                ▼
  Step 2.3: Static Sensor-to-Segment Tare & Gyro Bias Calibration (`calibrate_sensors`)
            (Aligns gravitational reference & preserves anatomical extension offset)
                │
                ▼
  Step 2.4 & 2.5: 3D Madgwick / Complementary Sensor Fusion & SVD Hinge Axis Projection
                  (`madgwick_fusion` + `compute_relative_orientation`)
            (Fuses 3D accelerometer tilt with 3D gyroscope rate along the principal knee hinge axis;
             drift-free on slow isometric holds without 0.05 Hz high-pass distortion)
                │
                ▼
  Continuous Anatomical Knee Flexion Angle: θ(t) [degrees] & Velocity: ω(t) = dθ/dt [deg/s]
                │
        ┌───────┴───────────────────────────────────────┐
        ▼                                               ▼
Biomechanical Parameters                      SPARC Smoothness & Adaptive Repetitions
(ROM, Peak Depth, Extensor Lag, IQR, RMS)     (Spectral Arc Length, Adaptive Prominence Peaks)
        │                                               │
        └───────┬───────────────────────────────────────┘
                ▼
  Standardized 16-D Clinical Feature Vector (Input B)
```

---

## 2. Preliminary Signal & Sensor Fusion Calculations (Steps 2.1–2.5)

### Step 2.1: Cubic-Spline Temporal Resampling (`resample_to_100hz`)
Heterogeneous sensor sampling rates ($50\text{ Hz}$ in KneE-PAD, $100\text{ Hz}$ in PHYTMO, $56.3\text{ Hz}$ in HuGaDB, or live BLE notifications) are interpolated onto a uniform $100.0\text{ Hz}$ time grid using `scipy.interpolate.CubicSpline`.

### Step 2.2: Zero-Phase 4th-Order Butterworth Low-Pass Filter (`butter_lowpass_filter`)
A 4th-order zero-phase Butterworth low-pass filter ($f_c = 5.0\text{ Hz}$) removes high-frequency soft-tissue wobble while preserving human knee kinematics ($0.2\text{--}3.5\text{ Hz}$).

### Step 2.3: Sensor-to-Segment Static Tare & Gyro Bias Calibration (`calibrate_sensors`)
Quiet low-velocity frames within the window are identified to remove residual gyroscope bias while computing the true anatomical resting extension angle $\theta_{\text{lag,ref}} \ge 0^\circ$ from the relative 3D gravity vectors of the thigh and shin accelerometers.

### Steps 2.4 & 2.5: 3D Sensor Fusion & Principal Hinge Orientation (`madgwick_fusion`, `compute_relative_orientation`)
Instead of relying on single-axis gyroscope integration with a $0.05\text{ Hz}$ high-pass filter (which drifts and distorts slow isometric holds), AKIRA fuses:
1. **3D Relative Gyroscope Projection ($\omega_{\text{hinge}}(t)$):** Using Singular Value Decomposition (SVD) on the differential angular velocity $\boldsymbol{\omega}_{\text{rel}}(t) = \boldsymbol{\omega}_{\text{shin}}(t) - \boldsymbol{\omega}_{\text{thigh}}(t)$, the principal anatomical knee flexion-extension hinge axis $\mathbf{u}_{\text{hinge}} \in \mathbb{R}^3$ is identified, making the angle invariant to strap mounting rotation.
2. **3D Accelerometer Gravity Reference ($\theta_{\text{acc}}(t)$) & Madgwick Quaternion Orientation:** Combines low-frequency 3D accelerometer tilt with high-frequency gyroscope integration via complementary/Madgwick fusion ($\alpha = 0.96$), anchored to the calibrated anatomical extension baseline so **Feature 1 (ROM)**, **Feature 2 (Peak Depth)**, and **Feature 3 (Extensor Lag)** remain distinct and physically accurate.

### Optical Motion Capture Validation (`run_optitrack_mocap_validation`)
Validated against synchronized **OptiTrack 3D Optical Motion Capture** ground truth on PHYTMO trial `A01KFER0_1` (`app/api/models/optitrack_validation_metrics.json`):
* **Mean Absolute Error (MAE):** $0.91^\circ$
* **Root Mean Square Error (RMSE):** $1.12^\circ$
* **Pearson Correlation ($r$):** $0.9988$

---

## 3. Step-by-Step Breakdown of the 16 Clinical Features

Since $\theta(t) = 0^\circ$ represents full anatomical knee extension and $\theta(t) > 0^\circ$ represents knee flexion:
* **Flexion (bending the knee)** produces a positive angular velocity: $\omega(t) = \frac{d\theta}{dt} > 0$.
* **Extension (straightening the knee)** produces a negative angular velocity: $\omega(t) = \frac{d\theta}{dt} < 0$.

---

### Feature 1: Range of Motion (`rom_deg`)
* **Mathematical Formula:**
  $$\text{ROM} = \max_{t} \theta(t) - \min_{t} \theta(t)$$
* **Unit:** Degrees ($^\circ$)
* **How It Is Calculated:** Difference between maximum flexion angle and minimum extension angle achieved during the window.
* **Clinical Meaning:** Total active functional excursion of the knee joint, distinct from peak flexion depth whenever an extension deficit ($\theta_{\min} > 0^\circ$) is present.

---

### Feature 2: Peak Flexion Depth (`peak_depth_deg`)
* **Mathematical Formula:**
  $$\text{Depth} = \max_{t} \theta(t)$$
* **Unit:** Degrees ($^\circ$)
* **How It Is Calculated:** Maximum knee flexion angle relative to anatomical zero ($0^\circ$), satisfying $\text{Depth} = \text{ROM} + \theta_{\min}$.
* **Clinical Meaning:** Verifies weight-bearing flexion safety boundaries (e.g., maintaining Mini Squat depth within the $30^\circ\text{--}45^\circ$ safe zone for early/mid-stage Tibial Plateau ORIF).

---

### Feature 3: Terminal Extension Deficit / Quadriceps Lag (`extensor_lag_deg` / `min_angle_deg`)
* **Mathematical Formula:**
  $$\theta_{\text{lag}} = \max\left(0^\circ, \min_{t} \theta(t)\right)$$
* **Unit:** Degrees ($^\circ$)
* **How It Is Calculated:** Minimum angle reached during terminal extension relative to the $0^\circ$ full-extension reference.
* **Clinical Meaning:** Directly quantifies **Quadriceps Extensor Lag** (inability to reach $0^\circ$ full anatomical extension due to arthrogenic muscle inhibition or capsular stiffness). Across the 1,036-trial clinical dataset, $\theta_{\text{lag}}$ has a mean of $5.42^\circ \pm 5.55^\circ$.

---

### Feature 4: Peak Angular Velocity (`peak_angular_velocity`)
* **Mathematical Formula:**
  $$\omega_{\text{peak}} = \max_{t} |\omega(t)|$$
* **Unit:** Degrees per second ($\text{deg/s}$)
* **Clinical Meaning:** Reflects peak rotational speed across the movement cycle; reduced peak velocity indicates pain guarding or neuromuscular hesitation.

---

### Feature 5: Mean Angular Velocity (`mean_angular_velocity`)
* **Mathematical Formula:**
  $$\omega_{\text{mean}} = \frac{1}{N} \sum_{i=1}^{N} |\omega(t_i)|$$
* **Unit:** Degrees per second ($\text{deg/s}$)
* **Clinical Meaning:** Measures overall movement cadence and pacing consistency.

---

### Feature 6: SPARC Movement Smoothness (`sparc_smoothness`)
* **Mathematical Formula:**
  $$\text{SPARC} = -\int_{0}^{\omega_c} \sqrt{\left(\frac{1}{\omega_c}\right)^2 + \left(\frac{d\hat{V}(\omega)}{d\omega}\right)^2} d\omega$$
* **Unit:** Dimensionless ($\le 0$)
* **Clinical Meaning:** Spectral Arc Length of the normalized velocity Fourier spectrum $\hat{V}(\omega)$. Smooth coordinated trajectories yield values near $-1.8 \text{ to } -3.0$, whereas jerky post-ORIF tremor or compensatory sub-movements yield more negative values ($< -4.5$).

---

### Feature 7: Repetition Count (`repetition_count`)
* **Mathematical Formula:**
  $$N_{\text{reps}} = \text{AdaptivePeakCount}\Big(\theta(t), \text{prominence} = \max(3.5^\circ, \min(18^\circ, 0.30 \cdot \text{ROM})), \Delta t_{\min} = 0.45\text{ s}\Big)$$
* **Unit:** Integer count
* **How It Is Calculated:** Adaptive prominence peak segmentation (`segment_repetitions`) detects both large-ROM movements (Seated Extension, Step-Up, Walking) and small-ROM exercises (Quadriceps Sets, Straight-Leg Raise, Mini Squat). Returns $0$ when the limb is stationary ($\text{ROM} < 3.0^\circ$).

---

### Feature 8: Mean Repetition Duration (`mean_repetition_duration`)
* **Mathematical Formula:**
  $$\bar{T}_{\text{rep}} = \begin{cases} \frac{1}{K} \sum_{k=1}^{K} (t_{\text{end}, k} - t_{\text{start}, k}) & \text{if } K \ge 1 \\ T_{\text{active}}(|\omega(t)| > 0.15\,\omega_{\text{peak}}) & \text{if } K = 0 \text{ and } \text{ROM} \ge 3.0^\circ \\ 0.0\text{ s} & \text{if stationary } (\text{ROM} < 3.0^\circ) \end{cases}$$
* **Unit:** Seconds ($\text{s}$)
* **How It Is Calculated:** Mean trough-to-trough duration of segmented repetitions, falling back to the active velocity envelope duration if a single slow movement spans the window, or $0.0\text{ s}$ when stationary (never defaulting to the arbitrary window length).

---

### Feature 9: Knee Angle Standard Deviation (`angle_std`)
* **Mathematical Formula:**
  $$\sigma_{\theta} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left(\theta(t_i) - \bar{\theta}\right)^2}$$
* **Unit:** Degrees ($^\circ$)

---

### Feature 10: Angular Velocity Standard Deviation (`velocity_std`)
* **Mathematical Formula:**
  $$\sigma_{\omega} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left(\omega(t_i) - \bar{\omega}\right)^2}$$
* **Unit:** Degrees per second ($\text{deg/s}$)

---

### Feature 11: Angle Interquartile Range (`angle_iqr`)
* **Mathematical Formula:**
  $$\text{IQR}_{\theta} = Q_{75}(\theta) - Q_{25}(\theta)$$
* **Unit:** Degrees ($^\circ$)

---

### Feature 12: Peak Flexion Angular Velocity (`max_ang_vel`)
* **Mathematical Formula:**
  $$\omega_{\text{flex,max}} = \max_{t} \omega(t) \quad (> 0)$$
* **Unit:** Degrees per second ($\text{deg/s}$)
* **How It Is Calculated:** Maximum positive angular velocity ($d\theta/dt > 0$) as the knee bends toward peak flexion.
* **Clinical Meaning:** Quantifies concentric/eccentric flexion phase speed (e.g., descent speed during Mini Squat or upward step flexion during Step-Up).

---

### Feature 13: Peak Extension Angular Velocity (`min_ang_vel`)
* **Mathematical Formula:**
  $$\omega_{\text{ext,min}} = \min_{t} \omega(t) \quad (< 0)$$
* **Unit:** Degrees per second ($\text{deg/s}$)
* **How It Is Calculated:** Most negative angular velocity ($d\theta/dt < 0$) as the knee straightens toward $0^\circ$ extension.
* **Clinical Meaning:** Quantifies active quadriceps extension speed during straightening (e.g., returning to full extension in Seated Knee Extension or ascending in Mini Squat).

---

### Feature 14: Mean Joint Angle (`mean_angle`)
* **Mathematical Formula:**
  $$\bar{\theta} = \frac{1}{N} \sum_{i=1}^{N} \theta(t_i)$$
* **Unit:** Degrees ($^\circ$)

---

### Feature 15: Angle Root Mean Square (`angle_rms`)
* **Mathematical Formula:**
  $$\text{RMS}_{\theta} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \theta(t_i)^2}$$
* **Unit:** Degrees ($^\circ$)

---

### Feature 16: Normalized Smoothness Rate (`sparc_per_duration`)
* **Mathematical Formula:**
  $$\text{Eff} = \begin{cases} \frac{\text{SPARC}}{\bar{T}_{\text{rep}}} & \text{if } \bar{T}_{\text{rep}} > 0.1\text{ s} \\ \text{SPARC} & \text{otherwise} \end{cases}$$
* **Unit:** Inverse seconds ($\text{s}^{-1}$)

---

## 4. Quick Summary Reference Table

| # | Feature Name | Variable | Mathematical Formula | Unit | Typical Clinical Range |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **1** | Range of Motion | `rom_deg` | $\max(\theta) - \min(\theta)$ | $^\circ$ | $30^\circ \text{--} 125^\circ$ |
| **2** | Peak Flexion Depth | `peak_depth_deg` | $\max(\theta)$ | $^\circ$ | $35^\circ \text{--} 130^\circ$ |
| **3** | Terminal Extension Lag | `min_angle_deg` | $\max(0, \min(\theta))$ | $^\circ$ | $0^\circ \text{--} 18^\circ$ ($0^\circ$ = full lock) |
| **4** | Peak Angular Velocity | `peak_angular_velocity` | $\max \|\omega\|$ | $\text{deg/s}$ | $60 \text{--} 250\text{ deg/s}$ |
| **5** | Mean Angular Velocity | `mean_angular_velocity` | $\text{mean} \|\omega\|$ | $\text{deg/s}$ | $15 \text{--} 75\text{ deg/s}$ |
| **6** | SPARC Smoothness | `sparc_smoothness` | $-\int \sqrt{(1/\omega_c)^2 + (d\hat{V}/d\omega)^2} d\omega$ | - | $-1.8 \text{ to } -3.0$ (smooth) / $< -4.5$ (jerky) |
| **7** | Repetition Count | `repetition_count` | Adaptive prominence peak count | reps | $0 \text{--} 4\text{ reps}$ (per 2s window) |
| **8** | Repetition Duration | `mean_repetition_duration`| Mean active cycle duration | $\text{s}$ | $0.6 \text{--} 2.0\text{ s}$ ($0\text{ s}$ if still) |
| **9** | Angle Std Dev | `angle_std` | $\text{std}(\theta)$ | $^\circ$ | $10^\circ \text{--} 40^\circ$ |
| **10** | Velocity Std Dev | `velocity_std` | $\text{std}(\omega)$ | $\text{deg/s}$ | $20 \text{--} 90\text{ deg/s}$ |
| **11** | Angle IQR | `angle_iqr` | $Q_{75}(\theta) - Q_{25}(\theta)$ | $^\circ$ | $15^\circ \text{--} 70^\circ$ |
| **12** | Peak Flexion Velocity | `max_ang_vel` | $\max(\omega) > 0$ (bending) | $\text{deg/s}$ | $+40 \text{ to } +220\text{ deg/s}$ |
| **13** | Peak Extension Velocity | `min_ang_vel` | $\min(\omega) < 0$ (straightening) | $\text{deg/s}$ | $-40 \text{ to } -220\text{ deg/s}$ |
| **14** | Mean Joint Angle | `mean_angle` | $\text{mean}(\theta)$ | $^\circ$ | $15^\circ \text{--} 70^\circ$ |
| **15** | Angle RMS | `angle_rms` | $\sqrt{\text{mean}(\theta^2)}$ | $^\circ$ | $20^\circ \text{--} 80^\circ$ |
| **16** | Smoothness Rate | `sparc_per_duration` | $\text{SPARC} / \bar{T}_{\text{rep}}$ | $\text{s}^{-1}$ | $-1.0 \text{ to } -4.5\text{ s}^{-1}$ |
