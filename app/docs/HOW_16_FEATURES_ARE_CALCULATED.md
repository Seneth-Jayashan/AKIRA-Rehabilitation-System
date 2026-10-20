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
  Step 2.3: Gyro Bias Calibration & Quiet Reference Sample (`calibrate_sensors`)
                │
                ▼
  Step 2.4 & 2.5: 6-DoF Madgwick Fusion per Sensor, Relative Quaternion in the Thigh Frame,
                  Projection on the SVD Hinge Axis (`madgwick_fusion` + `compute_relative_orientation`)
                │
                ▼
  Step 2.6: Unwrap, Polarity Convention & Zero at the Most-Extended Sample (`_orient_and_zero`)
                │
                ▼
  Continuous Knee Angle θ(t) ≥ 0 [degrees] & Velocity ω(t) = dθ/dt [deg/s]
                │
        ┌───────┴───────────────────────────────────────┐
        ▼                                               ▼
Biomechanical Parameters                      SPARC Smoothness & Adaptive Repetitions
(ROM, Peak Depth, Return Deficit, IQR, RMS)  (Spectral Arc Length, Adaptive Prominence Peaks)
        │                                               │
        └───────┬───────────────────────────────────────┘
                ▼
  Standardized 16-D Clinical Feature Vector (Input B)
```

---

## 2. Preliminary Signal & Sensor Fusion Calculations (Steps 2.1–2.5)

### Step 2.1: Cubic-Spline Temporal Resampling (`resample_to_100hz`)
Heterogeneous sensor sampling rates ($148.15\text{ Hz}$ in KneE-PAD, $100\text{ Hz}$ in PHYTMO, $52\text{ Hz}$ in StrengthSense, $56.35\text{ Hz}$ in HuGaDB, or live BLE notifications) are interpolated onto a uniform $100.0\text{ Hz}$ time grid using `scipy.interpolate.CubicSpline`.

### Step 2.2: Zero-Phase 4th-Order Butterworth Low-Pass Filter (`butter_lowpass_filter`)
A 4th-order zero-phase Butterworth low-pass filter ($f_c = 5.0\text{ Hz}$) removes high-frequency soft-tissue wobble while preserving human knee kinematics ($0.2\text{--}3.5\text{ Hz}$).

### Step 2.3: Gyro Bias Calibration (`calibrate_sensors`)
The quietest low-velocity frames of the window are used to remove residual gyroscope bias (only when the
segment is genuinely near rest, < 15 deg/s) and to choose the reference sample `quiet_idx`. The function
also reports a gravity "tare" angle, but it is **not** added to the knee angle: on real data the quiet sample
is often not full extension, and adding it produced impossible negative angles (down to −122°).

### Steps 2.4 & 2.5: 3D Madgwick Fusion & Relative Orientation (`madgwick_fusion`, `compute_relative_orientation`)
1. **Madgwick filter:** 6-DoF unit quaternions $\mathbf{q}_{\text{thigh}}(t)$ and $\mathbf{q}_{\text{shin}}(t)$, each
   initialised from its first gravity reading.
2. **Relative orientation:** $\mathbf{q}_{\text{rel}}(t) = \mathbf{q}_{\text{thigh}}^{-1}(t) \otimes \mathbf{q}_{\text{shin}}(t)$; the change since the
   reference sample is taken **in the thigh frame**, $\mathbf{q}_{\text{rel}}(t) \otimes \mathbf{q}_{\text{rel}}^{-1}(t_{\text{ref}})$ (taking it in the
   shin frame shrinks or flips the angle whenever the shin sensor is mounted at an angle).
3. **Hinge projection:** the knee hinge axis $\mathbf{u}$ is the first SVD direction of the relative angular velocity
   (shin gyro rotated into the thigh frame minus thigh gyro). The signed angle is
   $\theta_s(t) = 2\,\operatorname{atan2}(\mathbf{q}_{v}(t)\cdot\mathbf{u},\; q_w(t))$.

### Step 2.6: Unwrap, Polarity and Zero (`_orient_and_zero`)
$\theta_s$ is unwrapped (removes ±180° jumps), its sign is chosen so the larger excursion from the median is
positive, and it is shifted so its minimum is $0°$: $\theta(t) = \pm\theta_s(t) - \min(\pm\theta_s)$.
* **Zero = most-extended posture of the recording**, not anatomical full extension. Public datasets have no
  straight-leg calibration pose and unknown sensor mounting, so absolute extension (and true extensor lag)
  cannot be recovered from them.
* **Polarity is a convention.** IMU data alone cannot tell flexion from extension without a known sensor
  orientation. For seated knee extension (resting posture flexed) the positive direction is usually
  extension, and for near-symmetric windows it can flip. ROM, angular-speed magnitude, SPARC and repetition
  timing do not depend on polarity.

### Optical Motion Capture Validation (`run_optitrack_mocap_validation`)
Compared with synchronised OptiTrack rigid bodies on PHYTMO (thigh-frame relative rotation projected on its own
hinge axis, same polarity rule applied independently, no rescaling, no high-pass filter, no trial excluded for
disagreeing with the IMU). Results (`app/api/models/optitrack_validation_metrics.json`: 97 trials evaluated,
93 of which the system itself accepts, ROM 4–155°):
* **Angle-trace error ignoring polarity:** mean MAE **5.84°**, median **3.43°**
* **As output (including polarity):** mean MAE **12.45°**, median **4.17°**
* **Polarity agreement with OptiTrack:** **79.4%** of trials
* Exclusions are reported, not hidden: 34 optical marker drop-outs, 7 failed synchronisations, 5 files without
  thigh/shin rigid bodies.

---

## 3. Step-by-Step Breakdown of the 16 Clinical Features

$\theta(t) = 0^\circ$ is the most-extended posture of the recording and $\theta(t) > 0^\circ$ is excursion away from it
(flexion for walking, squats and stairs; usually extension excursion for seated knee extension, see Step 2.6):
* Movement away from the zero posture gives $\omega(t) = \frac{d\theta}{dt} > 0$.
* Movement back toward it gives $\omega(t) = \frac{d\theta}{dt} < 0$.

---

### Feature 1: Range of Motion (`rom_deg`)
* **Mathematical Formula:**
  $$\text{ROM} = \max_{t} \theta(t) - \min_{t} \theta(t) = \max_t \theta(t)$$
* **Unit:** Degrees ($^\circ$)
* **How It Is Calculated:** Total excursion of the knee angle in the window (independent of the polarity convention).
* **Clinical Meaning:** Active functional excursion of the knee joint.

---

### Feature 2: Peak Depth (`peak_depth_deg`)
* **Mathematical Formula:**
  $$\text{Depth} = \begin{cases} \frac{1}{K}\sum_k \theta(t_{\text{peak},k}) & K \ge 2 \\ \max_t \theta(t) & \text{otherwise} \end{cases}$$
* **Unit:** Degrees ($^\circ$), measured from the most-extended posture of the recording
* **How It Is Calculated:** Mean repetition peak when at least two repetitions are segmented, otherwise the maximum.
* **Clinical Meaning:** Typical excursion depth per repetition (compared with the clinician's safe-flexion limit).

---

### Feature 3: Extension Return Deficit (`extension_return_deficit_deg`)
* **Mathematical Formula:**
  $$\text{Deficit} = \frac{1}{2K}\sum_{k=1}^{K}\big(\theta(t_{\text{start},k}) + \theta(t_{\text{end},k})\big) - \min_t \theta(t) \quad (0 \text{ if } K = 0)$$
* **Unit:** Degrees ($^\circ$)
* **How It Is Calculated:** Mean trough angle at the start and end of each segmented repetition above the
  most-extended posture of the recording.
* **Clinical Meaning:** How completely the patient returns to extension between repetitions. It is **not** an
  absolute extensor lag (that needs a straight-leg calibration pose); the clinician's lag threshold is applied
  to this measure and labelled accordingly in the app. (Older code stored this slot as `min_angle_deg`; the
  feature dictionary keeps `extensor_lag_deg` as a backward-compatible alias.)

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
| **2** | Peak Depth | `peak_depth_deg` | mean rep peak, or $\max(\theta)$ | $^\circ$ | $30^\circ \text{--} 125^\circ$ |
| **3** | Extension Return Deficit | `extension_return_deficit_deg` | mean rep trough $- \min(\theta)$ | $^\circ$ | $0^\circ \text{--} 20^\circ$ |
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
