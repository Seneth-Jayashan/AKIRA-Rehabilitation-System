# Mathematical & Clinical Specification: How the 16 Kinematic Features Are Calculated

**Project:** Smart Post-ORIF Rehabilitation Ecosystem with Edge-AI Kinematic Tracking (AKIRA)  
**Target Joint:** Knee Joint (Tibial Plateau ORIF Rehabilitation)  
**Sensor Configuration:** 2 Synchronized 6-Axis IMUs (Anterior Mid-Thigh + Anterior Upper Shin) @ $100\text{ Hz}$ ($dt = 0.01\text{ s}$)  
**Implementation Source:** [`api/math_feature_extractor.py`](../api/math_feature_extractor.py)  

---

## 1. Overview of the Calculation Flow

Before the 16 features can be extracted, the raw sensor streams must be converted into the continuous anatomical **Knee Angle $\theta(t)$** and **Angular Velocity $\omega(t)$**:

```
[ Thigh Gyro Y(t) ] ──┐
                      ├─► Relative Angular Velocity: ω(t) = Shin_Gyro_Y(t) - Thigh_Gyro_Y(t)
[ Shin Gyro Y(t)  ] ──┘                                      │
                                                             ▼
                                             Numerical Integration (dt = 0.01 s)
                                             + 1st-Order High-Pass Filter (0.05 Hz)
                                                             │
                                                             ▼
                                             Continuous Knee Angle: θ(t) (degrees)
                                                             │
                                                             ▼
                                     Instantaneous Angular Velocity: ω(t) = dθ/dt
                                                             │
                                     ┌───────────────────────┴───────────────────────┐
                                     ▼                                               ▼
                         Biomechanical Features                          SPARC Smoothness & Reps
                         (ROM, Depth, Lag, IQR, RMS)                     (FFT Arc Length, Peak Detection)
                                     │                                               │
                                     └───────────────────────┬───────────────────────┘
                                                             ▼
                                               Standardized 16-D Feature Vector
```

---

## 2. Preliminary Signal Calculations

### A. Relative Knee Angular Velocity $\omega_{\text{joint}}(t)$
The lateral Y-axis of each IMU aligns with the knee joint flexion/extension axis in the anatomical sagittal plane:
$$\omega_{\text{joint}}(t) = \text{Gyro}_{Y,\text{shin}}(t) - \text{Gyro}_{Y,\text{thigh}}(t) \quad [\text{deg/s}]$$

### B. Continuous Knee Joint Angle $\theta(t)$
Numerical integration yields raw angular displacement, followed by zero-phase high-pass filtering ($f_c = 0.05\text{ Hz}$) to remove gyroscope bias drift without distorting exercise repetitions:
$$\theta_{\text{raw}}(t) = \sum_{k=0}^{t} \omega_{\text{joint}}(k) \cdot \Delta t, \quad \text{where } \Delta t = 0.01\text{ s}$$
$$\theta(t) = \text{filtfilt}\left(\theta_{\text{raw}}(t)\right) - \min\left(\text{filtfilt}(\theta_{\text{raw}})\right) \quad [\text{degrees}]$$
*Calibrated so anatomical straight extension is $0^\circ$ and flexion is positive ($> 0^\circ$).*

### C. Smoothed Angular Velocity $\omega(t)$
$$\omega(t) = \frac{d\theta}{dt} \approx \frac{\theta(t+1) - \theta(t-1)}{2 \cdot \Delta t} \quad [\text{deg/s}]$$
*(Smoothed with a 2nd-order zero-phase Butterworth filter at $5\text{ Hz}$).*

---

## 3. Step-by-Step Breakdown of the 16 Features

---

### Feature 1: Range of Motion (`rom_deg`)
* **Mathematical Formula:**
  $$\text{ROM} = \max_{t} \theta(t) - \min_{t} \theta(t)$$
* **Unit:** Degrees ($^\circ$)
* **How It Is Calculated:** Finds the absolute highest angle minus the lowest angle in the entire trial.
* **Clinical Meaning:** The total active functional arc of the knee joint. In post-ORIF recovery, tracking ROM recovery from $0^\circ \text{--} 60^\circ$ (early phase) to $0^\circ \text{--} 120^\circ$ (functional restoration) is the primary indicator of joint capsule elasticity and arthrofibrosis prevention.

---

### Feature 2: Peak Movement Depth (`peak_depth_deg`)
* **Mathematical Formula:**
  $$\text{Depth} = \max_{t} \theta(t)$$
* **Unit:** Degrees ($^\circ$)
* **How It Is Calculated:** The maximum flexion angle reached during the trial.
* **Clinical Meaning:** Verifies weight-bearing safety boundaries during closed-chain exercises (e.g. mini-squats). For tibial plateau fractures, surgeons restrict knee flexion to $\le 45^\circ$ or $\le 60^\circ$ in early weeks to prevent catastrophic compressive plate loading.

---

### Feature 3: Terminal Extension Deficit / Min Angle (`min_angle_deg`)
* **Mathematical Formula:**
  $$\theta_{\min} = \min_{t} \theta(t)$$
* **Unit:** Degrees ($^\circ$)
* **How It Is Calculated:** The lowest angle reached during knee extension.
* **Clinical Meaning:** **Directly quantifies Quadriceps Extensor Lag.** A healthy knee reaches $0^\circ$ full anatomical extension lock. If $\theta_{\min} = 10^\circ$, the patient has a $10^\circ$ extension lag due to quadriceps inhibition or posterior capsule adhesion—a major post-operative complication requiring targeted physical therapy.

---

### Feature 4: Peak Angular Velocity (`peak_angular_velocity`)
* **Mathematical Formula:**
  $$\omega_{\text{peak}} = \max_{t} |\omega(t)|$$
* **Unit:** Degrees per second ($\text{deg/s}$)
* **How It Is Calculated:** Takes the maximum absolute value of the instantaneous angular velocity.
* **Clinical Meaning:** Reflects peak neuromuscular burst capability. Slow peak velocities indicate patient fear of pain, weakness, or joint guarding.

---

### Feature 5: Mean Angular Velocity (`mean_angular_velocity`)
* **Mathematical Formula:**
  $$\omega_{\text{mean}} = \frac{1}{N} \sum_{i=1}^{N} |\omega(t_i)|$$
* **Unit:** Degrees per second ($\text{deg/s}$)
* **How It Is Calculated:** The average absolute speed across all samples in the session.
* **Clinical Meaning:** Measures overall movement cadence and pacing control.

---

### Feature 6: SPARC Movement Smoothness (`sparc_smoothness`)
* **Mathematical Formula:**
  $$\text{SPARC} = -\int_{0}^{\omega_c} \sqrt{\left(\frac{1}{\omega_c}\right)^2 + \left(\frac{d\hat{V}(\omega)}{d\omega}\right)^2} d\omega$$
  Where:
  * $\hat{V}(\omega) = \frac{|V(\omega)|}{|V(0)|}$ is the normalized magnitude spectrum of the angular velocity obtained via Fast Fourier Transform (FFT).
  * $\omega_c$ is the cutoff frequency where the spectrum drops below $5\%$ of peak amplitude (capped at $10\text{ Hz}$).
* **Unit:** Dimensionless (always negative, $\le 0$)
* **How It Is Calculated:** Computes the arc length of the normalized frequency spectrum curve of the velocity signal.
* **Clinical Meaning:** **The clinical gold standard for motor control smoothness.** 
  * Healthy, smooth movement has a single bell-shaped velocity curve $\to$ short spectral arc length ($\text{SPARC} \approx -2.0 \text{ to } -2.5$).
  * Post-ORIF guarding, tremor, and hesitation produce jagged, multiple sub-movements $\to$ long spectral arc length ($\text{SPARC} < -4.5$).

---

### Feature 7: Repetition Count (`repetition_count`)
* **Mathematical Formula:**
  $$N_{\text{reps}} = \sum \text{peaks}\Big(\theta(t) \ge 20^\circ, \text{distance} \ge 0.8\text{ s}\Big)$$
* **Unit:** Integer count
* **How It Is Calculated:** Identifies valid repetition cycles using `scipy.signal.find_peaks` requiring at least $20^\circ$ amplitude and $0.8\text{ s}$ temporal separation.
* **Clinical Meaning:** Confirms patient adherence to prescribed exercise dosage without requiring self-reporting.

---

### Feature 8: Mean Repetition Duration (`mean_repetition_duration`)
* **Mathematical Formula:**
  $$\bar{T}_{\text{rep}} = \frac{1}{K} \sum_{k=1}^{K} \left(t_{\text{end}, k} - t_{\text{start}, k}\right)$$
* **Unit:** Seconds ($\text{s}$)
* **How It Is Calculated:** Average elapsed time between cycle onset and cycle completion.
* **Clinical Meaning:** Quantifies cycle pacing. Unusually long cycle durations ($> 3.5\text{ s}$) indicate patient fatigue, hesitation, or pain midway through a repetition.

---

### Feature 9: Knee Angle Standard Deviation (`angle_std`)
* **Mathematical Formula:**
  $$\sigma_{\theta} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left(\theta(t_i) - \bar{\theta}\right)^2}$$
* **Unit:** Degrees ($^\circ$)
* **How It Is Calculated:** Standard deviation of the entire angle series.
* **Clinical Meaning:** Captures total dynamic excursion spread across the active session.

---

### Feature 10: Angular Velocity Standard Deviation (`velocity_std`)
* **Mathematical Formula:**
  $$\sigma_{\omega} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \left(\omega(t_i) - \bar{\omega}\right)^2}$$
* **Unit:** Degrees per second ($\text{deg/s}$)
* **How It Is Calculated:** Standard deviation of instantaneous joint speed.
* **Clinical Meaning:** Measures acceleration-deceleration dynamics. Smooth controlled sets display predictable, bounded velocity spread.

---

### Feature 11: Angle Interquartile Range (`angle_iqr`)
* **Mathematical Formula:**
  $$\text{IQR}_{\theta} = Q_3(\theta) - Q_1(\theta)$$
* **Unit:** Degrees ($^\circ$)
* **How It Is Calculated:** The difference between the 75th percentile and 25th percentile of the angle values.
* **Clinical Meaning:** Robust measure of mid-range joint excursion. Unlike ROM, IQR is unaffected by transient jerks or sensor mounting bumps.

---

### Feature 12: Peak Extension Power / Max Velocity (`max_ang_vel`)
* **Mathematical Formula:**
  $$\omega_{\max} = \max_{t} \omega(t)$$
* **Unit:** Degrees per second ($\text{deg/s}$)
* **How It Is Calculated:** Maximum positive rate of angular change.
* **Clinical Meaning:** Represents explosive quadriceps concentric power during the extension phase (kicking leg straight).

---

### Feature 13: Controlled Flexion Descent Speed (`min_ang_vel`)
* **Mathematical Formula:**
  $$\omega_{\min} = \min_{t} \omega(t)$$
* **Unit:** Degrees per second ($\text{deg/s}$)
* **How It Is Calculated:** Maximum negative rate of angular change.
* **Clinical Meaning:** Measures eccentric quadriceps control during descent into flexion. High negative speed indicates uncontrolled falling into flexion; controlled moderate speed indicates healthy eccentric braking.

---

### Feature 14: Mean Joint Angle (`mean_angle`)
* **Mathematical Formula:**
  $$\bar{\theta} = \frac{1}{T} \int_{0}^{T} \theta(t) dt \approx \frac{1}{N} \sum_{i=1}^{N} \theta(t_i)$$
* **Unit:** Degrees ($^\circ$)
* **How It Is Calculated:** Time-averaged knee flexion angle across the trial.
* **Clinical Meaning:** Indicates whether the exercise was performed predominantly in extension (low mean angle, e.g. Quad sets) or deep flexion (high mean angle, e.g. squats).

---

### Feature 15: Angle Root Mean Square (`angle_rms`)
* **Mathematical Formula:**
  $$\text{RMS}_{\theta} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} \theta(t_i)^2}$$
* **Unit:** Degrees ($^\circ$)
* **How It Is Calculated:** Quadratic mean of the angle series.
* **Clinical Meaning:** Represents cumulative joint mechanical work and displacement energy.

---

### Feature 16: Normalized Smoothness Efficiency (`sparc_per_duration`)
* **Mathematical Formula:**
  $$\text{Eff} = \frac{\text{SPARC}}{\bar{T}_{\text{rep}} + \epsilon}$$
* **Unit:** Inverse seconds ($\text{s}^{-1}$)
* **How It Is Calculated:** The SPARC smoothness score divided by the average repetition duration.
* **Clinical Meaning:** Normalizes movement coordination against speed. Distinguishes a patient who moves smoothly at a normal pace from one who appears smooth simply because they moved extremely slowly.

---

## 4. Quick Summary Reference Table

| # | Feature Name | Variable | Mathematical Formula | Unit | Typical Clinical Range |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **1** | Range of Motion | `rom_deg` | $\max(\theta) - \min(\theta)$ | $^\circ$ | $45^\circ \text{--} 130^\circ$ |
| **2** | Peak Movement Depth | `peak_depth_deg` | $\max(\theta)$ | $^\circ$ | $50^\circ \text{--} 130^\circ$ |
| **3** | Terminal Extension Lag | `min_angle_deg` | $\min(\theta)$ | $^\circ$ | $0^\circ \text{--} 15^\circ$ ($0^\circ$ = ideal lock) |
| **4** | Peak Angular Velocity | `peak_angular_velocity` | $\max \|\omega\|$ | $\text{deg/s}$ | $80 \text{--} 250\text{ deg/s}$ |
| **5** | Mean Angular Velocity | `mean_angular_velocity` | $\text{mean} \|\omega\|$ | $\text{deg/s}$ | $15 \text{--} 60\text{ deg/s}$ |
| **6** | SPARC Smoothness | `sparc_smoothness` | $-\int \sqrt{(1/\omega_c)^2 + (d\hat{V}/d\omega)^2} d\omega$ | - | $-2.0 \text{ to } -2.8$ (smooth) / $< -4.0$ (jerky) |
| **7** | Repetition Count | `repetition_count` | $\sum \text{detected peaks}$ | count | $2 \text{--} 15\text{ reps}$ |
| **8** | Repetition Duration | `mean_repetition_duration`| $\frac{1}{K}\sum \Delta t_{\text{rep}}$ | $\text{s}$ | $1.2 \text{--} 2.8\text{ s}$ |
| **9** | Angle Std Dev | `angle_std` | $\text{std}(\theta)$ | $^\circ$ | $15^\circ \text{--} 45^\circ$ |
| **10** | Velocity Std Dev | `velocity_std` | $\text{std}(\omega)$ | $\text{deg/s}$ | $20 \text{--} 90\text{ deg/s}$ |
| **11** | Angle IQR | `angle_iqr` | $Q_3(\theta) - Q_1(\theta)$ | $^\circ$ | $25^\circ \text{--} 75^\circ$ |
| **12** | Max Extension Speed | `max_ang_vel` | $\max(\omega)$ | $\text{deg/s}$ | $+60 \text{ to } +220\text{ deg/s}$ |
| **13** | Min Flexion Descent | `min_ang_vel` | $\min(\omega)$ | $\text{deg/s}$ | $-60 \text{ to } -220\text{ deg/s}$ |
| **14** | Mean Joint Angle | `mean_angle` | $\text{mean}(\theta)$ | $^\circ$ | $20^\circ \text{--} 70^\circ$ |
| **15** | Angle RMS | `angle_rms` | $\sqrt{\text{mean}(\theta^2)}$ | $^\circ$ | $30^\circ \text{--} 80^\circ$ |
| **16** | SPARC per Duration | `sparc_per_duration` | $\text{SPARC} / \bar{T}_{\text{rep}}$ | $\text{s}^{-1}$ | $-1.0 \text{ to } -3.0\text{ s}^{-1}$ |
