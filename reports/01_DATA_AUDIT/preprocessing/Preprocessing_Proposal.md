# Preprocessing Assessment & Raw Data Structure Report

Based on a recursive inspection of the raw `DOT` CSV files and the MATLAB ground truth structures (e.g., `H_Joint_Angles_Aligned_FINAL`) for Subject 1, here is the detailed breakdown.

## A. Raw Data Structure Report

### 1. IMU Files (RUA & RLA)
- **Filename Example:** `3RUA_0A8BB2DFBE36_20230110_154846.csv` (Right Upper Arm) / `4RLA...` (Right Lower Arm)
- **File Type:** `.csv`
- **File Size:** ~150 KB to ~800 KB
- **Number of Samples:** ~600 to ~8000 per trial
- **Number of Channels:** 15 columns
- **Column/Variable Names:** `PacketCounter, SampleTimeFine, Quat_W, Quat_X, Quat_Y, Quat_Z, Acc_X, Acc_Y, Acc_Z, Gyr_X, Gyr_Y, Gyr_Z, Mag_X, Mag_Y, Mag_Z`
- **Data Type:** Float64 (Sensors) / Int64 (Counters/Timers)
- **Timestamp Info:** `SampleTimeFine` tracks microseconds since the sensor network synced/powered on.
- **Sampling Frequency:** ~120 Hz (Delta is consistently ~8333 microseconds).
- **Subject / Trial / Exercise ID:** Identified by the directory hierarchy (e.g., `subject 1` > `2. ElbowFlexCali`).
- **Missing / NaN / Duplicate:** 0

### 2. Optical Ground Truth Files
- **Filename Example:** `H_Joint_Angles_Aligned_FINAL` (Note: No `.mat` extension in the filesystem, but it is a MATLAB v5 file).
- **File Type:** MATLAB v5 (`.mat`)
- **File Size:** ~18.2 MB
- **Number of Samples:** Arrays match the varying lengths of the trials (e.g., `(1222, 3)`, `(3000, 3)`).
- **Internal Structure:** 
  - Root dict contains `j_aligned_eu`.
  - `j_aligned_eu` contains fields for different alignment protocols (`opto`, `nPose`, `func`, `manual`, `oneAxis`).
  - `opto` contains a struct array with joint fields: `HumeroThoracic, GlenoHumeral, ScapuloThoracic, Elbow, Wrist`.
  - `Elbow` contains `N` arrays corresponding to `N` continuous tasks, each shaped `(Samples, 3)`.
- **Data Type:** Float64
- **Timestamp Info:** `UNKNOWN — REQUIRES VERIFICATION`. The `.mat` file stores plain arrays without explicit time vectors. The title "Aligned" suggests row $i$ maps to synchronized timestamp $i$.
- **Sampling Frequency:** `UNKNOWN — REQUIRES VERIFICATION`. Optotrak typically runs at 100 Hz, but since it's aligned to the IMUs (120 Hz), resampling may have already occurred.
- **Missing / NaN:** 0 inside the arrays.

---

## B. Exact Signal Mapping Table

| Dataset Variable | Physical Meaning | Sensor | Axis/Direction | Unit | Sampling Rate | Target/Input |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Acc_X, Acc_Y, Acc_Z` | Linear Acceleration | RUA & RLA | X, Y, Z | `UNKNOWN — REQUIRES VERIFICATION` (Likely $m/s^2$) | 120 Hz | Input |
| `Gyr_X, Gyr_Y, Gyr_Z` | Angular Velocity | RUA & RLA | X, Y, Z | `UNKNOWN — REQUIRES VERIFICATION` (Likely $deg/s$) | 120 Hz | Input |
| `j_aligned_eu['opto']['Elbow'][:, 0]` | Flexion/Extension Angle | Optical MoCap | `UNKNOWN — REQUIRES VERIFICATION` | `UNKNOWN` (Likely Degrees) | `UNKNOWN` | Target |
| `j_aligned_eu['opto']['Elbow'][:, 1]` | Carrying Angle (Ab/Ad) | Optical MoCap | `UNKNOWN — REQUIRES VERIFICATION` | `UNKNOWN` (Likely Degrees) | `UNKNOWN` | Ignored |
| `j_aligned_eu['opto']['Elbow'][:, 2]` | Pronation/Supination Angle | Optical MoCap | `UNKNOWN — REQUIRES VERIFICATION` | `UNKNOWN` (Likely Degrees) | `UNKNOWN` | Target |

---

## C & D. Sampling Rates & Synchronization Method

- **Timestamps:** IMUs provide explicit microsecond timestamps (`SampleTimeFine`). MoCap ground truth files provide row-indexed matrices.
- **Clock Sharing:** The Xsens DOT network natively synchronizes the internal clocks of the RUA and RLA sensors.
- **Synchronization Method:** The presence of `H_Joint_Angles_Aligned_FINAL` alongside `I_tasks_split` strongly implies the dataset authors have *already* performed temporal alignment via dynamic time warping (DTW) or cross-correlation on the backend. Row 0 of the MoCap array corresponds to the start of the defined task interval in the IMU CSV. 
- **Unresolved Timing:** We must verify if the length of the `.mat` arrays strictly matches the number of rows in the `.csv` files for that specific task.

---

## E & F. Proposed Preprocessing Pipeline

### 1. Data Loading
- **Action:** Load RUA CSV, RLA CSV, and extract the corresponding `Elbow` matrices from `H_Joint_Angles_Aligned_FINAL` utilizing the task indices in `I_tasks_split`.
- **Why:** Separating the load logic ensures we don't mix up the unstructured `.mat` arrays with the wrong IMU task.

### 2. Timestamp Handling & Synchronization Verification
- **Action:** Convert `SampleTimeFine` to continuous relative seconds. Assert that the length of the IMU arrays equals the length of the ground-truth array.
- **Why:** Because the `.mat` arrays lack timestamps, we must rely on row-to-row matching. If lengths differ, it indicates dropped packets or differing sampling rates.

### 3. Resampling / Interpolation (If Required)
- **Action:** If lengths differ, apply 1D cubic spline interpolation to resample the MoCap array to match the IMU's 120Hz time vector.
- **Why:** Deep learning architectures require synchronized inputs and targets at identical time steps.

### 4. Sensor Signal Filtering
- **Action:** Apply a 4th-order Butterworth low-pass filter (cutoff ~10 Hz) to the accelerometer and gyroscope signals.
- **Why:** IMU signals contain high-frequency mechanical vibrations and electrical noise that act as artifacts, impeding the model from learning the actual smooth human kinematics.

### 5. Ground-Truth Alignment & Validation
- **Action:** Plot `Elbow[:, 0]` and `Elbow[:, 2]` against the raw IMU gyroscope integration to visually confirm the axes correspond to Flexion and Pronation, respectively, and deduce their units (Degrees vs. Radians).
- **Why:** We flagged these as `UNKNOWN — REQUIRES VERIFICATION`. Assuming ISB standard (X=Flexion, Z=Pronation) without visual confirmation leads to mathematically inverted or swapped targets.

### 6. Window Segmentation
- **Action:** Segment the continuous time-series into overlapping sliding windows (e.g., 2.0 seconds with 50% overlap).
- **Why:** Recurrent Neural Networks (LSTMs) or Transformers need finite context windows to process temporal dependencies without suffering from vanishing gradients or massive memory overhead.

### 7. Label Assignment
- **Action:** Assign the continuous angle matrix as the `Y_reg` (regression target) and the folder's task name as the `Y_cls` (classification target).

### 8. Train/Validation/Test Split Strategy
- **Action:** Leave-One-Subject-Out (LOSO) Cross-Validation.
- **Why:** Splitting randomly across all data will result in "subject leakage," where the model memorizes a specific person's unique kinematic signature. LOSO guarantees the model generalizes to *new* patients (which is vital for a clinical tool).

---

## G. Unresolved Issues

1. **Exact Column Mapping of `.mat` Elbow Angles:** We still need to run a correlation check to verify which of the 3 columns (0, 1, or 2) corresponds to Flexion and which to Pronation.
2. **Physical Units:** We must plot the data to verify if the IMUs exported $m/s^2$ vs. $g$, and if the MoCap exported Radians vs. Degrees, before applying standard normalization.
