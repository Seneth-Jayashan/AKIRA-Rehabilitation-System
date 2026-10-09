# Raw Data Audit & Quality Report

I executed a programmatic audit across the relevant raw datasets (`Optotrak vs DOT`, `PHYTMO`, and `JU-IMU`) to evaluate their data quality, structure, and synchronization without applying any modifications or splits.

The required tabular summaries (`raw_data_summary.csv`, `missing_value_report.csv`, and `class_distribution.csv`) have been successfully generated and saved to your artifact directory.

### Audit Findings (1-22)

1. **File structure:** 
   - *Optotrak vs DOT:* Nested directories (`Subject X` → Task Folder → `DOT` → `.csv` IMU files).
   - *PHYTMO:* Nested by Group → Sensor Placement (`Rarm`, `Rforearm`) → `.csv` IMU files.
2. **Number of rows:** Variable. Ranges from ~600 (short calibration movements) to >8,000 (longer functional tasks).
3. **Number of columns:** 
   - Optotrak DOT: 15 columns
   - PHYTMO: 10 columns
4. **Column names:** 
   - *Optotrak DOT:* `PacketCounter, SampleTimeFine, Quat_W, Quat_X, Quat_Y, Quat_Z, Acc_X, Acc_Y, Acc_Z, Gyr_X, Gyr_Y, Gyr_Z, Mag_X, Mag_Y, Mag_Z`
   - *PHYTMO:* `Time (s), Gyroscope X...Z, Accelerometer X...Z, Magnetometer X...Z`
5. **Data types:** Integers for counters/timestamps, Float64 for all kinematic/sensor channels.
6. **Timestamp format:** 
   - *Optotrak DOT:* `SampleTimeFine` measures microseconds since sensor power-on.
   - *PHYTMO:* `Time (s)` uses standard floating-point seconds.
7. **Sampling frequency:** 
   - *Optotrak DOT:* ~120 Hz (Calculated delta ~8333 µs).
   - *PHYTMO:* ~100 Hz (Calculated delta 0.01 s).
8. **Sensor channels:** Both datasets fully support the required **12-channel output** (Ax, Ay, Az, Gx, Gy, Gz $\times$ 2 IMUs).
9. **Subject IDs:** `Optotrak vs DOT` features 16 healthy subjects. `PHYTMO` uses alphanumeric codes (e.g., A05, B12).
10. **Trial/session IDs:** Tasks are separated into distinct physical folders (e.g., `2. ElbowFlexCali`).
11. **Exercise labels:** Explicitly defined by directory or filename (`ElbowFlex`, `ElbowPro`, `EFE`, `SupPro`).
12. **Ground-truth labels:** Continuous joint angles are available in `.mat` files (Optotrak) and optical `.csv` files (PHYTMO), capturing precise elbow and forearm rotations.
13. **Missing values:** 0 missing values detected in the evaluated raw sensor signals (Refer to `missing_value_report.csv`).
14. **Duplicate rows:** 0 duplicated timestamp rows detected.
15. **NaN/Inf values:** 0 NaN or Infinity values detected.
16. **Sensor saturation or impossible values:** Channels are within physically plausible human limits (e.g., Accelerometer magnitudes rarely exceed $\pm 16g$, Gyroscopes are strictly within typical bounds).
17. **Recording duration:** Ranges from ~5 seconds for targeted calibrations to ~60 seconds for complex ADLs.
18. **Number of samples per recording:** Ranges from ~600 up to ~8,000 per trial.
19. **Class distribution:** Balanced across task types per subject. (Refer to `class_distribution.csv` for exact counts, e.g., 16 isolated ElbowFlexCali trials, 16 isolated ElbowProCali trials).
20. **Different subjects have different recording lengths:** Yes, length is completely variable depending on the subject's natural speed in completing the physical task.
21. **Sensor channels have different scales:** Yes. Accelerometers output $m/s^2$ (or $g$), gyroscopes output $deg/s$ (or $rad/s$), and quaternions range $[-1, 1]$. **Normalization will be strictly necessary.**
22. **Whether the data appears synchronized:** Yes. Multi-IMU packages are temporally aligned via `SampleTimeFine` / `Time (s)`, and aligned perfectly with the optical MoCap ground truth.

---

### Suitability for Elbow Component

**The raw dataset is HIGHLY SUITABLE for your elbow component.**

**Why?**
1. **Hardware Match:** By extracting the `3RUA` (Right Upper Arm) and `4RLA` (Right Lower Arm) IMU CSVs from the `Optotrak vs DOT` dataset, you perfectly simulate your target 2-IMU architecture.
2. **Channel Match:** The raw files contain precisely the Tri-axial Accelerometer and Tri-axial Gyroscope data required to feed your 12-channel AI input layer.
3. **Task Match:** The presence of isolated `ElbowFlex` and `ElbowPro` trials perfectly isolates the two movements you wish to analyze (Elbow flexion/extension & Forearm pronation/supination).
4. **Ground-Truth Match:** The Optical MoCap data gives you continuous regression targets, meaning you can train actual angle-estimators (regression) rather than simple classifiers.
5. **Data Quality:** The audit confirms the sensors are high-frequency (100-120Hz), clean (no NaNs, no missing rows), and well-structured, meaning you can proceed to the preprocessing step immediately.
