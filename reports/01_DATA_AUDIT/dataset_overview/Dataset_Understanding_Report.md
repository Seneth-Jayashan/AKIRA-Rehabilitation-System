# Dataset Understanding Report

Based on a thorough analysis of the project files provided in the `datasets/raw/` workspace directory, here is a detailed breakdown of the available data and how it aligns with your research methodology for **Olecranon ORIF Rehabilitation** using a **two-IMU architecture**.

### 1. Files & Datasets Found
I discovered a comprehensive collection of 13 public datasets in your workspace. 
**Dataset Names:**
1. `Optotrak vs DOT`
2. `PHYTMO`
3. `JU-IMU`
4. `MoCap and IMU Dataset for Upper Limb Joint Angle Estimation`
5. `VIDIMU`
6. `Smartwatch IMU Dataset for Upper-Limb Movement Ana`
7. `EMG elbow dataset`
8. `ULTra-MoCap`
9. `StrokeRehab`
10. `REHAB`
11. `UCI Biceps Curl`
12. `mm-fit`
13. `ULF_in_ADL`

### 3. File Formats
- **Time-series Sensor Data:** `.csv`, `.raw`, `.txt`
- **MoCap/Kinematics:** `.c3d` (3D coordinate), `.trc` (track row column), `.mot` (OpenSim motion), `.sto` (OpenSim states/kinematics), `.osim` (OpenSim models)
- **Metadata/Aggregated:** `.h5`, `.json`, `.md`, `.npy`

### 4. Subjects, Sessions, and Trials
- **Optotrak vs DOT:** 16 subjects. Includes calibration trials (e.g., Elbow Flexion, Elbow Pronation) and 10 functional tasks per subject.
- **JU-IMU:** 9 Stroke patients and various healthy controls, multiple ROM (Range of Motion) trials per subject.
- **PHYTMO:** Multiple subjects (Group A, B, C...) performing exercises (EAH, EFE, SQZ) in multiple trials.

### 5 & 6. Sensors Used & Sensor Placement
- **Optotrak vs DOT:** Xsens DOT (IMU) and Optotrak (Optical MoCap). IMUs are placed on the **Right Upper Arm (RUA)**, **Right Lower Arm / Forearm (RLA)**, Right Scapula, Right Wrist, and Trunk.
- **PHYTMO:** Wearable IMUs and OptiTrack (Optical MoCap). IMUs placed on **Upper Arm (Rarm)** and **Forearm (Rforearm)**.
- **Smartwatch IMU:** Dual Apple Watches (Left and Right Wrists).
- **JU-IMU:** 5 IMU sensors across the upper extremity.
- **VIDIMU:** 5 IMUs focused primarily on the lower body / pelvis.

### 7. Sampling Frequency
- **Xsens DOT IMUs (Optotrak vs DOT):** 120 Hz (Derived from `SampleTimeFine` stepping at ~8333 microseconds).
- **Optotrak (MoCap):** Typically 100 Hz.
- **PHYTMO IMUs:** 100 Hz (0.01s timestamp increments).

### 8. Available Channels
The datasets featuring dedicated IMUs (like Xsens DOT in `Optotrak vs DOT` and `PHYTMO`) output **Quaternions, Accelerometer (Ax, Ay, Az), Gyroscope (Gx, Gy, Gz), and Magnetometer (Mx, My, Mz)**. 
By extracting only the Upper Arm and Forearm IMUs, you can directly build your intended **synchronized 12-channel time-series signal** (6 channels $\times$ 2 IMUs).

### 9 & 10. Available Labels & Exercises
- **Continuous Labels:** Ground truth joint angles (calculated from optical MoCap data in OpenSim).
- **Categorical Labels:** Exercise/Task types based on file and folder nomenclature.
- **Exercises present:** Elbow Flexion/Extension (`ElbowFlexCali`, `EFE`, `ElbFlex`), Forearm Pronation/Supination (`ElbowProCali`, `SupPro`), Shoulder Rotation/Abduction/Flexion, Wrist Flexion/Deviation, and various functional ADL (Activities of Daily Living) tasks.

### 11 & 12. Elbow Angle & Forearm Rotation Ground Truth
- **Yes.** The `Optotrak vs DOT` and `PHYTMO` datasets feature optical motion capture alongside the IMUs. These optical markers provide high-fidelity, continuous ground truth for both **elbow angle (flexion/extension)** and **forearm rotation (pronation/supination)**.

---

### 13. Does the dataset match your two-IMU architecture?
**YES.** Specifically, the **`Optotrak vs DOT`** and **`PHYTMO`** datasets are **perfect matches** for your architecture. 
While they include additional sensors (like the trunk, scapula, or wrist), you can isolate the **Right Upper Arm (RUA / Rarm)** and **Right Lower Arm (RLA / Rforearm)** data to exactly mirror your proposed 2-IMU hardware setup.

### 14. Feasible Implementations (What you CAN build right now)
Using `Optotrak vs DOT` and/or `PHYTMO`, you can directly implement:
1. **Elbow-angle regression:** Training a model mapping the 12-channel IMU data to the continuous optical MoCap elbow angle.
2. **Forearm-rotation regression:** Mapping IMU data to the optical pronation/supination angle.
3. **Exercise classification:** Using the structured trial files (e.g., flex vs. pro/sup vs. resting) to train a task classifier.
4. **ROM (Range of Motion) estimation:** Derivable by calculating the min/max of your regression outputs.
5. **Angular velocity / Rotation velocity:** Directly derivable from your regressed angle time-series or IMU gyroscope kinematics.
6. **Repetition counting:** Implementable by applying peak-detection on the predicted continuous angles.

### 15. Limitations (What you CANNOT yet implement)
The datasets lack the required information to directly implement:
1. **Olecranon ORIF specific pathology:** The datasets consist of healthy subjects (and stroke patients in `JU-IMU` / `StrokeRehab`). They do not contain specific post-operative Olecranon ORIF movement patterns, compensatory mechanics, or cast/brace-induced artifacts.
2. **Longitudinal rehabilitation tracking:** These datasets are cross-sectional (single session per subject). Tracking recovery over weeks/months requires longitudinal data of the same patient.
3. **Movement quality (Clinical):** While you can calculate smoothness (e.g., spectral arc length), establishing a clinically validated "Quality Score" for post-op patients requires clinical baseline scores (like DASH or MEPS) which are not included in these public sets.
