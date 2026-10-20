# Longitudinal Rehabilitation Data Audit & Recovery Trend Prediction Assessment

**Date**: 2026-10-21  
**Project**: AI Elbow Rehabilitation System (`D:\me\project`)  
**Objective**: Determine whether genuine longitudinal patient recovery data exists in the raw datasets, and define the methodology for recovery trend tracking.

---

## 1. Executive Summary & Audit Conclusion

A systematic audit of all 13 datasets in `datasets/raw/` was conducted to evaluate whether multi-week, multi-session longitudinal patient rehabilitation data exists with repeated clinical outcome measures across recovery timelines.

### Audit Result:
* **No genuine multi-week longitudinal patient recovery trajectories exist in the raw datasets.**
* The available rehabilitation datasets (`StrokeRehab`, `JU-IMU`, `REHAB`, `PHYTMO`, `ULTra-MoCap`, `Optotrak vs DOT`, etc.) represent **single-session cross-sectional biomechanical recordings** or single-session task executions.
* No patient was tracked over weeks or months of recovery with serial clinical evaluations (such as Fugl-Meyer scores or ROM gains across physical therapy appointments).

### Policy and Ethical Determination:
In accordance with rigorous scientific integrity guidelines:
1. **No Synthetic or Fabricated Patient Timelines**: We will **NOT** synthesize fake multi-week patient recovery curves or train pseudo-predictive models on fabricated timelines.
2. **Descriptive Trajectory Tracking**: We implement robust **empirical descriptive progress tracking** based strictly on recorded patient sessions.
3. **Clinician-Defined ROM Targets**: We integrate customizable clinician-defined functional goals (e.g., standard functional elbow flexion target of $130.0^\circ$).
4. **Mandatory Clinical Disclaimers**: All trend lines, rate-of-change metrics ($\Delta\text{ROM}/\text{session}$), and projected milestones are explicitly labeled as **descriptive empirical fits**, with clear notifications that they **do not predict biological tissue healing or guarantee functional recovery**.

---

## 2. Dataset-by-Dataset Audit Matrix

| Dataset | Modality | Subject Cohort | Time Horizon / Protocol | Longitudinal Recovery Status |
| :--- | :--- | :--- | :--- | :--- |
| **StrokeRehab** | Wearable IMU | Healthy controls only in raw archive (`C4`–`C32`) | Single-session protocol (left/right arm trials) | ❌ Cross-sectional only; no stroke patients or multi-week visits. |
| **JU-IMU** | Dual/Multi IMU | 15 stroke survivors (`ND10`–`ND26`) | Single evaluation protocol (1 trial per exercise) | ❌ Cross-sectional only; 1 recording per drill per patient. |
| **REHAB** | Wearable IMU | 222 cross-sectional assessment trials | Single clinical assessment session per trial | ❌ Cross-sectional Fugl-Meyer assessment; no time-series of recovery visits. |
| **PHYTMO** | Dual IMU (Arm/Forearm) | 16 participants (A01–D04) | Single recording session per subject | ❌ Single session; tests execution quality (correct vs. compensatory). |
| **ULTra-MoCap** | Optical MoCap + Dual IMU | 20 participants (P01–P20) | Single session at varying speeds (normal, fast, slow) | ❌ Single session kinematics benchmark. |
| **Optotrak vs DOT** | Optical vs IMU | 20 participants (Subj 01–20) | Single laboratory calibration session (19 tasks) | ❌ Single session sensor validation benchmark. |
| **Smartwatch IMU** | Apple Watch IMU (20 Hz) | Healthy volunteers | Single laboratory session per participant | ❌ Cross-sectional single-session recordings. |
| **mm-fit / UCI Curl / EMG** | Smartwatch / Smartphone / EMG | Healthy gym participants | Single workout sessions | ❌ Fitness/biceps curl cross-sectional data. |
| **VIDIMU / ULF_in_ADL** | Video + IMU | Healthy participants | Single ADL task performances | ❌ Cross-sectional ADL benchmarks. |

---

## 3. Approved Implementation Architecture

### A. Persistent Session Database (`data/sessions/rehabilitation_sessions.db`)
An ACID-compliant SQLite relational database stores complete clinical rehabilitation session records:
* `session_id`: Unique identifier (`sess_YYYYMMDD_HHMMSS_xxxx`).
* `timestamp`: ISO-8601 UTC timestamp.
* `participant_id`: Pseudonymous patient identifier (e.g. `P012`, `SUBJ05`).
* `exercise_name`: Specific drill (e.g. `Elbow Flexion-Extension`, `Forearm Pronation-Supination`).
* `elbow_rom_deg`: Measured trial Range of Motion in degrees.
* `forearm_rom_deg`: Measured forearm rotational ROM in degrees (or `NULL` if not evaluated).
* `max_elbow_angle_deg` & `min_elbow_angle_deg`: Angular envelope.
* `repetition_count`: Number of completed movement cycles.
* `movement_smoothness_sparc`: Validated Spectral Arc Length smoothness metric.
* `movement_quality_class`: `Correct Execution` vs `Incorrect / Compensatory`.
* `movement_quality_confidence_pct`: Classifier confidence.
* `model_name` & `model_version`: Auditable AI provenance.
* `data_quality_status`: Input signal validity and sampling fidelity.
* `notes`: Clinician observations or clinical notes.

### B. Descriptive Progress Tracking & Trend Estimation
When $\ge 2$ sessions exist for a patient:
* **Empirical Rate of Change**: $\Delta\text{ROM} = \text{ROM}_{\text{latest}} - \text{ROM}_{\text{baseline}}$
* **Session-to-Session Velocity**: Slope $m = \frac{\Delta\text{ROM}}{\Delta\text{session}}$ or $\text{deg/week}$.
* **Descriptive Fit ($R^2$)**: Quantifies consistency of observed progress without extrapolating unobserved biological mechanisms.
* **Clinician-Defined Target**: Default $130.0^\circ$ (functional elbow flexion), customizable by the therapist per patient.
* **Proximity to Target**: $\frac{\text{ROM}_{\text{latest}}}{\text{Target ROM}} \times 100\%$.

### C. Data Review & Export
* Comprehensive session query API supporting participant filtering, date-range filtering, and exercise filtering.
* One-click export to **CSV** and **JSON** formats for integration with clinical electronic health records (EHR) and external audit.
