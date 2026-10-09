# Requirements Matrix: Intended Methodology vs. Actual Dataset

This matrix rigorously evaluates the **Optotrak vs DOT** dataset (the only dataset in the workspace perfectly matching your hardware specifications) against your proposed research methodology for **Olecranon ORIF Rehabilitation**.

| Research Requirement | Required Data | Dataset Provides It? | Evidence | Implementation Possibility | Limitation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Upper-arm IMU** | 6 DoF IMU on upper arm | **YES** | `3RUA` CSV files in `DOT` folder | Direct raw input | None |
| **2. Forearm IMU** | 6 DoF IMU on forearm | **YES** | `4RLA` CSV files in `DOT` folder | Direct raw input | None |
| **3. Accelerometer** | Tri-axial acceleration | **YES** | `Acc_X, Acc_Y, Acc_Z` columns | Direct feature extraction | None |
| **4. Gyroscope** | Tri-axial angular velocity | **YES** | `Gyr_X, Gyr_Y, Gyr_Z` columns | Direct feature extraction | None |
| **5. 12-channel synchronized input** | 2 IMUs $\times$ 6 channels temporally aligned | **YES** | `SampleTimeFine` exists in both RUA & RLA files | High; align via timestamps | Minor interpolation required if packets drop |
| **6. Elbow flexion/extension** | Kinematic data for flexion/extension | **YES** | `2. ElbowFlexCali` task folders | High; supervised training | None |
| **7. Forearm pronation/supination**| Kinematic data for rotation | **YES** | `3. ElbowProCali` task folders | High; supervised training | None |
| **8. Elbow angle ground truth** | Continuous angle targets | **YES** | Optical MoCap `H_Joint_Angles_Aligned_FINAL.mat` | Supervised regression | Must align MAT arrays with CSV timestamps |
| **9. Forearm rotation ground truth**| Continuous angle targets | **YES** | Optical MoCap `H_Joint_Angles_Aligned_FINAL.mat` | Supervised regression | Optical forearm tracking can suffer skin artifact |
| **10. Exercise labels** | Categorical labels for tasks | **YES** | Distinct directory names (e.g., `task 1`, `ElbowFlex`) | Supervised classification | Labels apply to entire trials, not granular |
| **11. Repetition labels** | Start/End time of individual reps | **NOT AVAILABLE** | No annotation files or markers found | Impossible directly via ML | Requires mathematical peak-detection heuristic |
| **12. ROM** | Max/Min angle scalars | **NOT AVAILABLE** | No pre-calculated ROM scalars provided | Impossible directly via ML | Must be derived post-prediction from angle output |
| **13. Angular velocity** | Rate of change of elbow angle | **NOT AVAILABLE** | Ground truth only provides position/angles | Impossible directly via ML | Must be mathematically differentiated from angle |
| **14. Rotation velocity** | Rate of change of forearm rotation | **NOT AVAILABLE** | Ground truth only provides position/angles | Impossible directly via ML | Must be mathematically differentiated from angle |
| **15. Movement trajectory** | 3D spatial position of joints | **YES** | `C_JointData_optotrack` files contain coordinates | Supervised positional regression | High complexity to map 6-DoF IMU to 3D space |
| **16. Movement quality** | Clinical scores (DASH, MEPS, etc.) | **NOT AVAILABLE** | `Info.txt` only has Age/Sex/Weight/Height | Impossible | No clinical basis for movement quality training |
| **17. Longitudinal sessions** | Recordings over weeks/months | **NOT AVAILABLE** | Dataset is cross-sectional (1 session per subject) | Impossible | Cannot model recovery dynamics over time |
| **18. Rehabilitation progress labels**| Stages of post-operative recovery | **NOT AVAILABLE** | Subjects are healthy; no pathology labels exist | Impossible | Cannot predict a pathology it has never seen |

---

### Scientifically Defensible ML Task Recommendation

Based on a strict evaluation of the raw data, it is **scientifically indefensible** to train models for *Movement Quality* or *Rehabilitation Progress Tracking* directly. You cannot train an AI to evaluate clinical rehabilitation progress if it has never been exposed to recovering patients, specific Olecranon ORIF compensatory mechanics, or validated clinical scores.

However, the dataset provides excellent, high-fidelity synchronized IMU data perfectly paired with gold-standard continuous Optical MoCap angle ground truths.

**The most scientifically defensible and robust ML architecture to train is:**
### Multi-Task Kinematic Estimation (Continuous Joint Angle Regression + Task Classification)

**Task 1 (Regression Component):** 
Train a Deep Sequence Model (e.g., BiLSTM, TCN, or Transformer) to map the 12-channel IMU time-series strictly to the continuous Elbow Flexion/Extension and Forearm Pronation/Supination angles provided by the Optotrak MoCap.

**Task 2 (Classification Component):**
Share the feature-extraction layers of the regression model to simultaneously classify the current exercise/task type (e.g., ADL vs. Flexion vs. Pronation).

**Handling the Missing Components:**
Once this foundational kinematic estimation model is trained and validated, you can **mathematically derive** the missing features during the actual clinical application without needing ML to "guess" them:
- **ROM:** Calculated by taking the min/max of the model's continuous angle predictions.
- **Angular / Rotation Velocity:** Calculated by taking the first derivative of the model's angle predictions over time.
- **Repetition Counting:** Calculated by applying a standard peak-detection/zero-crossing algorithm to the model's continuous angle predictions.
