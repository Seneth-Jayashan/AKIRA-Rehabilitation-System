# AKIRA Wrist Component — Model 4 (TCN) Feasibility Audit

**Date:** 2026-10-10
**Task:** Investigate the feasibility of training a Temporal Convolutional Network (TCN) using existing datasets.

## 1. Dataset Availability & Signal Verification
An inspection of the existing datasets within `01_DATASETS/RAW/` reveals the following constraints:
- **WatchHand Dataset:** Contains ordered temporal sequences (Raw Accelerometer/Gyroscope), but lacks ground-truth rehabilitation exercise labels.
- **Wrist Kinematics Dataset:** Contains reference wrist angles (flexion/extension, radial/ulnar), but lacks paired raw IMU sensor data required to map signals to angles.
- **3DCDataset (`3DCDataset-main.zip`):** Contains ordered temporal sequences, discrete ground-truth labels (Neutral, Wrist Flexion, Radial Deviation, Ulnar Deviation), and distinct `participant_id`s. **Crucially, it only provides spatial Quaternions (W, X, Y, Z) from a single sensor.**
- **Dual Synchronized Sensors:** None of the existing datasets contain synchronized Forearm + Hand IMUs.

## 2. Assessment of Possible Tasks
Based on the available data, the proposed TCN applications are evaluated as follows:

### A. Wrist-Angle Estimation — NO-GO
*Requirements:* Continuous temporal IMU sequences strictly paired with synchronized reference joint angles.
*Status:* The required paired dataset does not exist. The `Wrist Kinematics Dataset` has angles but no IMU data; `3DCDataset` has IMU orientations but no continuous angle references.

### B. Movement-Quality or Rehabilitation-Outcome Prediction — NO-GO
*Requirements:* Clinical recovery scores, post-operative DRF data, or annotated execution errors.
*Status:* Existing data comprises healthy subjects performing discrete, idealized gestures.

### C. Quaternion-Sequence Classification — TECHNICALLY POSSIBLE
*Requirements:* Pre-segmented temporal sequences with discrete labels.
*Status:* `3DCDataset` supports this (previously used for Models 2 & 3).

## 3. TCN Value Assessment (vs. 1D CNN)
If we were to pursue Task C (Quaternion-Sequence Classification), would a TCN add scientific value over the existing 1D CNN (Model 3)?

A TCN uses dilated causal convolutions to achieve massive receptive fields, making it highly effective for continuous forecasting or extremely long sequences where historical context matters. 
However:
1. **Sequence Length:** The `3DCDataset` sequences are relatively short and pre-segmented (padded to 250 steps). Standard 1D CNN convolutions easily cover this receptive field. 
2. **The Fundamental Feature Flaw:** As proven by Model 3 (47.05% Accuracy) compared to Model 2 (52.7% Accuracy), the primary barrier to generalization is not the *temporal architecture*, but the *input features*. Absolute single-sensor quaternions cannot capture relative wrist kinematics across different participants' varying resting arm positions. 
A TCN would suffer from the exact same feature-representation flaw as the 1D CNN. Swapping the architecture will not solve the physical requirement for synchronized dual-IMU relative kinematics.

## 4. Fair Evaluation Proposal
If a TCN were to be trained against these recommendations, it must strictly adhere to:
- **Input:** `(None, 250, 4)` raw temporal quaternion sequences from `3DCDataset`.
- **Validation Split:** The exact `GroupShuffleSplit` (grouped by `participant_id`, `random_state=42`) used for Models 2 and 3, holding out the identical 5 test participants to prevent leakage.
- **Target:** The same 4 target classes (Neutral, Radial Deviation, Wrist Flexion, Ulnar Deviation).

## 5. Final Decision: NO-GO
**Decision:** NO-GO for training Model 4 (TCN).
**Reasoning:** While technically possible to train another classifier on the `3DCDataset` quaternions, doing so lacks scientific defensibility. A TCN adds no demonstrable value over the existing 1D CNN for short, pre-segmented absolute quaternion sequences. The true bottleneck is the absence of a dual-IMU kinematic dataset. Training a TCN would merely fabricate the illusion of progress without addressing the fundamental physical constraints required by the final AKIRA system. No model will be trained.
