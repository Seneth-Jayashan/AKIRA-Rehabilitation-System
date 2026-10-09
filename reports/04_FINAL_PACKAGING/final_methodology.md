# Thesis Methodology Document

1. **Research objective**: Subject-independent kinematic estimation and exercise recognition on Edge AI.
2. **Dataset**: Healthy subject dataset containing RUA, RLA IMUs, and Optotrak ground truth.
3. **Subjects**: 15 valid subjects (Subject 8 excluded due to data corruption).
4. **Sensors**: RUA and RLA IMUs (Accelerometer + Gyroscope).
5. **Input representation**: 12-channel continuous input buffer.
6. **Ground truth**: Optotrak kinematics.
7. **Signal synchronization**: Sample-to-sample aligned prior to training.
8. **Preprocessing**: 10Hz zero-phase 4th order Butterworth low-pass filter.
9. **Subject-independent splitting**: 10 Train, 3 Validation, 2 Test.
10. **Windowing**: 240 samples (2.0s at 120Hz).
11. **Target representation**: Forearm (sin/cos), Elbow (degrees).
12. **TCN architecture**: 5 blocks, dilations (1,2,4,8,16), 32 filters.
13. **Multi-task learning**: Shared latent representation feeding 3 independent heads.
14. **Loss functions**: Forearm: MSE, Elbow: Masked Huber, Cls: CCE.
15. **Training protocol**: AdamW, DWA weighting, patience 20.
16. **Baseline models**: LSTM, Linear.
17. **Evaluation metrics**: MAE, RMSE, Macro F1.
18. **Rehabilitation metric derivation**: ROM, Velocity, Repetitions via peak-finding.
19. **Edge-AI considerations**: < 150KB size, highly parallelizable 1D depth.
20. **Limitations**: Non-clinical cohort.
