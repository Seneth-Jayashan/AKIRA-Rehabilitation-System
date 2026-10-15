# Autoencoder Model Report

1. **Objective**: Build a SINGLE-IMU MOVEMENT ANOMALY DETECTION PROTOTYPE.
2. **Dataset**: WatchHand (Single wrist IMU prototype).
3. **Input features**: Accelerometer and Gyroscope statistical features (mean, std, rms, etc.).
4. **Preprocessing**: Handled missing values, standardized features with StandardScaler.
5. **Train/validation/test strategy**: Participant-aware splitting to prevent data leakage.
6. **Architecture**: 64-32-8-32-64 dense layers with ReLU activation.
7. **Hyperparameters**: Adam optimizer, MSE loss, 50 epochs (Early Stopping).
8. **Training procedure**: Model saved as autoencoder.keras.
9. **Reconstruction error**: Calculated using MSE.
10. **Threshold methodology**: 95th percentile of validation set reconstruction error.
11. **Anomaly results**: Mapped samples above threshold as ANOMALOUS.
12. **Visualizations**: Generated in 07_VISUALIZATIONS/autoencoder/.
13. **Limitations**: This is an algorithmic prototype only. It does not calculate true forearm-hand wrist orientation.
14. **Scientific interpretation**: Features causing highest MSE indicate irregular movement.
15. **Future dual-IMU implementation**: The final AKIRA model requires synchronized Forearm + Hand IMU.

## CRITICAL LIMITATIONS
1. Current model uses public single-IMU data.
2. It is NOT trained on post-ORIF patients.
3. It does NOT represent the final AKIRA dual-IMU model.
4. It does NOT calculate true forearm-hand relative wrist orientation.
5. It does NOT provide clinically validated rehabilitation scores.
6. The model is a technical anomaly-detection prototype.
7. Final AKIRA training requires synchronized: FOREARM IMU + HAND IMU
