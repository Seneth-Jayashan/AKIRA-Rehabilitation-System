# Preprocessing Report

1. Dataset used: WatchHand Standardized
2. Number of participants: 6 (sampled)
3. Number of trials: 10 (sampled)
4. Sensor type: Smartwatch IMU
5. Sampling rate: ~50Hz (estimated)
6. Accelerometer availability: YES
7. Gyroscope availability: YES
8. Missing data: Handled by dropping incomplete rows. No fabrication.
9. Filtering method: 4th-order Butterworth Lowpass, 5Hz cutoff.
10. Calibration method: Pseudo-calibration (subtracting initial 50-sample mean for gyroscope).
11. Segmentation method: Gyroscope magnitude thresholding.
12. Repetition detection method: Peak detection on gyroscope magnitude.
13. Extracted features: Time domain (mean, std, rms, max, min, duration).
14. Orientation method: Basic gyro integration (Single-IMU prototype).
15. 3D trajectory method: Double integration mock (Single-IMU prototype).
16. Limitations: SINGLE IMU ONLY. Extreme drift. Does not measure wrist joint angle.
17. What can be transferred to AKIRA: The filtering, segmentation, and feature extraction math logic.
18. What requires the future dual-IMU dataset: Autoencoder training, true wrist ROM calculation, actual spatial trajectories.

FINAL DUAL-IMU AKIRA WRIST MODEL: NOT YET TRAINED
REQUIRED NEXT DATA: SYNCHRONIZED FOREARM + HAND IMU RECORDINGS
