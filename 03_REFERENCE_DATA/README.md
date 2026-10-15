# AKIRA Reference Datasets

This directory contains standardized outputs and references extracted from publicly available datasets to support the development of the AKIRA Smart Post-Operative Rehabilitation System.

## Dataset Roles

* **WatchHand (`02_STANDARDIZED_DATA/WatchHand_standard.csv`)**:
  Raw single-IMU development data. This dataset contains true accelerometer and gyroscope measurements from a single wrist-worn sensor. It is useful for single-IMU sensor fusion and pipeline development.
  
* **3DCDataset (`03_REFERENCE_DATA/3DCDataset_exercise_reference.csv`)**:
  Wrist exercise reference. This dataset correctly categorizes the exact wrist ROM exercises required by AKIRA (Wrist Flexion, Wrist Extension, Radial Deviation, Ulnar Deviation, Pronation, Supination).

* **29666735 (`03_REFERENCE_DATA/wrist_kinematic_reference.csv`)**:
  Wrist kinematic reference. This dataset provides high-quality ground-truth wrist joint angles (kinematics in radians) recorded via an exoskeleton, serving as a mathematical reference for joint angle calculation.

## Important Disclaimer

**None of these datasets is the final AKIRA two-IMU dataset.**

AKIRA requires a synchronized dual-IMU configuration (Forearm + Hand). None of these public datasets contain this specific hardware configuration with raw 6-axis measurements. A custom dataset must be collected using the actual AKIRA hardware.
