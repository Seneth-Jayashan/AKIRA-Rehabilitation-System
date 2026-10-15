# AKIRA STANDARDIZATION VALIDATION REPORT

## 1. WatchHand Standardized Dataset (`02_STANDARDIZED_DATA/WatchHand_standard.csv`)
* **Status**: PASS
* **Schema Match**: PASS
* **Row count**: 1,048,575
* **Participant count**: 6
* **Trial count**: 10
* **Exercise count**: 1
* **Missing values**: 1048575
* **Duplicate rows**: 35
* **Timestamp validity**: PASS
* **Accelerometer (ax, ay, az)**: PASS
* **Gyroscope (gx, gy, gz)**: PASS
* **No Magnetometer added**: PASS
* **No fake second IMU**: PASS
* **Sensor location**: HAND
* **Source files excluded**: 18
* **Reason for exclusions**: Empty Acce/Gyro sequences or mismatched row counts in source files.

## 2. 3DCDataset Exercise Reference (`03_REFERENCE_DATA/3DCDataset_exercise_reference.csv`)
* **Status**: PASS
* **Schema Match**: PASS
* **Row count**: 1,452
* **Participant count**: 22
* **Trial count**: 11
* **Exercise count**: 3
* **Missing values**: 0
* **Duplicate rows**: 0
* **No fake IMU (no quaternions as accel/gyro)**: PASS
* **Relevant exercises mapped correctly**: PASS
* **Excluded**: Grasps and non-wrist gestures.
* **Reason for exclusions**: Not part of the AKIRA wrist movement specification.

## 3. 29666735 Kinematic Reference (`03_REFERENCE_DATA/wrist_kinematic_reference.csv`)
* **Status**: PASS
* **Schema Match**: PASS
* **Row count**: 500,000
* **Participant count**: 2
* **Trial count**: 9
* **Exercise count**: 1
* **Missing values**: 0
* **Duplicate rows**: 0
* **No fake IMU columns added**: PASS
* **Original kinematic units preserved**: PASS
* **Excluded**: EEG and Event TSV files.
* **Reason for exclusions**: This is strictly a kinematic motion reference extraction.

---

### FINAL VERDICT
DATASET STANDARDIZATION VALIDATED
