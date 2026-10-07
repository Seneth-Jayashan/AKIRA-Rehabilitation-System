# AKIRA Rehabilitation System - Dataset Inventory

## 1. AnkleImage (ScientificData)
- **Source**: ScientificData zip archive, unzipped into Tasks 1-5.
- **Number of subjects**: At least 4 (Sub_A01 to Sub_A04 in Task 1).
- **Number of recordings**: Multiple tasks (Task 1, Task 3, Task 5 observed).
- **Sensor type**: IMU / Ankle-specific.
- **Sensor location**: Lower leg/tibia and foot (inferred from proposal).
- **Sampling frequency**: TBD (Extracting from .mat/.xlsx required).
- **Accelerometer?**: Yes.
- **Gyroscope?**: Yes.
- **Magnetometer?**: TBD.
- **Orientation?**: TBD.
- **Timestamp?**: TBD.
- **Exercise/activity labels?**: Contains "Manual labeling" (.xlsx).
- **Subject labels?**: Yes, directory-based (`Sub_A01`, etc.).
- **Trial/session labels?**: Grouped by Tasks.
- **File format**: `.xlsx`, `.mat`.
- **Missing values?**: TBD (Requires deeper data parsing).

## 2. GAITEX
- **Source**: `data.zip` archive.
- **Number of subjects**: TBD (Data structured under `data/austra/gwo/`).
- **Number of recordings**: TBD.
- **Sensor type**: Optical markers / IMU (kinematic models).
- **Sensor location**: Full body (humerus, femur, tibia, foot, skull, etc. modelled).
- **Sampling frequency**: TBD.
- **Accelerometer?**: TBD (Likely computed/inverse kinematics).
- **Gyroscope?**: TBD.
- **Magnetometer?**: TBD.
- **Orientation?**: Yes (Inverse kinematics).
- **Timestamp?**: Yes (Time-series TRC).
- **Exercise/activity labels?**: TBD.
- **Subject labels?**: TBD.
- **Trial/session labels?**: TBD.
- **File format**: `.osim`, `.trc`, `.vtp` (OpenSim formats).
- **Missing values?**: TBD.

## 3. SDALLE
- **Source**: `Dataset_SDALLE.rar`.
- **Number of subjects**: TBD.
- **Number of recordings**: TBD.
- **Sensor type**: Motion/IMU.
- **Sensor location**: TBD.
- **Sampling frequency**: TBD.
- **Accelerometer?**: TBD.
- **Gyroscope?**: TBD.
- **Magnetometer?**: TBD.
- **Orientation?**: TBD.
- **Timestamp?**: TBD.
- **Exercise/activity labels?**: TBD.
- **Subject labels?**: TBD.
- **Trial/session labels?**: TBD.
- **File format**: TBD (Packed in RAR).
- **Missing values?**: TBD.
*(Note: Full extraction requires unrar utility or extraction in a compatible environment).*

## 4. PHYTMO
- **Source**: Unzipped directory containing `inertial` and `optical` folders.
- **Number of subjects**: Subjects denoted by prefixes (e.g., A01).
- **Number of recordings**: Multiple CSV files per subject (e.g. `A01GAT0_1.csv`).
- **Sensor type**: Inertial and Optical.
- **Sensor location**: Examples include `Lshin` (Left shin).
- **Sampling frequency**: Inferred ~100Hz from timestamp steps (e.g., 11.556, 11.566).
- **Accelerometer?**: Yes (X, Y, Z in `g`).
- **Gyroscope?**: Yes (X, Y, Z in `deg/s`).
- **Magnetometer?**: Yes (X, Y, Z in `uT`).
- **Orientation?**: TBD (Can be calculated).
- **Timestamp?**: Yes (`Time (s)` column).
- **Exercise/activity labels?**: Encoded in filenames (e.g., `GAT0`, `GAT1`).
- **Subject labels?**: Encoded in filenames (`A01`).
- **Trial/session labels?**: Encoded in filenames (suffix `_1`, `_2`).
- **File format**: `.csv`.
- **Missing values?**: Clean at first glance, requires full scan.

## 5. MM-Motion
- **Source**: `subject01.zip` to `subject16.zip`.
- **Number of subjects**: 16.
- **Number of recordings**: 10 poses/activities per subject (e.g., `pose01` - `pose10`).
- **Sensor type**: High-density motion sensors (48 channels per side).
- **Sensor location**: Right (R1-R48) and Left (L1-L48) sides of the body.
- **Sampling frequency**: High (based on timestamp fractions).
- **Accelerometer?**: Yes (Channels in `g`).
- **Gyroscope?**: TBD.
- **Magnetometer?**: TBD.
- **Orientation?**: TBD.
- **Timestamp?**: Yes (`f_timestamp` formatted as `YYYY/MM/DD HH:MM:SS.ms`).
- **Exercise/activity labels?**: Encoded in directories (`pose01` - `pose10`).
- **Subject labels?**: Yes (`subject01` - `subject16`).
- **Trial/session labels?**: Internal subdirectories (e.g., `pose01\1`).
- **File format**: `.csv`.
- **Missing values?**: Contains many empty trailing columns (Unnamed: 97 - 193) which need filtering during preprocessing.
